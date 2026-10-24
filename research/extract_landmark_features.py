"""Step 2: stream LandmarkDataset.csv once and write one row of face features per recording.

The file is 18 GB, but we only need 17 of its 511 landmarks, so it is read line by line straight
from Zenodo (or from a local copy) and never stored. Memory stays small; the run is limited by
download speed. Standard library + numpy only.

    # whole dataset, streamed from Zenodo
    python research/extract_landmark_features.py
    # from a local copy, or check the column mapping on the probe sample first
    python research/extract_landmark_features.py --source research/data/landmark_head.csv --check

Output: research/data/video_features.csv (video, participant, label, frames, features...).
A partial run (--max-bytes) only covers the recordings at the start of the file, which is a
biased subset; the output says so and the trainer should not be trusted on it.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from tilt.face_features import ALL_WINDOW_FEATURES, USED_LANDMARKS, frame_primitives, window_features  # noqa: E402

URL = "https://zenodo.org/records/17421590/files/LandmarkDataset.csv?download=1"
OUT = ROOT / "research" / "data" / "video_features.csv"
ID_COLS = ("Video_Name", "Frame", "Class")
N_FACE, N_IRIS, N_POSE = 468, 10, 33


# ---- column mapping -----------------------------------------------------------------------------

def _named_index(header: list[str], landmark: int, axis: str) -> int | None:
    lower = {h.strip().lower(): i for i, h in enumerate(header)}
    if landmark < N_FACE:
        i = landmark
        names = [f"face_{i}_{axis}", f"face_{axis}_{i}", f"{axis}_{i}", f"{axis}{i}", f"landmark_{i}_{axis}",
                 f"lm_{i}_{axis}", f"face{i}_{axis}", f"face_landmark_{i}_{axis}", f"facemesh_{i}_{axis}"]
    else:
        j = landmark - N_FACE
        names = [f"iris_{j}_{axis}", f"iris_{axis}_{j}", f"iris{j}_{axis}", f"iris_landmark_{j}_{axis}",
                 f"face_{landmark}_{axis}", f"{axis}_{landmark}", f"landmark_{landmark}_{axis}"]
    for n in names:
        if n in lower:
            return lower[n]
    return None


def resolve_columns(header: list[str], positional: str = "auto") -> tuple[dict, dict[int, tuple[int, int]], str]:
    """Return (id column indices, {landmark: (x_col, y_col)}, how it was resolved)."""
    hdr = [h.strip() for h in header]
    ids = {}
    for c in ID_COLS:
        if c not in hdr:
            raise SystemExit(f"column {c!r} not found; first columns are {hdr[:8]}")
        ids[c] = hdr.index(c)
    by_name = {lm: (_named_index(hdr, lm, "x"), _named_index(hdr, lm, "y")) for lm in USED_LANDMARKS}
    if positional != "force" and all(x is not None and y is not None for x, y in by_name.values()):
        return ids, by_name, "by column name"
    if positional == "never":
        missing = [lm for lm, (x, y) in by_name.items() if x is None or y is None]
        raise SystemExit(f"could not find landmark columns by name for {missing[:5]}...; header starts {hdr[:12]}")
    coords = [i for i, h in enumerate(hdr) if h not in ID_COLS]
    expected = 3 * (N_FACE + N_IRIS + N_POSE)
    if len(coords) != expected:
        raise SystemExit(f"{len(coords)} coordinate columns, expected {expected} for positional mapping; "
                         f"header starts {hdr[:12]}")
    # Assumed order: 468 face, then 10 iris, then 33 pose landmarks, each as x, y, z. Verified by --check.
    mapping = {lm: (coords[3 * lm], coords[3 * lm + 1]) for lm in USED_LANDMARKS}
    return ids, mapping, "by position (face, iris, pose; x,y,z each)"


def geometry_ok(points: dict[int, tuple[float, float]]) -> bool:
    """Sanity check that the mapping is right: a face that is the right way up and the right way round."""
    x = {k: v[0] for k, v in points.items()}
    y = {k: v[1] for k, v in points.items()}
    return (y[10] < y[1] < y[152]                      # forehead above nose above chin
            and y[105] < y[159] and y[334] < y[386]    # brows above eyes
            and y[13] <= y[14]                         # upper lip above lower lip
            and min(x[33], x[263]) < x[1] < max(x[33], x[263]))  # nose between outer eye corners


# ---- participant ids ----------------------------------------------------------------------------

def participant_of(video: str, pattern: str | None) -> str:
    if pattern:
        m = re.search(pattern, video)
        if not m:
            raise SystemExit(f"--participant-regex did not match video name {video!r}")
        return m.group(1)
    return video  # unknown until the naming scheme is confirmed; the trainer refuses 1 video per person


# ---- streaming ----------------------------------------------------------------------------------

def open_source(source: str):
    if source.startswith("http"):
        req = urllib.request.Request(source, headers={"User-Agent": "tilt-oxhack-research/1.0"})
        resp = urllib.request.urlopen(req, timeout=300)
        total = int(resp.headers.get("Content-Length") or 0)
        return io.TextIOWrapper(resp, encoding="utf-8", newline=""), total
    p = Path(source)
    return open(p, encoding="utf-8", newline=""), p.stat().st_size


def run(source: str, out: Path, participant_regex: str | None, max_bytes: int | None, check: bool,
        positional: str) -> dict:
    stream, total = open_source(source)
    reader = csv.reader(stream)
    header = next(reader)
    ids, cols, how = resolve_columns(header, positional)
    print(f"{len(header)} columns; landmark columns resolved {how}")

    frames: dict[str, list] = defaultdict(list)
    label: dict[str, int] = {}
    bad_geometry = rows = 0
    read_bytes = len(",".join(header))
    t0 = last = time.time()
    partial = False
    for row in reader:
        if len(row) != len(header):
            break  # truncated final line of a partial download
        rows += 1
        read_bytes += sum(map(len, row)) + len(row)
        video = row[ids["Video_Name"]]
        lab = int(float(row[ids["Class"]]))
        if label.setdefault(video, lab) != lab:
            raise SystemExit(f"recording {video} has more than one label")
        try:
            pts = {lm: (float(row[cx]), float(row[cy])) for lm, (cx, cy) in cols.items()}
        except ValueError:
            continue  # face not detected in this frame (empty cells)
        if any(v != v for xy in pts.values() for v in xy):
            continue
        if not geometry_ok(pts):
            bad_geometry += 1
            continue
        frames[video].append(frame_primitives(pts))
        if time.time() - last > 15:
            last = time.time()
            rate = read_bytes / (last - t0) / 1e6
            pct = f" {100 * read_bytes / total:.1f}%" if total else ""
            print(f"  {read_bytes / 1e9:.2f} GB{pct}, {rows:,} frames, {len(label)} recordings, {rate:.1f} MB/s")
        if max_bytes and read_bytes >= max_bytes:
            partial = True
            break
    stream.close()

    usable = rows - bad_geometry
    summary = {"source": source, "column_mapping": how, "frames_read": rows, "frames_failing_geometry": bad_geometry,
               "recordings": len(label), "partial_read": partial, "bytes_read": read_bytes}
    print(json.dumps(summary, indent=2))
    if check:
        ok = rows > 0 and bad_geometry / max(rows, 1) < 0.05
        print("Column mapping looks right." if ok else "Column mapping looks WRONG: most frames fail the face-geometry check.")
        sample = sorted(label)[:10]
        print("Example video names:", sample)
        return summary | {"mapping_ok": ok}
    if usable <= 0:
        raise SystemExit("no usable frames")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "participant", "label", *ALL_WINDOW_FEATURES, "frames"])
        for video in sorted(label):
            feats = window_features(frames.get(video, []))
            w.writerow([video, participant_of(video, participant_regex), label[video],
                        *[feats[k] for k in ALL_WINDOW_FEATURES], int(feats["frames"])])
    (out.with_suffix(".meta.json")).write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}" + ("  (PARTIAL: biased subset, do not report as results)" if partial else ""))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=URL, help="Zenodo URL (default) or a local CSV path")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--participant-regex", default=None,
                    help="regex with one group that extracts the participant id from Video_Name")
    ap.add_argument("--max-bytes", type=float, default=None, help="stop early (testing only)")
    ap.add_argument("--check", action="store_true", help="only verify the column mapping on the source")
    ap.add_argument("--positional", choices=["auto", "force", "never"], default="auto")
    a = ap.parse_args()
    run(a.source, a.out, a.participant_regex, int(a.max_bytes) if a.max_bytes else None, a.check, a.positional)
    return 0


if __name__ == "__main__":
    sys.exit(main())
