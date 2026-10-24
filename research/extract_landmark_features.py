"""Step 2: stream LandmarkDataset.csv once and write one row of face features per recording.

The file is 18 GB, but we only need 17 of its 511 landmarks, so it is read line by line straight
from Zenodo (or from a local copy) and never stored. Memory stays small; the run is limited by
download speed. Standard library + numpy only.

Progress is saved after every recording and dropped connections are retried, so if the run stops
(sleep, Ctrl+C, no network) just run the same command again and it carries on from where it was.
`--restart` throws the saved progress away.

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
import http.client
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

# MediaPipe's canonical face (Apache 2.0), used to reject frames where FaceMesh lost the face.
MESH_URL = ("https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/"
            "mediapipe/modules/face_geometry/data/canonical_face_model.obj")
MESH_PATH = ROOT / "research" / "data" / "canonical_face_model.obj"
FRAME_ASPECT = 9 / 16   # height / width of the I3D videos (1920x1080); best fit of every one sampled
MAX_MESH_ERROR = 0.10   # sampled clean frames sit at 0.05-0.08, lost-face frames at 0.12-0.18


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
    # Coordinate columns only: the I3D file also carries metadata (ethnic_group, ethnic_key). Its face
    # columns have descriptive names (Contour_Chin, Left_Eye_Upper_0, ...) but are in MediaPipe index
    # order: matching them blind against the canonical face mesh recovered that order (research/README.md).
    coords = [i for i, h in enumerate(hdr) if h[-2:] in ("_X", "_Y", "_Z")]
    expected = 3 * (N_FACE + N_IRIS + N_POSE)
    if len(coords) != expected:
        raise SystemExit(f"{len(coords)} coordinate columns, expected {expected} for positional mapping; "
                         f"header starts {hdr[:12]}")
    # Assumed order: 468 face, then 10 iris, then 33 pose landmarks, each as x, y, z. Verified by --check.
    mapping = {lm: (coords[3 * lm], coords[3 * lm + 1]) for lm in USED_LANDMARKS}
    return ids, mapping, "by position (face, iris, pose; x,y,z each)"


def load_canonical_mesh() -> np.ndarray:
    """468 x 3 canonical face, centred, in image orientation (y down, z away from the camera)."""
    if not MESH_PATH.exists():
        MESH_PATH.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(MESH_URL, headers={"User-Agent": "tilt-oxhack-research/1.0"})
        MESH_PATH.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    v = [[float(t) for t in line.split()[1:4]] for line in MESH_PATH.read_text().splitlines() if line.startswith("v ")]
    mesh = np.array(v) * [1, -1, -1]
    if mesh.shape != (N_FACE, 3):
        raise SystemExit(f"unexpected canonical mesh shape {mesh.shape}")
    return mesh - mesh.mean(0)


def mesh_error(face: np.ndarray, mesh: np.ndarray) -> float:
    """How far one frame's 468 points are from a rotated, scaled canonical face (0 = perfect).

    MediaPipe FaceMesh always returns 468 points, even when it has lost the face; on the sideways
    I3D videos it often does, and the points then jump to a distorted shape. A similarity fit to the
    canonical face separates the two cleanly. Expressions and head turns stay well under the cut.
    """
    p = face * [1, FRAME_ASPECT, 1]
    p = p - p.mean(0)
    u, s, vt = np.linalg.svd(mesh.T @ p)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    scale = (s * [1, 1, d]).sum() / (mesh ** 2).sum()
    spread = (p ** 2).sum()
    if spread <= 0:
        return float("inf")
    return float(np.sqrt(((p - scale * mesh @ r) ** 2).sum() / spread))


def upright(points: dict[int, tuple[float, float]]) -> tuple[dict[int, tuple[float, float]], str]:
    """Rotate a frame so the face is upright, as a webcam frame would be.

    Many I3D recordings are phone videos whose rotation flag was ignored when the landmarks were
    extracted, so the face lies on its side (forehead-to-chin along image x). Features compare x with
    x and y with y, so they are only meaningful once the face is the right way up. The rotation is
    decided per frame from the forehead-to-chin direction; quarter turns keep coordinates in [0, 1].
    """
    dx = points[152][0] - points[10][0]
    dy = points[152][1] - points[10][1]
    if abs(dy) >= abs(dx):
        if dy > 0:
            return points, "upright"
        return {k: (1 - x, 1 - y) for k, (x, y) in points.items()}, "upside_down"
    if dx > 0:
        return {k: (1 - y, x) for k, (x, y) in points.items()}, "chin_right"
    return {k: (y, 1 - x) for k, (x, y) in points.items()}, "chin_left"


def geometry_ok(points: dict[int, tuple[float, float]]) -> bool:
    """Sanity check that the mapping is right: a face that is the right way up and the right way round."""
    x = {k: v[0] for k, v in points.items()}
    y = {k: v[1] for k, v in points.items()}
    return (y[10] < y[1] < y[152]                      # forehead above nose above chin
            and y[105] < y[159] and y[334] < y[386]    # brows above eyes
            and y[13] <= y[14] + 0.02 * (y[152] - y[10])  # upper lip not below lower (they touch when closed)
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

OUT_COLUMNS = ["video", "participant", "label", *ALL_WINDOW_FEATURES, "frames", "frames_seen"]
COUNTERS = ("frames_read", "frames_failing_mesh", "frames_failing_geometry")
MAX_RETRIES = 30  # consecutive reconnects without progress before giving up


def open_at(source: str, offset: int):
    """Binary stream of `source` from byte `offset`, and the total size in bytes."""
    if source.startswith("http"):
        headers = {"User-Agent": "tilt-oxhack-research/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        resp = urllib.request.urlopen(urllib.request.Request(source, headers=headers), timeout=120)
        if offset and resp.status != 206:
            raise SystemExit(f"server ignored the Range request (status {resp.status}); cannot resume")
        rng = resp.headers.get("Content-Range")  # "bytes start-end/total"
        total = int(rng.rsplit("/", 1)[1]) if rng and "/" in rng else offset + int(resp.headers.get("Content-Length") or 0)
        return resp, total
    p = Path(source)
    f = open(p, "rb")
    f.seek(offset)
    return f, p.stat().st_size


def parse_line(raw: bytes) -> list[str]:
    return next(csv.reader([raw.decode("utf-8").rstrip("\r\n")]), [])


class Recording:
    """Everything kept about the recording being read; rows of one recording are contiguous."""

    def __init__(self, name: str, label: int):
        self.name, self.label = name, label
        self.frames: list[dict] = []
        self.seen = 0
        self.counts = dict.fromkeys(COUNTERS, 0)
        self.orientation: dict[str, int] = defaultdict(int)


class Extractor:
    def __init__(self, header: list[str], positional: str, participant_regex: str | None):
        self.header = header
        self.ids, self.cols, self.how = resolve_columns(header, positional)
        coords = [i for i, h in enumerate(header) if h.strip()[-2:] in ("_X", "_Y", "_Z")]
        self.face_cols = coords[:3 * N_FACE] if len(coords) >= 3 * N_FACE else None
        self.mesh = load_canonical_mesh() if self.face_cols else None
        self.participant_regex = participant_regex

    def add(self, rec: Recording, row: list[str]) -> None:
        rec.counts["frames_read"] += 1
        if int(float(row[self.ids["Class"]])) != rec.label:
            raise SystemExit(f"recording {rec.name} has more than one label")
        try:
            pts = {lm: (float(row[cx]), float(row[cy])) for lm, (cx, cy) in self.cols.items()}
        except ValueError:
            return  # face not detected in this frame (empty cells)
        if any(v != v for xy in pts.values() for v in xy):
            return
        rec.seen += 1
        if self.face_cols:
            try:
                face = np.array([float(row[c]) for c in self.face_cols]).reshape(N_FACE, 3)
            except ValueError:
                return
            if not mesh_error(face, self.mesh) <= MAX_MESH_ERROR:
                rec.counts["frames_failing_mesh"] += 1
                return
        pts, turn = upright(pts)
        rec.orientation[turn] += 1
        if not geometry_ok(pts):
            rec.counts["frames_failing_geometry"] += 1
            return
        rec.frames.append(frame_primitives(pts))

    def output_row(self, rec: Recording) -> list:
        feats = window_features(rec.frames)
        return [rec.name, participant_of(rec.name, self.participant_regex), rec.label,
                *[feats[k] for k in ALL_WINDOW_FEATURES], int(feats["frames"]), rec.seen]


def run(source: str, out: Path, participant_regex: str | None, max_bytes: int | None, check: bool,
        positional: str, restart: bool = False) -> dict:
    """Stream the file once. Unless checking or stopping early, progress is saved after every
    recording (`<out>.partial.csv` plus `<out>.progress.json`), so a dropped connection, a sleeping
    laptop or a Ctrl+C costs at most one recording: run the same command again to carry on.
    """
    persist = not check and not max_bytes
    partial_csv, progress_json = out.with_suffix(".partial.csv"), out.with_suffix(".progress.json")
    if persist:
        out.parent.mkdir(parents=True, exist_ok=True)
    if restart:
        partial_csv.unlink(missing_ok=True)
        progress_json.unlink(missing_ok=True)

    state = json.loads(progress_json.read_text()) if persist and progress_json.exists() else None
    if state and state["source"] != source:
        raise SystemExit(f"{progress_json} belongs to {state['source']}; pass --restart to start over")
    if state is None:
        stream, _ = open_at(source, 0)
        first = stream.readline()
        stream.close()
        state = {"source": source, "header": parse_line(first), "offset": len(first), "recordings": 0,
                 "counts": dict.fromkeys(COUNTERS, 0), "orientation": {}, "complete": False}
    else:
        print(f"resuming at byte {state['offset']:,} after {state['recordings']} recordings")
    header = state["header"]
    ex = Extractor(header, positional, participant_regex)
    print(f"{len(header)} columns; landmark columns resolved {ex.how}")
    if ex.face_cols is None:
        print("no full face mesh in this file; frames are not quality-checked")

    done: set[str] = set()
    rows_out: list[list] = []  # only used when not persisting
    if persist:
        if partial_csv.exists():
            with open(partial_csv, newline="") as f:
                done = {r[0] for r in csv.reader(f) if r and r[0] != "video"}
        else:
            with open(partial_csv, "w", newline="") as f:
                csv.writer(f).writerow(OUT_COLUMNS)

    def finish(rec: Recording, next_offset: int) -> None:
        row = ex.output_row(rec)
        for k in COUNTERS:
            state["counts"][k] += rec.counts[k]
        for k, v in rec.orientation.items():
            state["orientation"][k] = state["orientation"].get(k, 0) + v
        state["recordings"] += 1
        state["offset"] = next_offset
        done.add(rec.name)
        if persist:
            with open(partial_csv, "a", newline="") as f:
                csv.writer(f).writerow(row)
            tmp = progress_json.with_suffix(".tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(progress_json)
        else:
            rows_out.append(row)

    partial = False
    failures = 0
    t0, start_offset, last = time.time(), state["offset"], 0.0
    while not state["complete"] and not partial:
        rec, pos = None, state["offset"]
        try:
            stream, total = open_at(source, pos)
            while True:
                raw = stream.readline()
                if not raw:
                    if total and pos < total:
                        raise ConnectionError(f"stream ended early at byte {pos:,} of {total:,}")
                    break
                line_start, pos = pos, pos + len(raw)
                row = parse_line(raw)
                if not row:
                    continue
                if len(row) != len(header):
                    if not raw.endswith(b"\n") and total and pos < total:
                        raise ConnectionError("truncated line")
                    break  # truncated final line of a partial file
                video = row[ex.ids["Video_Name"]]
                if rec is None or video != rec.name:
                    if rec is not None:
                        finish(rec, line_start)
                        failures = 0
                    if video in done:
                        raise SystemExit(f"recording {video} appears twice; the file is not grouped by recording")
                    rec = Recording(video, int(float(row[ex.ids["Class"]])))
                ex.add(rec, row)
                if time.time() - last > 15:
                    last = time.time()
                    rate = (pos - start_offset) / max(last - t0, 1e-9) / 1e6
                    pct = f" {100 * pos / total:.1f}%" if total else ""
                    print(f"  {pos / 1e9:.2f} GB{pct}, {state['recordings']} recordings done, {rate:.1f} MB/s",
                          flush=True)
                if max_bytes and pos >= max_bytes:
                    partial = True
                    break
            stream.close()
            if rec is not None:
                finish(rec, pos)
            state["complete"] = not partial
        except (OSError, ConnectionError, http.client.HTTPException) as exc:  # URLError and timeouts are OSErrors
            failures += 1
            if failures > MAX_RETRIES:
                raise SystemExit(f"giving up after {MAX_RETRIES} failed reconnects; run again to resume") from exc
            wait = min(300, 5 * 2 ** min(failures, 6))
            print(f"  connection lost ({exc!r}); resuming from byte {state['offset']:,} in {wait}s", flush=True)
            time.sleep(wait)
    if persist:
        progress_json.write_text(json.dumps(state))

    counts = state["counts"]
    summary = {"source": source, "column_mapping": ex.how, **counts, "max_mesh_error": MAX_MESH_ERROR,
               "frame_orientation": state["orientation"], "recordings": state["recordings"],
               "partial_read": partial, "bytes_read": state["offset"]}
    print(json.dumps(summary, indent=2))
    if check:
        # Frames that pass the mesh check must also be the right way up and round; lost-face frames
        # were already set aside, so this tests the column mapping on its own.
        passed_mesh = counts["frames_read"] - counts["frames_failing_mesh"]
        ok = passed_mesh > 0 and counts["frames_failing_geometry"] / max(passed_mesh, 1) < 0.05
        print("Column mapping looks right." if ok else "Column mapping looks WRONG: most frames fail the face-geometry check.")
        print("Example video names:", sorted(done)[:10])
        return summary | {"mapping_ok": ok}

    if persist:
        with open(partial_csv, newline="") as f:
            rows_out = [r for r in csv.reader(f) if r and r[0] != "video"]
    if not rows_out:
        raise SystemExit("no recordings read")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(OUT_COLUMNS)
        w.writerows(sorted(rows_out, key=lambda r: r[0]))
    (out.with_suffix(".meta.json")).write_text(json.dumps(summary, indent=2))
    if persist:
        partial_csv.unlink()
        progress_json.unlink()
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
    ap.add_argument("--restart", action="store_true", help="discard saved progress and start from the beginning")
    a = ap.parse_args()
    run(a.source, a.out, a.participant_regex, int(a.max_bytes) if a.max_bytes else None, a.check, a.positional,
        a.restart)
    return 0


if __name__ == "__main__":
    sys.exit(main())
