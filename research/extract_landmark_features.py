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
    coords = [i for i, h in enumerate(header) if h.strip()[-2:] in ("_X", "_Y", "_Z")]
    face_cols = coords[:3 * N_FACE] if len(coords) >= 3 * N_FACE else None
    mesh = load_canonical_mesh() if face_cols else None
    if face_cols is None:
        print("no full face mesh in this file; frames are not quality-checked")

    frames: dict[str, list] = defaultdict(list)
    label: dict[str, int] = {}
    bad_geometry = bad_mesh = rows = 0
    orientation: dict[str, int] = defaultdict(int)
    frames_seen: dict[str, int] = defaultdict(int)
    read_bytes = len(",".join(header))
    t0 = last = time.time()
    partial = False
    for row in reader:
        if not row:
            continue
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
        frames_seen[video] += 1
        if face_cols:
            try:
                face = np.array([float(row[c]) for c in face_cols]).reshape(N_FACE, 3)
            except ValueError:
                continue
            if not mesh_error(face, mesh) <= MAX_MESH_ERROR:
                bad_mesh += 1
                continue
        pts, turn = upright(pts)
        orientation[turn] += 1
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

    usable = rows - bad_mesh - bad_geometry
    summary = {"source": source, "column_mapping": how, "frames_read": rows, "frames_failing_mesh": bad_mesh,
               "max_mesh_error": MAX_MESH_ERROR, "frames_failing_geometry": bad_geometry,
               "frame_orientation": dict(orientation), "recordings": len(label), "partial_read": partial,
               "bytes_read": read_bytes}
    print(json.dumps(summary, indent=2))
    if check:
        # Frames that pass the mesh check must also be the right way up and round; lost-face frames
        # were already set aside, so this tests the column mapping on its own.
        ok = rows - bad_mesh > 0 and bad_geometry / max(rows - bad_mesh, 1) < 0.05
        print("Column mapping looks right." if ok else "Column mapping looks WRONG: most frames fail the face-geometry check.")
        sample = sorted(label)[:10]
        print("Example video names:", sample)
        return summary | {"mapping_ok": ok}
    if usable <= 0:
        raise SystemExit("no usable frames")

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "participant", "label", *ALL_WINDOW_FEATURES, "frames", "frames_seen"])
        for video in sorted(label):
            feats = window_features(frames.get(video, []))
            w.writerow([video, participant_of(video, participant_regex), label[video],
                        *[feats[k] for k in ALL_WINDOW_FEATURES], int(feats["frames"]), frames_seen[video]])
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
