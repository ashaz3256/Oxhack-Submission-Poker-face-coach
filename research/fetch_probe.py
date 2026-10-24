"""Step 1: fetch the small labelled files and probe the 18 GB landmark CSV without downloading it.

Dataset: Rahayu et al., "Multimodal Deception Detection Dataset with Personality Profiling
across Indonesian Ethnic Groups", Zenodo, doi:10.5281/zenodo.17421590, CC BY 4.0.

Standard library only. Run from the repo root:

    python research/fetch_probe.py

Writes everything to research/data/ (git-ignored). Takes well under a minute.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://zenodo.org/records/17421590/files/"
SMALL = ["feature_descriptions.json", "AudioDataset_Features.csv", "TextDataset_English.csv"]
LANDMARKS = "LandmarkDataset.csv"
HEAD_BYTES = 3_000_000
OUT = Path(__file__).resolve().parent / "data"
UA = {"User-Agent": "tilt-oxhack-research/1.0"}


def get(url: str, headers: dict | None = None) -> tuple[int, dict, bytes]:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, {k.title(): v for k, v in r.headers.items()}, r.read()  # Zenodo sends lowercase names


def probe_range(start: int, length: int) -> dict:
    status, headers, body = get(BASE + LANDMARKS + "?download=1", {"Range": f"bytes={start}-{start + length - 1}"})
    return {"status": status, "content_range": headers.get("Content-Range"),
            "content_length": headers.get("Content-Length"), "bytes": len(body), "body": body}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"source": "doi:10.5281/zenodo.17421590", "licence": "CC BY 4.0"}

    for name in SMALL:
        status, _, body = get(BASE + name + "?download=1")
        (OUT / name).write_bytes(body)
        report[name] = {"status": status, "bytes": len(body)}
        print(f"saved {name} ({len(body):,} bytes)")

    head = probe_range(0, HEAD_BYTES)
    report["landmark_range_head"] = {k: v for k, v in head.items() if k != "body"}
    if head["status"] != 206:
        print("Server ignored the Range header (status %s); partial access is not possible." % head["status"])
    text = head["body"].decode("utf-8", errors="replace")
    lines = text.split("\n")[:-1]  # drop the partial last line
    (OUT / "landmark_head.csv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="")
    header = lines[0].split(",") if lines else []
    report["landmark_columns"] = len(header)
    report["landmark_rows_in_head"] = max(len(lines) - 1, 0)
    (OUT / "landmark_header.json").write_text(json.dumps(header), encoding="utf-8")
    print(f"landmark head: status {head['status']}, {len(header)} columns, {len(lines) - 1} rows")

    total = None
    if head["content_range"] and "/" in head["content_range"]:
        total = int(head["content_range"].rsplit("/", 1)[1])
    report["landmark_total_bytes"] = total
    if head["status"] == 206 and total:
        mid = probe_range(total // 2, 200_000)
        report["landmark_range_middle"] = {k: v for k, v in mid.items() if k != "body"}
        mid_lines = mid["body"].decode("utf-8", errors="replace").split("\n")[1:-1]
        (OUT / "landmark_middle.csv").write_text("\n".join(lines[:1] + mid_lines) + "\n", encoding="utf-8",
                                                 newline="")
        print(f"middle probe: status {mid['status']}, {len(mid_lines)} complete rows")

    (OUT / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nDone. Attach the files in {OUT} to the chat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
