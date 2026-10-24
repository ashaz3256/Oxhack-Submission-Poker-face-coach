"""Forward vitals from any source into a Tilt session.

Pipe one JSON object per line on stdin, e.g. from a small wrapper around the Presage
SmartSpectra SDK that prints {"hr": 74.2, "br": 15.1} each time it gets a reading:

    presage-wrapper | python presage_bridge.py --session <session id>

Each line is timestamped on arrival and posted to /api/sessions/<id>/vitals.
"""
import argparse
import json
import sys
import time
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()
    url = f"{args.api}/api/sessions/{args.session}/vitals"
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        reading = json.loads(line)
        sample = {"t": time.time(), "hr": float(reading["hr"]), "br": reading.get("br")}
        req = urllib.request.Request(url, data=json.dumps({"samples": [sample]}).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
        print(f"sent hr={sample['hr']} br={sample['br']}", file=sys.stderr)


if __name__ == "__main__":
    main()
