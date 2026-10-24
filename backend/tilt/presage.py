"""Presage Physiology: heart rate and breathing rate from short webcam clips.

The browser records the call in rolling ~15 second clips and posts each one here. We upload it to
the Presage Physiology API (the same calls their Python SDK makes with preprocess=False), poll
for the result in a background thread, and keep the latest reading on the session. The lie model
then uses Presage's heart rate instead of our own forehead estimate, plus breathing rate.

Readings lag the call by the clip length plus processing time, so they describe the last half
minute or so rather than the exact answer. Our in-browser heart rate stays as the fallback.
"""
from __future__ import annotations

import logging
import os
import shutil
import statistics
import subprocess
import tempfile
import threading
import time

import httpx

log = logging.getLogger("tilt.presage")

API = os.environ.get("PRESAGE_API_URL", "https://api.physiology.presagetech.com")
PART = 5 * 1024 * 1024
FRESH_SECONDS = 90
POLL_SECONDS = 180
_sleep = time.sleep


def api_key() -> str | None:
    return os.environ.get("PRESAGE_API_KEY") or None


def _to_mp4(data: bytes, ext: str) -> tuple[bytes, str]:
    """Presage takes MP4, MOV or AVI. Chrome may record WebM; convert it when ffmpeg is around."""
    if ext in ("mp4", "mov", "avi") or not shutil.which("ffmpeg"):
        return data, ext
    with tempfile.TemporaryDirectory() as d:
        src, dst = os.path.join(d, f"in.{ext}"), os.path.join(d, "out.mp4")
        with open(src, "wb") as fh:
            fh.write(data)
        done = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-an", "-c:v", "libx264",
                               "-preset", "ultrafast", "-pix_fmt", "yuv420p", dst], capture_output=True)
        if done.returncode != 0:
            return data, ext
        with open(dst, "rb") as fh:
            return fh.read(), "mp4"


def _numbers(series) -> list[float]:
    """Presage returns {time: value} per vital; values may be bare numbers or {"value": x, ...}."""
    out = []
    items = series.values() if isinstance(series, dict) else series if isinstance(series, list) else []
    for v in items:
        if isinstance(v, dict):
            v = v.get("value", v.get("val"))
        if isinstance(v, (int, float)) and v == v and v > 0:
            out.append(float(v))
    return out


def parse(result: dict) -> dict:
    """Pull the latest heart rate and breathing rate out of a retrieve-data payload."""
    found: dict[str, list[float]] = {}

    def walk(node):
        if not isinstance(node, dict):
            return
        for key, val in node.items():
            k = key.lower()
            if k in ("hr", "heart_rate", "pulse", "pulse_rate") and "hr" not in found:
                found["hr"] = _numbers(val)
            elif k in ("rr", "br", "breathing", "breathing_rate", "respiration_rate") and "rr" not in found:
                found["rr"] = _numbers(val)
            elif isinstance(val, dict):
                walk(val)

    walk(result)
    out = {}
    for k, vals in found.items():
        if vals:
            out[k] = round(statistics.median(vals[-5:]), 1)
    return out


class PresageVitals:
    """Per-session Presage state: the latest reading and whether a clip is in flight."""

    def __init__(self, key: str | None = None, client: httpx.Client | None = None, clock=time.time):
        self.key = key if key is not None else api_key()
        self.client = client
        self.clock = clock
        self.hr: float | None = None
        self.rr: float | None = None
        self.updated: float | None = None
        self.status = "off" if not self.key else "waiting"
        self.error: str | None = None
        self.clips = 0
        self.readings = 0
        self._busy = False
        self._lock = threading.Lock()

    def view(self) -> dict:
        return {"enabled": bool(self.key), "hr": self.hr, "rr": self.rr, "status": self.status,
                "error": self.error, "clips": self.clips, "readings": self.readings,
                "age": round(self.clock() - self.updated, 1) if self.updated else None}

    def fresh(self) -> dict:
        """The latest reading if it is recent enough to describe what is happening now."""
        if self.updated is None or self.clock() - self.updated > FRESH_SECONDS:
            return {}
        return {k: v for k, v in (("hr", self.hr), ("rr", self.rr)) if v is not None}

    def submit(self, data: bytes, ext: str, background: bool = True) -> bool:
        """Queue a clip. Only one clip is processed at a time; extra clips are dropped."""
        if not self.key:
            return False
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        self.clips += 1
        self.status = "processing"
        if background:
            threading.Thread(target=self._process, args=(data, ext), daemon=True).start()
        else:
            self._process(data, ext)
        return True

    def _process(self, data: bytes, ext: str) -> None:
        client = self.client or httpx.Client(timeout=30)
        try:
            data, ext = _to_mp4(data, ext)
            reading = parse(self._run(client, data))
            if reading:
                self.hr = reading.get("hr", self.hr)
                self.rr = reading.get("rr", self.rr)
                self.updated = self.clock()
                self.readings += 1
                self.status = "ok"
                self.error = None
            else:
                self.status = "no reading"
                self.error = "Presage returned no vitals for that clip. Keep your face still and well lit."
        except Exception as exc:  # network, auth or format problems shouldn't break the game
            log.warning("presage clip failed: %s", exc)
            self.status = "error"
            self.error = str(exc)[:200]
        finally:
            if self.client is None:
                client.close()
            self._busy = False

    def _run(self, client: httpx.Client, data: bytes) -> dict:
        headers = {"x-api-key": self.key}
        res = client.post(f"{API}/v1/upload-url", headers=headers,
                          json={"file_size": len(data), "hr_br": {"to_process": True}})
        if res.status_code in (401, 403):
            raise RuntimeError("Presage rejected the API key")
        res.raise_for_status()
        body = res.json()
        parts = []
        for i, url in enumerate(body["urls"]):
            put = client.put(url, content=data[i * PART:(i + 1) * PART])
            put.raise_for_status()
            parts.append({"ETag": put.headers.get("ETag"), "PartNumber": i + 1})
        client.post(f"{API}/v1/complete", headers=headers,
                    json={"id": body["id"], "upload_id": body["upload_id"], "parts": parts}).raise_for_status()
        stop = time.time() + POLL_SECONDS
        while time.time() < stop:
            got = client.post(f"{API}/retrieve-data", headers=headers, json={"id": body["id"], "reshape": False})
            if got.status_code == 200:
                return got.json()
            if got.status_code in (401, 403):
                raise RuntimeError("Presage rejected the API key")
            _sleep(2)
        raise TimeoutError("Presage took too long on that clip")
