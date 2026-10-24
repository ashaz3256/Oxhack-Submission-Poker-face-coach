"""ElevenLabs voice logic with the network call replaced."""
import urllib.error

import pytest

from tilt import voices


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    voices._cache.clear()
    voices._broken_models.clear()
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.delenv("ELEVENLABS_MODEL", raising=False)


def test_tags_are_kept_filtered_or_added():
    assert voices.direct("Ava", "[sarcastic] Convenient.") == "[sarcastic] Convenient."
    assert voices.direct("Ava", "[dances wildly] Hi") == "Hi"
    assert voices.direct("Narrator", "Night falls. Everyone close your eyes.").startswith("[whispers]")
    assert voices.strip_tags("[nervous] I was home.") == "I was home."


def test_falls_back_to_flash_without_tags_and_remembers(monkeypatch):
    calls = []

    def fake(voice, body, key):
        calls.append(body)
        if body["model_id"] == "eleven_v4_turbo":
            raise urllib.error.HTTPError("u", 422, "bad", {}, None)
        return b"mp3"

    monkeypatch.setattr(voices, "_post", fake)
    assert voices.speak("[nervous] I was home.", "Cleo") == b"mp3"
    assert calls[-1] == {"text": "I was home.", "model_id": "eleven_flash_v2_5"}
    calls.clear()
    voices.speak("Another line", "Cleo")
    assert [c["model_id"] for c in calls] == ["eleven_flash_v2_5"]  # does not retry the failing model


def test_uses_profile_and_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(voices, "_post", lambda v, b, k: calls.append(b) or b"mp3")
    voices.speak("[laughs] Relax.", "Ben")
    voices.speak("[laughs] Relax.", "Ben")
    assert len(calls) == 1
    assert calls[0]["text"] == "[laughs] Relax." and calls[0]["voice_settings"] == voices.PROFILES["Ben"]


def test_no_key_means_browser_voice(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    assert voices.speak("Hello", "Ava") is None
