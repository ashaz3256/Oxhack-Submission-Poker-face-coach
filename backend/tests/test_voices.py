"""ElevenLabs delivery logic with the network call replaced."""
import urllib.error

import pytest

from tilt import voices

BASE = {"stability": 0.35, "similarity_boost": 0.8, "style": 0.45, "use_speaker_boost": True}


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    voices._cache.clear()
    voices._broken_models.clear()
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test")
    monkeypatch.delenv("ELEVENLABS_EXPRESSIVE_MODEL", raising=False)


def say(text, speaker="Cleo"):
    return voices.synthesize(text, speaker, "voice-id", BASE, "eleven_multilingual_v2")


def test_tags_are_kept_filtered_or_added():
    assert voices.direct("Ava", "[sarcastic] Convenient.") == "[sarcastic] Convenient."
    assert voices.direct("Ava", "[dances wildly] Hi") == "Hi"
    assert voices.direct("Narrator", "Night falls. Everyone close your eyes.").startswith("[whispers]")
    assert voices.strip_tags("[nervous] I was home.") == "I was home."


def test_falls_back_to_the_existing_model_without_tags_and_remembers(monkeypatch):
    calls = []

    def fake(voice, body, key):
        calls.append(body)
        if body["model_id"] == "eleven_v4_turbo":
            raise urllib.error.HTTPError("u", 422, "bad", {}, None)
        return b"mp3"

    monkeypatch.setattr(voices, "_post", fake)
    assert say("[nervous] I was home.") == b"mp3"
    assert calls[-1]["model_id"] == "eleven_multilingual_v2" and calls[-1]["text"] == "I was home."
    calls.clear()
    say("Another line")
    assert [c["model_id"] for c in calls] == ["eleven_multilingual_v2"]


def test_settings_layer_on_base_and_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(voices, "_post", lambda v, b, k: calls.append(b) or b"mp3")
    say("[laughs] Relax.", "Ben")
    say("[laughs] Relax.", "Ben")
    assert len(calls) == 1
    assert calls[0]["text"] == "[laughs] Relax." and calls[0]["model_id"] == "eleven_v4_turbo"
    assert calls[0]["voice_settings"] == {**BASE, "style": 0.5}


def test_off_switch_and_no_key(monkeypatch):
    calls = []
    monkeypatch.setattr(voices, "_post", lambda v, b, k: calls.append(b) or b"mp3")
    monkeypatch.setenv("ELEVENLABS_EXPRESSIVE_MODEL", "off")
    say("Hello")
    assert calls[0]["model_id"] == "eleven_multilingual_v2"
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    assert say("Hello again") is None
