"""Understanding what you said: names, card numbers and yes/no, robust to speech-to-text slips."""
from __future__ import annotations

import re

from .cues import words

ALIASES = {
    "Ava": ["ava", "eva", "ava's", "ava's", "avah", "ever", "iva"],
    "Ben": ["ben", "benny", "bens", "ben's", "benjamin"],
    "Cleo": ["cleo", "clio", "chloe", "cleo's", "kleo", "clea"],
    "Dev": ["dev", "dave", "deb", "dev's", "devin", "def"],
    "Eli": ["eli", "ellie", "ely", "elie", "eli's", "elle", "ally"],
}
NUMBERS = {
    "one": 1, "first": 1, "won": 1, "1": 1,
    "two": 2, "second": 2, "to": None, "too": 2, "2": 2,
    "three": 3, "third": 3, "free": 3, "3": 3,
    "four": 4, "fourth": 4, "for": None, "4": 4,
    "five": 5, "fifth": 5, "5": 5,
    "six": 6, "sixth": 6, "sex": 6, "6": 6,
}
AGAIN = re.compile(r"\b(again|yes|yeah|another|play|deal|sure|go|start|ready|next)\b")


def find_names(text: str, candidates: list[str]) -> list[str]:
    """Names in the order they were said, limited to `candidates`."""
    found = []
    for w in words(text):
        for name in candidates:
            if w in ALIASES.get(name, [name.lower()]) and name not in found:
                found.append(name)
    return found


def last_name(text: str, candidates: list[str]) -> str | None:
    """For votes: 'not Ben, I vote Cleo' should land on Cleo, so take the last name said."""
    hits = []
    for w in words(text):
        for name in candidates:
            if w in ALIASES.get(name, [name.lower()]):
                hits.append(name)
    return hits[-1] if hits else None


def parse_number(text: str, lo: int = 1, hi: int = 6) -> int | None:
    for w in words(text):
        n = NUMBERS.get(w)
        if n is not None and lo <= n <= hi:
            return n
    return None


def wants_again(text: str) -> bool:
    return bool(AGAIN.search(text.lower()))
