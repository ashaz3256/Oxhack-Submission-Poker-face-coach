"""The post-game breakdown: every answer, vote and kill in one game, marked as what you did right
or where it went wrong, plus your biggest tells and what to do about them."""
from __future__ import annotations

from .cues import LABELS
from .liemodel import CENTRE, SCALE
from .mafia import YOU, Game, MafiaSession

GOOD, BAD, MEH = "good", "bad", "neutral"

TIPS = {
    "pitch_z": "Your voice goes up when you lie. Breathe out first and speak from lower in your chest.",
    "pitch_var_z": "Your pitch wobbles. Slow down and keep your sentences flat and even.",
    "loudness_z": "Your volume changes when you lie. Keep it at your normal talking level.",
    "rate_z": "Your talking speed changes. Keep the same pace you use for the truth.",
    "pause_z": "You pause mid-answer. Decide your story before you open your mouth.",
    "latency_z": "You're slow to start. Have a short, confident first line ready.",
    "blink_z": "You blink more when you lie. Relax your eyes and look at the screen.",
    "brow_raise_z": "Your eyebrows go up. Keep your face as still as when you tell the truth.",
    "brow_down_z": "You frown. Soften your face between sentences.",
    "lip_press_z": "You press your lips together. Keep your mouth loose.",
    "gaze_away_z": "You look away. Hold eye contact with the camera.",
    "head_motion_z": "Your head goes still or stiff. Move like you normally would.",
    "smile_z": "Your smile changes. Don't force one, and don't drop it either.",
    "hr_z": "Your heart rate climbs. Take slow breaths while the others are talking.",
    "rr_z": "Your breathing speeds up. Breathe slowly through your nose between questions.",
    "hedges": "You hedge with 'honestly', 'trust me', 'I think'. Truth doesn't need selling, so say it plainly.",
    "fillers": "Lots of ums and ahs. Shorter answers give you fewer chances to stumble.",
    "negations": "Too many denials. Say what you did, not what you didn't do.",
    "log_words": "Your lies are short. Add one or two concrete, boring details.",
    "llm": "Your wording sounded evasive. Give specific details someone could check.",
}


def _matches_tip(k: str, mean: float) -> bool:
    """Only report a tell in the direction its tip describes (more pauses, not fewer)."""
    if k in ("smile_z", "head_motion_z", "loudness_z", "rate_z"):
        return abs(mean) >= 0.3
    if k == "log_words":
        return mean < CENTRE["log_words"]
    if k.endswith("_z"):
        return mean >= 0.3
    return mean > 0


def _pct(p: float | None) -> str:
    return "–" if p is None else f"{round(p * 100)}%"


def _find_game(s: MafiaSession, number: int | None) -> Game | None:
    if number is not None:
        return next((g for g in s.games if g.number == number), None)
    finished = [g for g in s.games if g.phase == "over"]
    return finished[-1] if finished else s.current


def _answer_moment(u, entry: dict, mafia: bool) -> dict:
    p = u.p_lie or 0.0
    cues = ", ".join(u.cues[:3])
    rank = entry.get("rank")
    after = " Straight after, you were the town's top suspect." if rank == 1 else ""
    if u.is_lie is None:
        verdict, title = MEH, "Very short answer"
        detail = f"Too short to read properly ({_pct(p)}). Short answers can also come across as dodging."
    elif mafia and p < 0.45:
        verdict, title = GOOD, "Smooth lie"
        detail = f"The detector only gave it {_pct(p)}. Nothing gave you away."
    elif mafia and p >= 0.6:
        verdict, title = BAD, "Your lie showed"
        detail = f"The detector read {_pct(p)}" + (f", from {cues}." if cues else ".") + after
    elif mafia:
        verdict, title = MEH, "Borderline lie"
        detail = f"{_pct(p)} chance of a lie. It got through, just." + (f" Watch: {cues}." if cues else "")
    elif p < 0.45:
        verdict, title = GOOD, "Calm and believable"
        detail = f"You told the truth and it looked like it ({_pct(p)})."
    elif p >= 0.6:
        verdict, title = BAD, "Truth that looked like a lie"
        detail = (f"You were honest, but the detector read {_pct(p)}" + (f" ({cues})" if cues else "")
                  + ". Nerves made the town suspect you." + after)
    else:
        verdict, title = MEH, "Slightly nervous truth"
        detail = f"{_pct(p)}. Honest, but a little tense."
    if u.llm is not None and u.llm >= 0.65 and verdict != GOOD:
        detail += " Your wording sounded evasive too."
    return {"type": "answer", "day": u.day, "verdict": verdict, "title": title, "detail": detail,
            "question": u.question, "said": u.transcript, "p_lie": u.p_lie, "cues": u.cues,
            "lying": bool(u.is_lie), "rank": rank}


def _vote_moment(e: dict, game: Game) -> list[dict]:
    out, mafia = e["out"], game.you_mafia
    voted = e["you_voted"]
    against = e["votes_on_you"]
    moments = []
    if not mafia:
        if e["you_voted_mafia"]:
            moments.append({"verdict": GOOD, "title": f"You voted for {voted}, the mafia",
                            "detail": "Good read."})
        else:
            moments.append({"verdict": BAD, "title": f"You voted for {voted}, who was innocent",
                            "detail": f"The mafia was {game.mafia}."})
    if out == YOU:
        detail = f"{against} of the others voted for you."
        if e.get("out_blind") and e["out_blind"] != YOU:
            detail += (f" Without your tells the town would have voted out {e['out_blind']} instead,"
                       " so your body cost you this one.")
        moments.append({"verdict": BAD, "title": "You were voted out", "detail": detail})
    elif mafia:
        if against >= 2:
            moments.append({"verdict": MEH, "title": f"{against} votes on you",
                            "detail": f"You survived, but it was close. The town removed {out} instead."})
        else:
            moments.append({"verdict": GOOD, "title": f"The town turned on {out}",
                            "detail": f"An innocent villager went out and only {against} voted for you. "
                                      "Exactly what the mafia wants."})
    else:
        moments.append({"verdict": GOOD if e["out_mafia"] else BAD,
                        "title": f"The town voted out {out}",
                        "detail": "They were the mafia." if e["out_mafia"] else "They were innocent."})
    return [{"type": "vote", "day": e["day"], **m} for m in moments]


def _tells(s: MafiaSession, answers: list) -> list[dict]:
    """The cues that pushed the detector hardest in this game, by weight x your average value."""
    if not answers:
        return []
    coef = s.model.coefficients()
    push = {}
    for k, w in coef.items():
        vals = [u.features.get(k) for u in answers if u.features.get(k) is not None]
        if not vals:
            continue
        if k == "llm":
            continue
        mean = sum(vals) / len(vals)
        if not _matches_tip(k, mean):
            continue
        push[k] = (w * (mean - CENTRE.get(k, 0.0)) * SCALE.get(k, 1.0), mean)
    llm = [u.llm for u in answers if u.llm is not None]
    if llm and "llm" in coef:
        mean = sum(llm) / len(llm)
        push["llm"] = (coef["llm"] * (mean - 0.5) * 4, mean)
    ranked = sorted(push.items(), key=lambda kv: kv[1][0], reverse=True)
    out = []
    for k, (score, mean) in ranked[:3]:
        if score <= 0.05:
            break
        if k.endswith("_z"):
            how = f"{'+' if mean > 0 else ''}{mean:.1f}σ from your normal"
        elif k == "llm":
            how = f"wording judged {round(mean * 100)}% suspicious"
        else:
            how = "above your usual"
        out.append({"key": k, "label": LABELS.get(k, k), "how": how, "tip": TIPS.get(k, "")})
    return out


def breakdown(s: MafiaSession, number: int | None = None) -> dict:
    game = _find_game(s, number)
    if game is None:
        return {"game": None}
    over = game.phase == "over"
    you_won = over and (game.winner == "mafia") == game.you_mafia
    moments = []
    for e in game.log:
        if e["type"] == "answer":
            moments.append(_answer_moment(s.utterances[e["utterance"]], e, game.you_mafia))
        elif e["type"] == "silent":
            moments.append({"type": "answer", "day": e["day"], "verdict": BAD, "title": "You said nothing",
                            "detail": "Silence made every player more suspicious of you.", "question": e["question"],
                            "said": "", "p_lie": None, "cues": []})
        elif e["type"] == "vote":
            moments.extend(_vote_moment(e, game))
        elif e["type"] == "kill":
            if e["by_you"]:
                moments.append({"type": "kill", "day": e["day"], "verdict": MEH,
                                "title": f"You killed {e['victim']} in the night", "detail": ""})
            else:
                moments.append({"type": "kill", "day": e["day"], "verdict": MEH,
                                "title": f"The mafia killed {e['victim']} in the night", "detail": ""})
    answers = [s.utterances[e["utterance"]] for e in game.log if e["type"] == "answer"]
    if game.you_mafia:
        lies = [u for u in answers if u.is_lie]
        scored = lies
    else:
        scored = [u for u in answers if u.is_lie is not None]
    caught = [u for u in scored if (u.p_lie or 0) >= 0.5]
    tells = _tells(s, scored)
    right = [m["title"] + (f": {m['detail']}" if m["detail"] else "") for m in moments if m["verdict"] == GOOD]
    wrong = [m["title"] + (f": {m['detail']}" if m["detail"] else "") for m in moments if m["verdict"] == BAD]
    if game.you_mafia:
        score_line = f"The detector caught {len(caught)} of your {len(scored)} lies."
    else:
        score_line = (f"{len(caught)} of your {len(scored)} honest answers looked like lies."
                      if scored else "You didn't give the detector much to read.")
    return {
        "game": game.number,
        "role": "mafia" if game.you_mafia else "villager",
        "mafia": game.mafia,
        "winner": game.winner,
        "you_won": you_won,
        "over": over,
        "days": game.day,
        "survived": game.alive[YOU],
        "score_line": score_line,
        "moments": moments,
        "right": right,
        "wrong": wrong,
        "tells": tells,
        "games": [g.number for g in s.games if g.phase == "over"],
    }
