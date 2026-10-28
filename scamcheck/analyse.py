"""Gemma 4 (via Ollama) judges the call, then the safety rules make sure it can't under-call a scam.

Flow: transcript -> rules.find_cues (known scam phrases, read-out numbers)
      -> Gemma 4 gets voice score + cues + transcript, follows skills/scam-call-analyst/SKILL.md
      -> rules.apply_floor raises the verdict if needed (e.g. AI voice + offer = SCAM)
If Gemma is unavailable, a rules-only verdict is returned instead of failing.

The instructions live in SKILL.md (Agent Skills format) and are re-read on every call: edit and retry.
"""

import json
import os
import re
from pathlib import Path

import ollama

from .rules import AI_VOICE_HIGH, AI_VOICE_MEDIUM, apply_floor, find_cues, rules_only_verdict

GEMMA = os.getenv("GEMMA_MODEL", "gemma4")  # e.g. GEMMA_MODEL=gemma4:e2b on weaker laptops
SKILL = Path(__file__).resolve().parent.parent / "skills" / "scam-call-analyst" / "SKILL.md"
MAX_TRANSCRIPT_CHARS = 6000  # keep long calls within Gemma's context: the most recent part matters most

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["SCAM", "SUSPICIOUS", "LIKELY SAFE"]},
        "risk_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "red_flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"quote": {"type": "string"}, "why": {"type": "string"}},
                "required": ["quote", "why"],
            },
        },
        "voice_note": {"type": "string"},
        "advice": {"type": "string"},
    },
    "required": ["verdict", "risk_score", "red_flags", "voice_note", "advice"],
}


def _instructions() -> str:
    """SKILL.md body without its YAML frontmatter."""
    text = SKILL.read_text(encoding="utf-8")
    return re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, flags=re.S).strip()


def _voice_line(voice: dict | None) -> str:
    if not voice:
        return "AI-voice detector: unavailable. Judge on the transcript only."
    p = voice["fake_prob"]
    if p >= AI_VOICE_HIGH:
        label = "VERY LIKELY AI-GENERATED (synthetic, cloned or robocall voice)"
    elif p >= AI_VOICE_MEDIUM:
        label = "POSSIBLY AI-GENERATED"
    else:
        label = "probably a real human voice"
    return (f"AI-voice detector ({voice.get('detector', 'AASIST')}): {p:.0%} average, "
            f"{voice.get('max_fake_prob', p):.0%} worst window -> {label}.")


def _ask_gemma(user: str) -> dict:
    kwargs = dict(
        model=GEMMA,
        messages=[{"role": "system", "content": _instructions()}, {"role": "user", "content": user}],
        format=SCHEMA,
        options={"temperature": 0, "num_predict": 800, "num_ctx": 8192},
        keep_alive="30m",  # keep Gemma in memory between checks (no reload lag mid-demo)
    )
    try:
        resp = ollama.chat(think=False, **kwargs)  # skip the thinking step: much faster on a laptop
    except Exception as e:
        if "think" not in str(e).lower():
            raise
        resp = ollama.chat(**kwargs)  # model/Ollama version without a thinking switch
    return json.loads(resp["message"]["content"])


def analyse(transcript: str, voice: dict | None) -> dict:
    transcript = transcript.strip()
    cues = find_cues(transcript)  # whole call, so early red flags aren't forgotten

    recent = transcript if len(transcript) <= MAX_TRANSCRIPT_CHARS else "..." + transcript[-MAX_TRANSCRIPT_CHARS:]
    cue_lines = "\n".join(f'- "{c["quote"]}": {c["why"]}' for c in cues) or "- none"
    user = (
        f"{_voice_line(voice)}\n\n"
        f"Automatic phrase checks flagged:\n{cue_lines}\n\n"
        f'Transcript (automatic speech-to-text, may contain errors):\n"""{recent or "(no speech detected)"}"""'
    )

    try:
        result = _ask_gemma(user)
    except Exception as e:  # Ollama down, model missing, bad JSON: still give a verdict
        result = rules_only_verdict(voice, cues)
        result["llm_error"] = f"{type(e).__name__}: {e}"
        return result
    result = apply_floor(result, voice, cues)
    result["source"] = "gemma+rules" if result.get("escalation") else "gemma"
    return result


if __name__ == "__main__":
    # Quick test without audio:
    #   python -m scamcheck.analyse "transcript text here"
    #   python -m scamcheck.analyse --voice 0.97 "transcript text here"
    import argparse
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--voice", type=float, default=0.5, help="pretend AI-voice score 0-1")
    args = ap.parse_args()
    text = " ".join(args.text) or "Hi, this is HMRC. You owe unpaid tax and will be arrested today unless you pay in gift cards."
    print(f"Asking {GEMMA}... (the first call loads the model into memory, can take a minute)", flush=True)
    t = time.time()
    out = analyse(text, {"detector": "test", "fake_prob": args.voice, "max_fake_prob": args.voice})
    print(json.dumps(out, indent=2))
    print(f"took {time.time() - t:.1f}s")
