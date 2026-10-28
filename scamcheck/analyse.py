"""Gemma 4 (via Ollama) reads the transcript + voice score and returns a structured verdict.

The instructions live in skills/scam-call-analyst/SKILL.md (Agent Skills format), so the
prompt can be tuned without touching Python. It is re-read on every call: edit and retry.
"""

import json
import os
import re
from pathlib import Path

import ollama

GEMMA = os.getenv("GEMMA_MODEL", "gemma4")  # e.g. GEMMA_MODEL=gemma4:e2b on weaker laptops
SKILL = Path(__file__).resolve().parent.parent / "skills" / "scam-call-analyst" / "SKILL.md"

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


def analyse(transcript: str, voice: dict | None) -> dict:
    transcript = transcript.strip() or "(no speech detected)"
    if voice:
        voice_line = (
            f"AI-voice detector ({voice.get('detector', 'AASIST')}): {voice['fake_prob']:.0%} average "
            f"probability of synthetic speech, {voice['max_fake_prob']:.0%} in the worst window."
        )
    else:
        voice_line = "AI-voice detector: unavailable, judge on the transcript only."
    user = f'{voice_line}\n\nTranscript:\n"""{transcript}"""'
    kwargs = dict(
        model=GEMMA,
        messages=[
            {"role": "system", "content": _instructions()},
            {"role": "user", "content": user},
        ],
        format=SCHEMA,
        options={"temperature": 0, "num_predict": 800},  # cap output length so calls stay quick
        keep_alive="30m",  # keep Gemma in memory between checks (no reload lag mid-demo)
    )
    try:
        resp = ollama.chat(think=False, **kwargs)  # skip the thinking step: much faster on a laptop
    except Exception as e:
        if "think" not in str(e).lower():
            raise
        resp = ollama.chat(**kwargs)  # model/Ollama version without a thinking switch
    return json.loads(resp["message"]["content"])


if __name__ == "__main__":
    # Quick prompt test without audio:  python -m scamcheck.analyse "transcript text here"
    import sys

    import time

    text = " ".join(sys.argv[1:]) or "Hi, this is HMRC. You owe unpaid tax and will be arrested today unless you pay in gift cards."
    print(f"Asking {GEMMA}... (the first call loads the model into memory, can take a minute)", flush=True)
    t = time.time()
    result = analyse(text, {"fake_prob": 0.5, "max_fake_prob": 0.5})
    print(json.dumps(result, indent=2))
    print(f"took {time.time() - t:.1f}s")
