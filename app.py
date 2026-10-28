"""Scam Call Checker UI.  Run from the repo root:  python app.py"""

from pathlib import Path

import gradio as gr

from scamcheck.analyse import GEMMA, analyse
from scamcheck.transcribe import transcribe
from scamcheck.voice import fake_voice_score

ICON = {"SCAM": "🚨", "SUSPICIOUS": "⚠️", "LIKELY SAFE": "✅"}
AUDIO_TYPES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
SAMPLES = sorted(
    str(p) for p in (Path(__file__).parent / "samples").glob("*") if p.suffix.lower() in AUDIO_TYPES
)


def run(path):
    if not path:
        return "Record or upload a clip first.", "", {}

    voice = fake_voice_score(path)
    text = transcribe(path)
    voice_line = f"**Voice (AASIST):** {voice['fake_prob']:.0%} likely AI-generated"

    try:
        r = analyse(text, voice)
    except Exception as e:  # Ollama not running, model not pulled, bad JSON...
        md = (
            f"## Gemma unavailable\n\n`{e}`\n\n"
            f"Is Ollama running, and has `{GEMMA}` been pulled?\n\n{voice_line}."
        )
        return md, text, {"voice": voice}

    flags = "\n".join(f'- **"{f["quote"]}"** — {f["why"]}' for f in r["red_flags"]) or "- None found"
    md = (
        f"## {ICON.get(r['verdict'], '')} {r['verdict']} — risk {r['risk_score']}/100\n\n"
        f"{voice_line}. {r['voice_note']}\n\n"
        f"**Red flags:**\n{flags}\n\n"
        f"**What to do:** {r['advice']}"
    )
    return md, text, {"voice": voice, "verdict": r}


demo = gr.Interface(
    fn=run,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Call audio"),
    outputs=[gr.Markdown(label="Verdict"), gr.Textbox(label="Transcript"), gr.JSON(label="Raw")],
    examples=[[s] for s in SAMPLES] or None,
    title="Scam Call Checker",
    description="AASIST fake-voice detection + Whisper + Gemma 4 scam reasoning. Runs fully on this laptop.",
)

if __name__ == "__main__":
    demo.launch()
