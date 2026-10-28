"""Scam Call Checker UI.  Run from the repo root:  python app.py

Tab 1 "Live call": listens to a mic / loopback device and updates the verdict as the call goes.
Tab 2 "Check a recording": record or upload a clip and analyse it in one go.
"""

import os
from pathlib import Path

import gradio as gr

from scamcheck.analyse import GEMMA, analyse
from scamcheck.detectors import get_detector
from scamcheck.transcribe import transcribe, warm_up

ICON = {"SCAM": "🚨", "SUSPICIOUS": "⚠️", "LIKELY SAFE": "✅"}
AUDIO_TYPES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
SAMPLES = sorted(
    str(p) for p in (Path(__file__).parent / "samples").glob("*") if p.suffix.lower() in AUDIO_TYPES
)
GEMMA_HINT = f"Is Ollama running, and has `{GEMMA}` been pulled?"


def render(verdict: dict | None, voice: dict | None, error: str | None = None) -> str:
    parts = []
    if verdict:
        parts.append(f"## {ICON.get(verdict['verdict'], '')} {verdict['verdict']} — risk {verdict['risk_score']}/100")
    if voice:
        line = (f"**Voice ({voice['detector']}):** {voice['fake_prob']:.0%} likely AI-generated "
                f"(worst window {voice['max_fake_prob']:.0%}).")
        parts.append(line + (f" {verdict['voice_note']}" if verdict else ""))
    if verdict:
        flags = "\n".join(f'- **"{f["quote"]}"** — {f["why"]}' for f in verdict["red_flags"]) or "- None found"
        parts += [f"**Red flags:**\n{flags}", f"**What to do:** {verdict['advice']}"]
    if error:
        parts.append(f"⚠️ `{error}`" + (f"\n\n{GEMMA_HINT}" if "Gemma" in error else ""))
    return "\n\n".join(parts)


# ---------- Check a recording ----------
def check_recording(path):
    if not path:
        return "Record or upload a clip first.", "", {}
    voice = get_detector().score_file(path)
    text = transcribe(path)
    try:
        verdict = analyse(text, voice)
        return render(verdict, voice), text, {"voice": voice, "verdict": verdict}
    except Exception as e:  # Ollama not running, model not pulled, bad JSON...
        return render(None, voice, f"Gemma: {e}"), text, {"voice": voice}


# ---------- Live call ----------
try:
    from scamcheck.live import LiveMonitor, input_devices

    DEVICES = ["Default microphone"] + [f"{i}: {name}" for i, name in input_devices()]
    LIVE_ERROR = None
except Exception as e:  # e.g. PortAudio missing
    DEVICES, LIVE_ERROR = ["Default microphone"], f"Live mode unavailable: {e}"

MONITOR = None


def live_view():
    if LIVE_ERROR:
        return LIVE_ERROR, ""
    if MONITOR is None:
        return "Not listening. Press **Start listening**.", ""
    s = MONITOR.state
    body = render(s["verdict"], s["voice_summary"], s["error"])
    if not s["verdict"] and s["transcript"]:
        body = "_Waiting for Gemma's first verdict..._\n\n" + body
    return f"**Status:** {s['status']}\n\n{body}", s["transcript"]


def start_live(choice):
    global MONITOR
    if LIVE_ERROR:
        return live_view()
    if MONITOR is not None:
        MONITOR.stop()
    device = None if choice == DEVICES[0] else int(choice.split(":", 1)[0])
    get_detector()
    warm_up()  # load models before audio starts piling up
    MONITOR = LiveMonitor(device=device)
    MONITOR.start()
    return live_view()


def stop_live():
    if MONITOR is not None:
        MONITOR.stop()
    return live_view()


# ---------- Layout ----------
with gr.Blocks(title="Scam Call Checker") as demo:
    gr.Markdown(
        "# Scam Call Checker\n"
        f"Fake-voice detection ({os.getenv('VOICE_DETECTOR', 'aasist')}) + Whisper + "
        f"Gemma 4 ({GEMMA}) scam reasoning. All open models, running on this laptop."
    )

    with gr.Tab("Live call"):
        gr.Markdown("Put the call on speaker next to the laptop, or pick a loopback device "
                    "that carries the call audio (see README).")
        device = gr.Dropdown(DEVICES, value=DEVICES[0], label="Listen on")
        with gr.Row():
            start_btn = gr.Button("Start listening", variant="primary")
            stop_btn = gr.Button("Stop")
        live_md = gr.Markdown()
        live_tx = gr.Textbox(label="Live transcript", lines=6)
        start_btn.click(start_live, device, [live_md, live_tx])
        stop_btn.click(stop_live, None, [live_md, live_tx])
        gr.Timer(1.0).tick(live_view, None, [live_md, live_tx], show_progress="hidden")

    with gr.Tab("Check a recording"):
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Call audio")
        check_btn = gr.Button("Check", variant="primary")
        rec_md = gr.Markdown()
        rec_tx = gr.Textbox(label="Transcript")
        rec_js = gr.JSON(label="Raw")
        check_btn.click(check_recording, audio, [rec_md, rec_tx, rec_js])
        if SAMPLES:
            gr.Examples([[s] for s in SAMPLES], inputs=audio)

if __name__ == "__main__":
    demo.launch()
