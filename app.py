"""Scam Call Checker UI.  Run from the repo root:  python app.py

Tab 1 "Live call": listens to a mic / loopback device and updates the verdict as the call goes.
Tab 2 "Check a recording": record or upload a clip and analyse it in one go.
"""

import os
import time
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
        v = verdict.get("verdict", "?")
        parts.append(f"## {ICON.get(v, '')} {v} — risk {verdict.get('risk_score', '?')}/100")
        if verdict.get("escalation"):
            said = verdict.get("escalated_from")
            parts.append(f"🛡️ **{verdict['escalation']}**" + (f" (Gemma alone said {said})" if said else ""))
        if verdict.get("llm_error"):
            parts.append(f"⚠️ Gemma unavailable, so this verdict is from the safety rules only. {GEMMA_HINT}\n\n"
                         f"`{verdict['llm_error']}`")
    if voice:
        line = (f"**Voice ({voice.get('detector', '?')}):** {voice['fake_prob']:.0%} likely AI-generated "
                f"(worst window {voice.get('max_fake_prob', voice['fake_prob']):.0%}).")
        parts.append(line + (f" {verdict.get('voice_note', '')}" if verdict else ""))
    if verdict:
        flags = "\n".join(f'- **"{f.get("quote", "")}"** — {f.get("why", "")}'
                          for f in verdict.get("red_flags", [])) or "- None found"
        parts += [f"**Red flags:**\n{flags}", f"**What to do:** {verdict.get('advice', '')}"]
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
SYSTEM_CHOICE = "System audio: whatever this laptop is playing (Discord, YouTube, calls)"
CHOICES = {"Default microphone": None}  # dropdown label -> device for LiveMonitor
try:
    from scamcheck.live import SYSTEM_AUDIO, LiveMonitor, input_devices, level_line, loopback_devices

    if os.name == "nt":
        CHOICES[SYSTEM_CHOICE] = SYSTEM_AUDIO
        for i, name in loopback_devices():  # each speaker/headset, e.g. the one Discord plays through
            CHOICES[f"System audio: {name} (#{i})"] = f"{SYSTEM_AUDIO}:{i}"
    for i, name in input_devices():
        CHOICES[f"Mic {i}: {name}"] = i
    LIVE_ERROR = None
except Exception as e:  # e.g. PortAudio missing
    LIVE_ERROR = f"Live mode unavailable: {e}"
DEVICES = list(CHOICES)

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
    level = level_line(s)
    return f"**Status:** {s['status']}" + (f"  \n`{level}`" if level else "") + f"\n\n{body}", s["transcript"]


def log_state(s):
    """Mirror live progress in the terminal, so you can see it's working even if the page lags."""
    vs = s["voice_summary"]
    v = f"{vs['fake_prob']:.0%} fake" if vs else "-"
    verdict = s["verdict"]["verdict"] if s["verdict"] else "-"
    raised = " (raised by safety rules)" if s["verdict"] and s["verdict"].get("escalation") else ""
    level = f"level {s['level']:.3f}" if s.get("level") is not None else "no audio yet"
    print(f"[live] {level} | voice {v} | {len(s['transcript'].split())} words | verdict {verdict}{raised}"
          + (f" | error: {s['error']}" if s["error"] else ""), flush=True)


def start_live(choice):
    """Generator: streams fresh state to the page every second until Stop is pressed."""
    global MONITOR
    if LIVE_ERROR:
        yield live_view()
        return
    if MONITOR is not None:
        MONITOR.stop()
    device = CHOICES.get(choice)
    get_detector()
    warm_up()  # load models before audio starts piling up
    mon = LiveMonitor(device=device, on_update=log_state)
    MONITOR = mon
    try:
        mon.start()
    except Exception as e:  # e.g. PyAudioWPatch missing, device busy
        MONITOR = None
        yield f"⚠️ Couldn't start listening: `{e}`", ""
        return
    while MONITOR is mon and mon.state["status"] != "stopped":
        try:
            yield live_view()
        except Exception as e:  # a display glitch shouldn't kill the live stream
            yield f"⚠️ Display error: `{type(e).__name__}: {e}` (still listening)", mon.state["transcript"]
        time.sleep(0.5)
    yield live_view()


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
        start_btn.click(start_live, device, [live_md, live_tx], show_progress="hidden")
        stop_btn.click(stop_live, None, [live_md, live_tx])

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
