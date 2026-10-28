"""Scam Call Checker UI.  Run from the repo root:  python app.py

Tab 1 "Live call": listens to a mic / loopback device and updates the verdict as the call goes.
Tab 2 "Check a recording": record or upload a clip and analyse it in one go.

The page is plain Gradio plus hand-written HTML panels; the look lives in app.css.
"""

import html
import inspect
import os
import re
import time
from pathlib import Path

import gradio as gr

from scamcheck.analyse import GEMMA, analyse
from scamcheck.detectors import get_detector
from scamcheck.rules import AI_VOICE_HIGH
from scamcheck.transcribe import transcribe, warm_up

ROOT = Path(__file__).parent
AUDIO_TYPES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
SAMPLES = sorted(str(p) for p in (ROOT / "samples").glob("*") if p.suffix.lower() in AUDIO_TYPES)
GEMMA_HINT = f"Is Ollama running, and has <code>{html.escape(GEMMA)}</code> been pulled?"
DETECTOR = os.getenv("VOICE_DETECTOR", "aasist")

TONE = {"SCAM": "hot", "SUSPICIOUS": "warn", "LIKELY SAFE": "ok"}
NEXT_STEPS = {
    "SCAM": ["Hang up now", "Call 159 for your bank", "Never share codes"],
    "SUSPICIOUS": ["Don't pay anything", "Call back on a number you trust"],
}
METER_SEGMENTS = 24
VOICE_BARS = 24  # most recent chunks / windows shown in the voice chart

# Small stroke icons (Lucide-style), inlined so the page works offline.
_I = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{}</svg>'
ICONS = {
    "mic": _I.format('<rect x="9" y="2" width="6" height="12" rx="3"/><path d="M19 10v1a7 7 0 0 1-14 0v-1"/><path d="M12 18v4"/>'),
    "shield": _I.format('<path d="M20 13c0 5-3.5 7.5-7.7 9a1 1 0 0 1-.6 0C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.2-2.7a1.2 1.2 0 0 1 1.6 0C14.5 3.8 17 5 19 5a1 1 0 0 1 1 1z"/>'),
    "wave": _I.format('<path d="M2 10v3"/><path d="M6 6v11"/><path d="M10 3v18"/><path d="M14 8v7"/><path d="M18 5v13"/><path d="M22 10v3"/>'),
    "file": _I.format('<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7z"/><path d="M14 2v4a2 2 0 0 0 2 2h4"/><path d="M10 13v3"/><path d="M14 11v7"/>'),
    "phone": _I.format('<path d="M22 16.9v3a2 2 0 0 1-2.2 2 19.8 19.8 0 0 1-8.6-3.1 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.1 4.2 2 2 0 0 1 4.1 2h3a2 2 0 0 1 2 1.7c.1 1 .4 1.9.7 2.8a2 2 0 0 1-.5 2.1L8 9.9a16 16 0 0 0 6 6l1.3-1.3a2 2 0 0 1 2.1-.4c.9.3 1.8.6 2.8.7a2 2 0 0 1 1.7 2z"/>'),
    "text": _I.format('<path d="M17 6.1H3"/><path d="M21 12.1H3"/><path d="M15.1 18H3"/>'),
}


def _e(value) -> str:
    return html.escape(str(value if value is not None else ""))


def _pct(p: float) -> int:
    return round(float(p) * 100)


def card_head(icon: str, title: str, sub: str, right: str = "") -> str:
    return (f'<div class="head"><span class="ico">{ICONS[icon]}</span>'
            f'<div class="head-text"><div class="h-title">{title}</div><div class="h-sub">{sub}</div></div>'
            f'<div class="h-right">{right}</div></div>')


def _clock(started: float | None) -> str:
    if not started:
        return "0:00"
    s = int(time.time() - started)
    return f"{s // 60}:{s % 60:02d}"


# ---------- Illustration (isometric phone, line-art like the rest of the UI) ----------
def _iso_layer(dy: int, inner: str) -> str:
    """One rounded slab seen from above at 30 degrees: an edge 6px below, then the face and its contents."""
    iso = "matrix(.866 .5 -.866 .5 245 {})"
    rect = '<rect x="0" y="0" width="110" height="190" rx="18" class="{}"/>'
    return (f'<g transform="{iso.format(dy + 6)}">{rect.format("iso-slab")}</g>'
            f'<g transform="{iso.format(dy)}">{rect.format("iso-face")}{inner}</g>')


def illustration() -> str:
    base = '<rect x="22" y="120" width="66" height="34" rx="9" class="iso-line"/><path d="M34 132h30M34 142h20" class="iso-line"/>'
    keys = "".join(f'<rect x="{x}" y="{y}" width="24" height="24" rx="7" class="iso-blue"/>'
                   for x, y in [(18, 70), (50, 86), (82, 102)])
    bars = [18, 34, 54, 30, 62, 40, 22, 46, 28]
    wave = "".join(f'<rect x="{16 + i * 9}" y="{60 - h / 2:.0f}" width="5" height="{h}" rx="2.5" class="iso-blue"/>'
                   for i, h in enumerate(bars))
    screen = ('<path d="M20 112h46M20 126h70M20 140h34" class="iso-line"/>'
              '<rect x="20" y="152" width="40" height="14" rx="7" class="iso-blue"/>')
    return ('<svg class="illus" viewBox="0 0 420 270" aria-hidden="true">'
            + _iso_layer(88, base) + _iso_layer(54, keys) + _iso_layer(20, wave + screen) + "</svg>")


# ---------- Voice panel ----------
def voice_panel(probs: list[float], summary: dict | None, caption: str, note: str = "") -> str:
    probs = [float(p) for p in probs][-VOICE_BARS:]
    hot = sum(p >= AI_VOICE_HIGH for p in probs)
    right = f'<span class="mono dim">{hot} of {len(probs)} over {_pct(AI_VOICE_HIGH)}%</span>' if probs else ""
    if summary:
        avg, worst = summary["fake_prob"], summary.get("max_fake_prob", summary["fake_prob"])
        big = (f'<div class="big"><span class="num mono">{_pct(avg)}<small>%</small></span>'
               f'<span class="delta"><b class="mono">worst {_pct(worst)}%</b><span>{_e(caption)}</span></span></div>')
        thumb = min(max(avg, 0.0), 1.0) * 100
    else:
        big = ('<div class="big"><span class="num mono dim">--<small>%</small></span>'
               '<span class="delta"><b class="mono">no audio yet</b><span>scored as audio comes in</span></span></div>')
        thumb = None

    if probs:
        bars = "".join(
            f'<i class="bar{" hot" if p >= AI_VOICE_HIGH else ""}{" now" if i == len(probs) - 1 else ""}" '
            f'style="height:{max(6.0, p * 100):.0f}%" title="{_pct(p)}% AI"></i>'
            for i, p in enumerate(probs))
    else:
        bars = "".join(f'<i class="bar ghost" style="height:{h}%"></i>'
                       for h in [22, 30, 18, 40, 26, 34, 20, 44, 28, 36, 24, 30])
    chart = (f'<div class="bars" style="--limit:{_pct(AI_VOICE_HIGH)}%">{bars}<span class="limit"></span></div>'
             f'<div class="axis mono"><span>older</span><span>{_pct(AI_VOICE_HIGH)}% line</span><span>now</span></div>')

    ticks = "".join(f'<i class="{"major" if i % 4 == 0 else ""}"></i>' for i in range(21))
    knob = f'<span class="knob" style="left:{thumb:.1f}%"></span>' if thumb is not None else ""
    fill = f'<span class="fill" style="width:{thumb:.1f}%"></span>' if thumb is not None else ""
    scale = (f'<div class="sect"><span>Human or AI</span><span class="mono blue">'
             f'{f"{avg:.2f}" if summary else "-"}</span></div>'
             f'<div class="slider"><span class="track"></span>{fill}{knob}</div>'
             f'<div class="ticks">{ticks}</div><div class="ends"><span>Human</span><span>AI-generated</span></div>')
    foot = f'<p class="note-text">{_e(note)}</p>' if note else ""
    return (f'<div class="card">{card_head("wave", "AI voice", f"{_e(DETECTOR.upper())} fake-voice detector")}'
            f'<div class="sect"><span>Per chunk</span>{right}</div>{big}{chart}{scale}{foot}</div>')


# ---------- Verdict panel ----------
def _note(tone: str, body: str) -> str:
    return f'<div class="note {tone}"><i class="dot"></i><div>{body}</div></div>'


def _status_row(state: str, label: str, meta: str) -> str:
    stop = '<span class="stop-hint">Stop <span class="kbd">esc</span></span>' if state == "live" else ""
    return (f'<div class="status {state}"><i class="dot"></i><b>{_e(label)}</b>'
            f'<span class="mono dim">{meta}</span><span class="grow"></span>{stop}</div>')


def verdict_panel(verdict: dict | None, *, chip: str, chip_meta: str, status: str = "",
                  error: str | None = None, empty: str | None = None) -> str:
    top = (f'<div class="v-top"><span class="chip">{ICONS["file"]}<b>{_e(chip)}</b>'
           f'<span class="dim">{_e(chip_meta)}</span></span>'
           f'<span class="from">{_e(DETECTOR)} + {_e(GEMMA)}</span></div>')
    if empty is not None:
        return (f'<div class="card verdict">{top}{status}<div class="empty"><div class="inset art">{illustration()}</div>'
                f'<h3>Is this call a scam?</h3><p>{empty}</p></div></div>')

    notes = []
    if verdict and verdict.get("escalation"):
        said = verdict.get("escalated_from")
        notes.append(_note("hot", _e(verdict["escalation"])
                           + (f' <span class="dim">Gemma alone said {_e(said)}.</span>' if said else "")))
    if verdict and verdict.get("llm_error"):
        notes.append(_note("warn", f"Gemma unavailable, so this verdict is from the safety rules only. {GEMMA_HINT}"
                                   f'<code class="err">{_e(verdict["llm_error"])}</code>'))
    if error:
        hint = f" {GEMMA_HINT}" if "Gemma" in error else ""
        notes.append(_note("warn", f'<code class="err">{_e(error)}</code>{hint}'))

    if not verdict:
        hero = ('<div class="hero waiting"><span class="num mono dim">--</span>'
                '<span class="delta"><b class="mono">/100 risk</b><span>Waiting for Gemma\'s first verdict</span></span></div>'
                f'<div class="meter shimmer">{"<i></i>" * METER_SEGMENTS}</div>')
        return f'<div class="card verdict">{top}{status}{hero}{"".join(notes)}</div>'

    v = verdict.get("verdict", "SUSPICIOUS")
    tone = TONE.get(v, "warn")
    try:
        risk = max(0, min(100, int(verdict.get("risk_score", 0))))
    except (TypeError, ValueError):
        risk = 0
    lit = round(risk / 100 * METER_SEGMENTS)
    meter = "".join(f'<i class="{"on" if i < lit else ""}"></i>' for i in range(METER_SEGMENTS))
    source = {"rules": "safety rules", "gemma+rules": "Gemma + safety rules", "gemma": "Gemma"}.get(
        verdict.get("source"), "safety rules" if verdict.get("escalation") else "Gemma")
    hero = (f'<div class="hero"><span class="num mono">{risk}</span>'
            f'<span class="delta"><b class="mono">/100 risk</b><span>judged by {_e(source)}</span></span>'
            f'<span class="grow"></span><span class="pill">{_e(v)}</span></div>'
            f'<div class="meter">{meter}</div>')

    flags = [f for f in verdict.get("red_flags") or [] if isinstance(f, dict)]
    flag_html = "".join(f'<blockquote class="flag"><p>&ldquo;{_e(f.get("quote"))}&rdquo;</p>'
                        f'<span>{_e(f.get("why"))}</span></blockquote>' for f in flags)
    flags_block = (f'<div class="sect"><span>Red flags</span><span class="mono dim">{len(flags)}</span></div>'
                   + (flag_html or '<p class="none">None found.</p>'))
    steps = NEXT_STEPS.get(v)
    then = (f'<div class="then"><span class="dim">Then</span>'
            + "".join(f'<span class="tag">{ICONS["shield"]}{_e(s)}</span>' for s in steps) + "</div>") if steps else ""
    advice = (f'<hr><div class="sect"><span>What to do</span></div>'
              f'<p class="advice">{_e(verdict.get("advice"))}</p>{then}')
    return f'<div class="card verdict tone-{tone}">{top}{status}{hero}{"".join(notes)}{flags_block}{advice}</div>'


# ---------- Transcript panel ----------
def _highlight(text: str, quotes: list[str]) -> str:
    """Escape the transcript and mark every place a red-flag quote appears in it."""
    spans = []
    low = text.lower()
    for q in quotes:
        q = q.strip(" .,;\"'").lower()
        if len(q) < 3:
            continue
        for m in re.finditer(re.escape(q), low):
            spans.append((m.start(), m.end()))
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))
    out, pos = [], 0
    for s, e in merged:
        out += [_e(text[pos:s]), f"<mark>{_e(text[s:e])}</mark>"]
        pos = e
    out.append(_e(text[pos:]))
    return "".join(out)


def transcript_panel(text: str, verdict: dict | None, placeholder: str) -> str:
    text = (text or "").strip()
    quotes = [str(f.get("quote", "")) for f in (verdict or {}).get("red_flags") or [] if isinstance(f, dict)]
    words = len(text.split())
    body = _highlight(text, quotes) if text else f'<span class="dim">{_e(placeholder)}</span>'
    return (f'<div class="card">{card_head("text", "Transcript", "Whisper speech-to-text, red flags marked")}'
            f'<div class="inset tx"><div>{body}</div></div>'
            f'<div class="tx-foot mono dim"><span>{words} words</span><span><i class="mark-key"></i>red flag</span></div></div>')


# ---------- Check a recording ----------
def check_recording(path):
    name = Path(path).name if path else "Recording"
    if not path:
        return (voice_panel([], None, ""),
                verdict_panel(None, chip=name, chip_meta="no clip", empty="Record or upload a clip first, then press <b>Check</b>."),
                transcript_panel("", None, "The transcript of the clip shows up here."), {})
    voice = get_detector().score_file(path)
    text = transcribe(path)
    meta = f"{len(text.split())} words"
    vp = voice_panel(voice.get("windows", []), voice, "average over the clip")
    try:
        verdict = analyse(text, voice)
    except Exception as e:  # Ollama not running, model not pulled, bad JSON...
        return (vp, verdict_panel(None, chip=name, chip_meta=meta, error=f"Gemma: {e}"),
                transcript_panel(text, None, "No speech found."), {"voice": voice})
    vp = voice_panel(voice.get("windows", []), voice, "average over the clip", verdict.get("voice_note", ""))
    return (vp, verdict_panel(verdict, chip=name, chip_meta=meta),
            transcript_panel(text, verdict, "No speech found."), {"voice": voice, "verdict": verdict})


# ---------- Live call ----------
SYSTEM_CHOICE = "System audio: whatever this laptop is playing (Discord, YouTube, calls)"
CHOICES = {"Default microphone": None}  # dropdown label -> device for LiveMonitor
level_line = lambda state: ""
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
LIVE_EMPTY = ("Put the call on speaker next to the laptop, or pick a loopback device that carries the call "
              "audio, then press <b>Start listening</b>. The verdict updates as the call goes.")


def live_view():
    if LIVE_ERROR or MONITOR is None:
        empty = (_note("warn", f'<code class="err">{_e(LIVE_ERROR)}</code>') + LIVE_EMPTY) if LIVE_ERROR else LIVE_EMPTY
        return (voice_panel([], None, ""),
                verdict_panel(None, chip="Live call", chip_meta="not listening", empty=empty),
                transcript_panel("", None, "Words appear here as the caller speaks."))
    s = MONITOR.state
    listening = s["status"] != "stopped"
    words = len(s["transcript"].split())
    where = s["status"].removeprefix("listening on: ")
    level = level_line(s)
    meta = " · ".join(x for x in (where, level, f"📲 {s['alert']}" if s.get("alert") else "") if x)
    status = _status_row("live" if listening else "stopped", "Listening" if listening else "Stopped",
                         _e(meta) if listening else _clock(s["started"]))
    note = (s["verdict"] or {}).get("voice_note", "")
    return (voice_panel(s["voice"], s["voice_summary"], "average, last 30 s", note),
            verdict_panel(s["verdict"], chip="Live call", chip_meta=f"{_clock(s['started'])} · {words} words",
                          status=status, error=s["error"]),
            transcript_panel(s["transcript"], s["verdict"], "Listening... words appear here as the caller speaks."))


def log_state(s):
    """Mirror live progress in the terminal, so you can see it's working even if the page lags."""
    vs = s["voice_summary"]
    v = f"{vs['fake_prob']:.0%} fake" if vs else "-"
    verdict = s["verdict"]["verdict"] if s["verdict"] else "-"
    raised = " (raised by safety rules)" if s["verdict"] and s["verdict"].get("escalation") else ""
    level = f"level {s['level']:.3f}" if s.get("level") is not None else "no audio yet"
    print(f"[live] {level} | voice {v} | {len(s['transcript'].split())} words | verdict {verdict}{raised}"
          + (f" | {s['alert']}" if s.get("alert") else "")
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
        voice, _, tx = live_view()
        yield voice, verdict_panel(None, chip="Live call", chip_meta="not listening",
                                   empty=_note("warn", f'Couldn\'t start listening: <code class="err">{_e(e)}</code>')
                                   + LIVE_EMPTY), tx
        return
    while MONITOR is mon and mon.state["status"] != "stopped":
        try:
            yield live_view()
        except Exception as e:  # a display glitch shouldn't kill the live stream
            print(f"[live] display error: {type(e).__name__}: {e}", flush=True)
        time.sleep(0.5)
    yield live_view()


def stop_live():
    if MONITOR is not None:
        MONITOR.stop()
    return live_view()


# ---------- Layout ----------
CSS = (ROOT / "app.css").read_text(encoding="utf-8")
HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<script>
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") document.querySelector("#stop-btn")?.click();
});
</script>
"""
# Gradio 6 takes css/head in launch(), Gradio 5 in Blocks().
STYLE = {"css": CSS, "head": HEAD}
STYLE_IN_LAUNCH = "css" in inspect.signature(gr.Blocks.launch).parameters

with gr.Blocks(title="Scam Call Checker", **({} if STYLE_IN_LAUNCH else STYLE)) as demo:
    gr.HTML(card_head("shield", "Scam Call Checker",
                      f"Fake-voice detection ({_e(DETECTOR)}) + Whisper + Gemma 4 ({_e(GEMMA)}). "
                      "All open models, running on this laptop.",
                      '<span class="chip local"><i class="dot"></i>Nothing leaves this laptop</span>'),
            elem_classes="page-head")

    with gr.Tabs(elem_classes="modes"):
        with gr.Tab("Live call"):
            with gr.Row(equal_height=False, elem_classes="grid"):
                with gr.Column(scale=5, min_width=320, elem_classes="stack"):
                    with gr.Column(elem_classes="card controls"):
                        gr.HTML(card_head("mic", "Listen to a call", "Phone on speaker, or a loopback device"))
                        gr.HTML('<div class="label">Listen on</div>')
                        device = gr.Dropdown(DEVICES, value=DEVICES[0], show_label=False, container=False,
                                             elem_classes="field")
                        with gr.Row(elem_classes="actions"):
                            stop_btn = gr.Button("Stop", variant="secondary", elem_id="stop-btn")
                            start_btn = gr.Button("Start listening", variant="primary", elem_id="start-btn")
                    live_voice = gr.HTML(elem_classes="host")
                with gr.Column(scale=7, min_width=320, elem_classes="stack"):
                    live_verdict = gr.HTML(elem_classes="host")
                    live_tx = gr.HTML(elem_classes="host")
            live_out = [live_voice, live_verdict, live_tx]
            start_btn.click(start_live, device, live_out, show_progress="hidden")
            stop_btn.click(stop_live, None, live_out, show_progress="hidden")

        with gr.Tab("Check a recording"):
            with gr.Row(equal_height=False, elem_classes="grid"):
                with gr.Column(scale=5, min_width=320, elem_classes="stack"):
                    with gr.Column(elem_classes="card controls"):
                        gr.HTML(card_head("phone", "Check a recording", "Record or upload a clip of the call"))
                        audio = gr.Audio(sources=["microphone", "upload"], type="filepath", show_label=False,
                                         elem_classes="field audio")
                        with gr.Row(elem_classes="actions"):
                            check_btn = gr.Button("Check", variant="primary", elem_id="check-btn")
                        if SAMPLES:
                            gr.Examples([[s] for s in SAMPLES], inputs=audio, label="Demo clips")
                    rec_voice = gr.HTML(elem_classes="host")
                with gr.Column(scale=7, min_width=320, elem_classes="stack"):
                    rec_verdict = gr.HTML(elem_classes="host")
                    rec_tx = gr.HTML(elem_classes="host")
                    with gr.Accordion("Raw output", open=False, elem_classes="card raw"):
                        rec_js = gr.JSON(show_label=False)
            check_btn.click(check_recording, audio, [rec_voice, rec_verdict, rec_tx, rec_js])

    demo.load(live_view, None, live_out)
    demo.load(lambda: check_recording(None)[:3], None, [rec_voice, rec_verdict, rec_tx])

if __name__ == "__main__":
    demo.launch(**(STYLE if STYLE_IN_LAUNCH else {}))
