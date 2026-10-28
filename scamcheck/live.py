"""Listen to a call live and flag scams as it goes.

    python -m scamcheck.live --list-devices      # find your mic / loopback device
    python -m scamcheck.live                     # default mic (e.g. phone on speaker next to laptop)
    python -m scamcheck.live --device 3          # a specific input (Stereo Mix, BlackHole, monitor...)
    python -m scamcheck.live --device system     # Windows: whatever the laptop plays (Discord, YouTube...)

Every CHUNK_S seconds: score the voice, transcribe the new speech.
Every LLM_EVERY_S seconds (in the background): ask Gemma for an updated verdict.
"""

import argparse
import os
import queue
import sys
import threading
import time

import librosa
import numpy as np
import sounddevice as sd

from .analyse import analyse
from .detectors import get_detector
from .rules import SEVERITY, apply_floor, find_cues
from .transcribe import transcribe_array, warm_up
from . import alerts

SR = 16000
CHUNK_S = float(os.getenv("LIVE_CHUNK_S", "3"))         # process new audio every 3 s
LLM_EVERY_S = float(os.getenv("LIVE_LLM_EVERY_S", "3"))  # ask Gemma at most this often (only if new words)
VOICE_CONTEXT_S = 5  # the voice check looks at the last 5 s, so short chunks don't hurt accuracy
SILENCE_RMS = 0.003  # chunks quieter than this are skipped
RECENT = 8           # voice score = average of the last N checks (~25 s)
_EMPTY_VERDICT = {"verdict": "LIKELY SAFE", "risk_score": 0, "red_flags": [], "voice_note": "", "advice": ""}


SYSTEM_AUDIO = "system"  # special device: whatever this computer is playing (Windows)


def input_devices() -> list[tuple[int, str]]:
    """Input devices you can actually record from (Windows WDM-KS endpoints are skipped: they
    often fail with 'Failed to read capture position register')."""
    apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        h = d.get("hostapi")
        api = apis[h]["name"] if isinstance(h, int) and 0 <= h < len(apis) else ""
        if "WDM-KS" in api:
            continue
        out.append((i, d["name"]))
    return out


def loopback_devices() -> list[tuple[int, str]]:
    """Windows: every output device whose sound can be captured (speakers, headsets...).
    Needs PyAudioWPatch; returns [] elsewhere."""
    try:
        import pyaudiowpatch as pyaudio
    except ImportError:
        return []
    p = pyaudio.PyAudio()
    try:
        return [(int(d["index"]), str(d["name"])) for d in p.get_loopback_device_info_generator()]
    except Exception:
        return []
    finally:
        p.terminate()


def parse_device(spec):
    """'3' -> 3 (mic index), 'system' -> default output, 'system:12' -> output loopback #12."""
    if spec is None or isinstance(spec, int):
        return spec
    spec = str(spec).strip()
    return int(spec) if spec.isdigit() else spec


class SystemAudioCapture:
    """Records what Windows is playing (Discord, YouTube, Phone Link calls) via WASAPI loopback.

    Needs: pip install PyAudioWPatch   (Windows only)
    index=None captures the *default output device*; pass a loopback device index (see
    loopback_devices()) to capture a specific speaker/headset, e.g. the one Discord uses.
    Note: Windows sends no audio here while nothing is playing on that device.
    """

    def __init__(self, push, index: int | None = None):
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as e:
            raise RuntimeError("System audio needs: pip install PyAudioWPatch (Windows only)") from e

        self.p = pyaudio.PyAudio()
        if index is not None:
            dev = self.p.get_device_info_by_index(index)
        else:
            try:
                dev = self.p.get_default_wasapi_loopback()
            except Exception:
                wasapi = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
                speakers = self.p.get_device_info_by_index(wasapi["defaultOutputDevice"])
                dev = next(
                    (d for d in self.p.get_loopback_device_info_generator() if speakers["name"] in d["name"]),
                    None,
                )
                if dev is None:
                    self.p.terminate()
                    raise RuntimeError(f"No loopback device found for {speakers['name']}")

        self.name = dev["name"]
        self.rate = int(dev["defaultSampleRate"])
        channels = max(1, int(dev["maxInputChannels"]))

        def callback(in_data, frame_count, time_info, status):
            audio = np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels).mean(axis=1)
            push(audio.copy())
            return (None, pyaudio.paContinue)

        self.stream = self.p.open(
            format=pyaudio.paFloat32, channels=channels, rate=self.rate, input=True,
            input_device_index=dev["index"], frames_per_buffer=1024, stream_callback=callback,
        )
        self.stream.start_stream()

    def close(self):
        try:
            self.stream.stop_stream()
            self.stream.close()
        finally:
            self.p.terminate()


class LiveMonitor:
    """Captures audio in a background thread and keeps `state` up to date."""

    def __init__(self, device=None, detector=None, on_update=None):
        self.device = device
        self.detector = detector or get_detector()
        self.on_update = on_update or (lambda state: None)
        self._q: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._llm_busy = threading.Lock()
        self._alert_lock = threading.Lock()
        self._alerts_sent: set[str] = set()
        self._stream = None
        self.state = self._fresh_state()

    @staticmethod
    def _fresh_state() -> dict:
        return {"status": "idle", "transcript": "", "voice": [], "voice_summary": None,
                "verdict": None, "error": None, "started": None, "level": None, "chunks": 0,
                "alert": None}

    # --- audio capture -------------------------------------------------------------------
    def start(self):
        self.state = self._fresh_state()
        self._stop.clear()
        if isinstance(self.device, str) and self.device.startswith(SYSTEM_AUDIO):
            idx = self.device.split(":", 1)[1] if ":" in self.device else ""
            self._stream = SystemAudioCapture(self._q.put, int(idx) if idx.isdigit() else None)
            self.rate = self._stream.rate
            name = f"system audio ({self._stream.name})"
        else:
            info = sd.query_devices(self.device, "input")
            self.rate = int(info["default_samplerate"])  # open at the device's own rate, resample later
            channels = min(2, int(info["max_input_channels"]))
            self._stream = sd.InputStream(
                samplerate=self.rate, channels=channels, dtype="float32",
                device=self.device, callback=self._callback,
            )
            self._stream.start()
            name = info["name"]
        self.state.update(status=f"listening on: {name}", started=time.time())
        threading.Thread(target=self._run, daemon=True).start()
        self._emit()

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            if isinstance(self._stream, SystemAudioCapture):
                self._stream.close()
            else:
                self._stream.stop()
                self._stream.close()
            self._stream = None
        self.state["status"] = "stopped"
        self._emit()

    def _callback(self, indata, frames, time_info, status):
        self._q.put(indata.mean(axis=1).copy())  # downmix to mono

    # --- processing ----------------------------------------------------------------------
    def _run(self):
        buf = np.zeros(0, dtype=np.float32)
        tail = np.zeros(0, dtype=np.float32)  # last VOICE_CONTEXT_S seconds of speech, at 16 kHz
        need = int(CHUNK_S * self.rate)
        last_llm = 0.0
        self._llm_text = ""
        while not self._stop.is_set():
            try:
                buf = np.concatenate([buf, self._q.get(timeout=0.5)])
            except queue.Empty:
                continue
            if len(buf) < need:
                continue
            chunk, buf = buf[:need], buf[need:]
            audio = chunk if self.rate == SR else librosa.resample(chunk, orig_sr=self.rate, target_sr=SR)
            audio = np.asarray(audio, dtype=np.float32)
            level = float(np.sqrt(np.mean(audio ** 2)))
            self.state["level"], self.state["chunks"] = level, self.state["chunks"] + 1
            if level < SILENCE_RMS:
                self._emit()  # still report the level, so "nothing is reaching us" is visible
                continue  # silence: nothing to score or transcribe
            tail = np.concatenate([tail, audio])[-int(VOICE_CONTEXT_S * SR):]

            # Voice and transcript are independent: one failing shouldn't stop the other.
            try:
                v = self.detector.score(tail)
                self.state["voice"].append(v["fake_prob"])
                recent = self.state["voice"][-RECENT:]
                self.state["voice_summary"] = {
                    "detector": v["detector"],
                    "fake_prob": float(np.mean(recent)),
                    "max_fake_prob": float(np.max(self.state["voice"])),
                }
            except Exception as e:
                self.state["error"] = f"Voice detector: {type(e).__name__}: {e}"
            try:
                text = transcribe_array(audio, prompt=self.state["transcript"][-200:], fast=True)
                if text:
                    self.state["transcript"] = f"{self.state['transcript']} {text}".strip()
            except Exception as e:
                self.state["error"] = f"Whisper: {type(e).__name__}: {e}"
            self._instant_verdict()
            self._emit()

            if (self.state["transcript"] and self.state["transcript"] != self._llm_text
                    and time.time() - last_llm >= LLM_EVERY_S and not self._llm_busy.locked()):
                last_llm = time.time()
                threading.Thread(target=self._check_llm, daemon=True).start()

    def _instant_verdict(self):
        """Show the safety rules' verdict straight away (no AI needed); Gemma's fuller answer follows.

        Uses the same floor Gemma's answer is held to, so the verdict never flickers back down.
        """
        if not self.state["transcript"] and not self.state["voice_summary"]:
            return
        try:
            fv = apply_floor(dict(_EMPTY_VERDICT), self.state["voice_summary"],
                             find_cues(self.state["transcript"]))
        except Exception:
            return
        current = self.state["verdict"]
        if fv["verdict"] == "LIKELY SAFE" or (
                current and SEVERITY.get(current.get("verdict"), 0) >= SEVERITY[fv["verdict"]]):
            return
        fv.pop("escalated_from", None)
        fv["escalation"] = (fv.get("escalation", "").replace("Raised by safety rules", "Instant safety check")
                            + " Gemma's full analysis follows.")
        fv["source"] = "rules (instant)"
        self.state["verdict"] = fv

    def _check_llm(self):
        with self._llm_busy:
            transcript, voice = self.state["transcript"], self.state["voice_summary"]
            self._llm_text = transcript
            try:
                result = analyse(transcript, voice)
                if self.state["transcript"] != transcript:
                    # Gemma judged an older snapshot: hold it to the rules for the latest words,
                    # so it can't pull an instant SCAM flag back down.
                    said = result.get("escalated_from")
                    result = apply_floor(result, self.state["voice_summary"], find_cues(self.state["transcript"]))
                    if said:
                        result["escalated_from"] = said
                self.state["verdict"] = result
                self.state["error"] = None
            except Exception as e:
                self.state["error"] = f"Gemma: {type(e).__name__}: {e}"
            self._emit()

    def _emit(self):
        try:
            self._maybe_alert()
        except Exception as e:
            self.state["alert"] = f"Alert failed: {type(e).__name__}: {e}"
        self._notify()

    def _notify(self):
        try:
            self.on_update(self.state)
        except Exception:
            pass

    # --- loved-one alert (opt-in: set ALERT_NTFY_TOPIC) ------------------------------------
    def _maybe_alert(self):
        """Push a phone notification when the call looks dangerous. At most one per level
        (AI voice, high risk, confirmed scam), so the phone isn't spammed every few seconds."""
        if not alerts.enabled():
            return
        verdict, voice = self.state["verdict"], self.state["voice_summary"]
        reason = alerts.alert_reason(verdict, voice, len(self.state["voice"]))
        if not reason:
            return
        if verdict and verdict.get("verdict") == "SCAM":
            key = "scam"
        elif verdict and "risk" in reason:
            key = "risk"
        else:
            key = "voice"
        with self._alert_lock:
            if key in self._alerts_sent or "scam" in self._alerts_sent:
                return
            self._alerts_sent.add(key)
        message = alerts.build_message(reason, verdict)

        def deliver():
            try:
                alerts.send(message)
                self.state["alert"] = f"Alert sent to loved one: {reason}"
            except Exception as e:
                self.state["alert"] = f"Alert failed: {type(e).__name__}: {e}"
            self._notify()

        threading.Thread(target=deliver, daemon=True).start()


# --- terminal view ---------------------------------------------------------------------------
COLOUR = {"SCAM": "\033[1;41;97m", "SUSPICIOUS": "\033[1;43;30m", "LIKELY SAFE": "\033[1;42;30m"}
RESET = "\033[0m"


def level_line(state: dict) -> str:
    """Input level meter, or a hint when no sound is reaching the app at all."""
    started = state.get("started")
    if not state.get("chunks"):
        if started and time.time() - started > CHUNK_S + 2:
            return "No audio received yet: is the call/video actually playing through this device?"
        return ""
    lvl = state.get("level") or 0.0
    bars = min(10, int(lvl / 0.01))  # ~0.1 RMS = full scale
    note = "  (silent: nothing loud enough is reaching this input)" if lvl < SILENCE_RMS else ""
    return f"Input level [{'#' * bars}{'-' * (10 - bars)}] {lvl:.3f}{note}"


def render_text(state: dict) -> str:
    lines = ["SCAM CALL CHECKER  (Ctrl+C to stop)", state["status"], level_line(state), ""]
    if state.get("alert"):
        lines.insert(3, f">>> {state['alert']}")
    vs = state["voice_summary"]
    if vs:
        lines.append(f"Voice [{vs['detector']}]: {vs['fake_prob']:.0%} AI-generated (last 30 s), "
                     f"worst {vs['max_fake_prob']:.0%}")
    r = state["verdict"]
    if r:
        lines += ["", f"{COLOUR.get(r['verdict'], '')} {r['verdict']}  risk {r.get('risk_score', '?')}/100 {RESET}"]
        if r.get("escalation"):
            said = r.get("escalated_from")
            lines.append(f"  {r['escalation']}" + (f" (Gemma alone said {said})" if said else ""))
        if r.get("llm_error"):
            lines.append(f"  [Gemma unavailable, safety rules only: {r['llm_error']}]")
        lines += [f'  ! "{f.get("quote", "")}" - {f.get("why", "")}' for f in r.get("red_flags", [])]
        lines += [f"  What to do: {r.get('advice', '')}"]
    elif state["transcript"]:
        lines.append("\nWaiting for Gemma's first verdict...")
    if state["error"]:
        lines.append(f"\n[error] {state['error']}")
    tail = state["transcript"][-400:]
    lines += ["", "Transcript:", ("..." if len(state["transcript"]) > 400 else "") + tail]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Listen to a call live and flag scams.")
    ap.add_argument("--device", help="mic index, 'system' (default output) or 'system:<index>' "
                                     "(a specific speaker/headset, see --list-devices)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--detector", help="aasist | aasist-l | wav2vec2 (default: VOICE_DETECTOR or aasist)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # don't crash on odd characters in Windows consoles
    if os.name == "nt":
        os.system("")  # enable colour codes in Windows terminals

    if args.list_devices:
        print("Microphones (use --device <number>):")
        for i, name in input_devices():
            print(f"  {i:3d}  {name}")
        loops = loopback_devices()
        if loops:
            print("\nSystem audio: what a speaker/headset is playing (use --device system:<number>):")
            for i, name in loops:
                print(f"  system:{i:<4d} {name}")
        return

    device = parse_device(args.device)
    print("Loading models (first run downloads Whisper)...")
    detector = get_detector(args.detector)
    warm_up()

    def show(state):
        print("\033[2J\033[H" + render_text(state), flush=True)

    mon = LiveMonitor(device=device, detector=detector, on_update=show)
    mon.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        mon.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
