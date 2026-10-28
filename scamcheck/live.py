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
from .transcribe import transcribe_array, warm_up

SR = 16000
CHUNK_S = 5          # process audio in 5 s chunks
LLM_EVERY_S = 15     # ask Gemma at most this often
SILENCE_RMS = 0.003  # chunks quieter than this are skipped
RECENT = 6           # voice score = average of the last N chunks (~30 s)


SYSTEM_AUDIO = "system"  # special device: whatever this computer is playing (Windows)


def input_devices() -> list[tuple[int, str]]:
    return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]


class SystemAudioCapture:
    """Records what Windows is playing (Discord, YouTube, Phone Link calls) via WASAPI loopback.

    Needs: pip install PyAudioWPatch   (Windows only)
    Captures the *default output device*, so set the call app's output to "Default".
    Note: Windows sends no audio here while nothing is playing.
    """

    def __init__(self, push):
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as e:
            raise RuntimeError("System audio needs: pip install PyAudioWPatch (Windows only)") from e

        self.p = pyaudio.PyAudio()
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
        self._stream = None
        self.state = self._fresh_state()

    @staticmethod
    def _fresh_state() -> dict:
        return {"status": "idle", "transcript": "", "voice": [], "voice_summary": None,
                "verdict": None, "error": None, "started": None}

    # --- audio capture -------------------------------------------------------------------
    def start(self):
        self.state = self._fresh_state()
        self._stop.clear()
        if self.device == SYSTEM_AUDIO:
            self._stream = SystemAudioCapture(self._q.put)
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
        need = CHUNK_S * self.rate
        last_llm = 0.0
        while not self._stop.is_set():
            try:
                buf = np.concatenate([buf, self._q.get(timeout=0.5)])
            except queue.Empty:
                continue
            if len(buf) < need:
                continue
            chunk, buf = buf[:need], buf[need:]
            audio = chunk if self.rate == SR else librosa.resample(chunk, orig_sr=self.rate, target_sr=SR)
            if float(np.sqrt(np.mean(audio ** 2))) < SILENCE_RMS:
                continue  # silence: nothing to score or transcribe

            # Voice and transcript are independent: one failing shouldn't stop the other.
            try:
                v = self.detector.score(audio)
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
                text = transcribe_array(audio, prompt=self.state["transcript"][-200:])
                if text:
                    self.state["transcript"] = f"{self.state['transcript']} {text}".strip()
            except Exception as e:
                self.state["error"] = f"Whisper: {type(e).__name__}: {e}"
            self._emit()

            if (self.state["transcript"] and time.time() - last_llm >= LLM_EVERY_S
                    and not self._llm_busy.locked()):
                last_llm = time.time()
                threading.Thread(target=self._check_llm, daemon=True).start()

    def _check_llm(self):
        with self._llm_busy:
            try:
                self.state["verdict"] = analyse(self.state["transcript"], self.state["voice_summary"])
                self.state["error"] = None
            except Exception as e:
                self.state["error"] = f"Gemma: {type(e).__name__}: {e}"
            self._emit()

    def _emit(self):
        try:
            self.on_update(self.state)
        except Exception:
            pass


# --- terminal view ---------------------------------------------------------------------------
COLOUR = {"SCAM": "\033[1;41;97m", "SUSPICIOUS": "\033[1;43;30m", "LIKELY SAFE": "\033[1;42;30m"}
RESET = "\033[0m"


def render_text(state: dict) -> str:
    lines = ["SCAM CALL CHECKER  (Ctrl+C to stop)", state["status"], ""]
    vs = state["voice_summary"]
    if vs:
        lines.append(f"Voice [{vs['detector']}]: {vs['fake_prob']:.0%} AI-generated (last 30 s), "
                     f"worst {vs['max_fake_prob']:.0%}")
    r = state["verdict"]
    if r:
        lines += ["", f"{COLOUR.get(r['verdict'], '')} {r['verdict']}  risk {r['risk_score']}/100 {RESET}"]
        lines += [f'  ! "{f["quote"]}" - {f["why"]}' for f in r["red_flags"]]
        lines += [f"  What to do: {r['advice']}"]
    elif state["transcript"]:
        lines.append("\nWaiting for Gemma's first verdict...")
    if state["error"]:
        lines.append(f"\n[error] {state['error']}")
    tail = state["transcript"][-400:]
    lines += ["", "Transcript:", ("..." if len(state["transcript"]) > 400 else "") + tail]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Listen to a call live and flag scams.")
    ap.add_argument("--device", help="input device index or name (default: system default mic)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--detector", help="aasist | aasist-l | wav2vec2 (default: VOICE_DETECTOR or aasist)")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")  # don't crash on odd characters in Windows consoles
    if os.name == "nt":
        os.system("")  # enable colour codes in Windows terminals

    if args.list_devices:
        for i, name in input_devices():
            print(f"{i:3d}  {name}")
        return

    device = int(args.device) if args.device and args.device.isdigit() else args.device
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
