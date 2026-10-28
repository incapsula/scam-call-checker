"""Speech-to-text with faster-whisper (open-source Whisper)."""

import os
from functools import lru_cache

import numpy as np
from faster_whisper import WhisperModel

# "small.en" is more accurate; "base.en" is ~2x faster if live mode lags:  $env:WHISPER_SIZE="base.en"
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "small.en")


@lru_cache(maxsize=1)
def _model() -> WhisperModel:
    return WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")


def warm_up():
    """Load (and on first run, download) the Whisper model before a live call starts."""
    _model()


def _run(audio, prompt: str | None = None, fast: bool = False) -> str:
    segments, _ = _model().transcribe(
        audio,
        language="en",
        vad_filter=True,
        initial_prompt=prompt or None,
        beam_size=1 if fast else 5,     # greedy decoding for live chunks: much faster
        without_timestamps=fast,
        condition_on_previous_text=not fast,
    )
    return " ".join(s.text.strip() for s in segments).strip()


def transcribe(path: str) -> str:
    """Transcribe an audio file (accuracy over speed)."""
    return _run(path)


def transcribe_array(audio: np.ndarray, prompt: str | None = None, fast: bool = False) -> str:
    """Transcribe 16 kHz mono float32 audio. `prompt` = recent transcript, for continuity."""
    return _run(np.asarray(audio, dtype=np.float32), prompt, fast)


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        print(f"--- {p}\n{transcribe(p)}")
