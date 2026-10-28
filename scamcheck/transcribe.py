"""Speech-to-text with faster-whisper (open-source Whisper)."""

from functools import lru_cache

import numpy as np
from faster_whisper import WhisperModel

WHISPER_SIZE = "small.en"  # switch to "base.en" if it's too slow / live mode lags behind


@lru_cache(maxsize=1)
def _model() -> WhisperModel:
    return WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")


def warm_up():
    """Load (and on first run, download) the Whisper model before a live call starts."""
    _model()


def _run(audio, prompt: str | None = None) -> str:
    segments, _ = _model().transcribe(
        audio, language="en", vad_filter=True, initial_prompt=prompt or None
    )
    return " ".join(s.text.strip() for s in segments).strip()


def transcribe(path: str) -> str:
    """Transcribe an audio file."""
    return _run(path)


def transcribe_array(audio: np.ndarray, prompt: str | None = None) -> str:
    """Transcribe 16 kHz mono float32 audio. `prompt` = recent transcript, for continuity."""
    return _run(np.asarray(audio, dtype=np.float32), prompt)


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        print(f"--- {p}\n{transcribe(p)}")
