"""Speech-to-text with faster-whisper (open-source Whisper)."""

from functools import lru_cache

from faster_whisper import WhisperModel

WHISPER_SIZE = "small.en"  # switch to "base.en" if it's too slow on your laptop


@lru_cache(maxsize=1)
def _model() -> WhisperModel:
    return WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")


def transcribe(path: str) -> str:
    segments, _ = _model().transcribe(path, language="en")
    return " ".join(s.text.strip() for s in segments)


if __name__ == "__main__":
    import sys

    for p in sys.argv[1:]:
        print(f"--- {p}\n{transcribe(p)}")
