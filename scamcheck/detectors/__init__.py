"""Pluggable fake-voice detectors.

Pick one with the VOICE_DETECTOR environment variable (default: aasist):
    VOICE_DETECTOR=wav2vec2 python app.py

To plug in a better model:
1. Add a file here with a subclass of `Detector` (see base.py) that sets `name`,
   `window` and implements `fake_probs(windows) -> probabilities`.
2. Register it in DETECTORS below.
The app, live mode and evaluation script all pick it up automatically.
"""

import os
from functools import lru_cache

from .base import Detector


def _aasist():
    from .aasist import AasistDetector
    return AasistDetector()


def _aasist_l():
    from .aasist import AasistDetector
    return AasistDetector(variant="AASIST-L")


def _wav2vec2():
    from .wav2vec2 import Wav2Vec2Detector
    return Wav2Vec2Detector()


DETECTORS = {
    "aasist": _aasist,
    "aasist-l": _aasist_l,
    "wav2vec2": _wav2vec2,
}


@lru_cache(maxsize=None)
def get_detector(name: str | None = None) -> Detector:
    name = (name or os.getenv("VOICE_DETECTOR", "aasist")).lower()
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector {name!r}. Options: {', '.join(DETECTORS)}")
    return DETECTORS[name]()
