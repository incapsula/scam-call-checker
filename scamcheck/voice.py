"""Fake-voice scoring for a whole file, using whichever detector is selected.

Check detectors on your own clips:
    python -m scamcheck.voice samples/*.wav
    python -m scamcheck.voice --detector wav2vec2 samples/*.wav
"""

import argparse

from .detectors import DETECTORS, get_detector


def fake_voice_score(path: str, detector: str | None = None) -> dict:
    return get_detector(detector).score_file(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--detector", choices=list(DETECTORS), default=None)
    args = ap.parse_args()
    for p in args.files:
        r = fake_voice_score(p, args.detector)
        print(f"{r['fake_prob']:.2f} fake (worst window {r['max_fake_prob']:.2f}) [{r['detector']}]  {p}")
