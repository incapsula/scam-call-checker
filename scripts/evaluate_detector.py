"""Measure how well a fake-voice detector does on labelled clips.

The folder must contain real/ and fake/ subfolders (scripts/get_dataset.py makes these,
or drop your own clips in).

    python scripts/evaluate_detector.py data/deepfake-audio-detection
    python scripts/evaluate_detector.py data/deepfake-audio-detection --detector wav2vec2 --limit 50

Prints accuracy at a 0.5 threshold and the EER (equal error rate: lower is better),
and saves per-file scores to <folder>/scores_<detector>.csv.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere
from scamcheck.detectors import DETECTORS, get_detector  # noqa: E402

AUDIO_TYPES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def files_in(folder: Path, limit: int | None) -> list[Path]:
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in AUDIO_TYPES)
    return files[:limit] if limit else files


def equal_error_rate(real: np.ndarray, fake: np.ndarray) -> tuple[float, float]:
    """EER and the threshold where 'real flagged as fake' == 'fake missed'."""
    thresholds = np.unique(np.concatenate([real, fake]))
    far = np.array([np.mean(real >= t) for t in thresholds])  # real flagged as fake
    frr = np.array([np.mean(fake < t) for t in thresholds])   # fake missed
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thresholds[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--detector", choices=list(DETECTORS), default=None)
    ap.add_argument("--limit", type=int, default=None, help="max clips per class")
    args = ap.parse_args()

    root = Path(args.folder)
    real_files, fake_files = files_in(root / "real", args.limit), files_in(root / "fake", args.limit)
    if not real_files or not fake_files:
        raise SystemExit(f"Need audio in both {root / 'real'} and {root / 'fake'}")

    det = get_detector(args.detector)
    rows, start = [], time.time()
    for label, files in (("real", real_files), ("fake", fake_files)):
        for n, path in enumerate(files, 1):
            try:
                p = det.score_file(str(path))["fake_prob"]
            except Exception as e:
                print(f"  skip {path.name}: {e}")
                continue
            rows.append((str(path), label, p))
            print(f"  [{label} {n}/{len(files)}] {p:.2f}  {path.name}")

    real = np.array([p for _, l, p in rows if l == "real"])
    fake = np.array([p for _, l, p in rows if l == "fake"])
    acc = (np.sum(real < 0.5) + np.sum(fake >= 0.5)) / (len(real) + len(fake))
    eer, thr = equal_error_rate(real, fake)

    out = root / f"scores_{det.name}.csv"
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows([("file", "label", "fake_prob"), *rows])

    print(f"\n{det.name} on {root.name}: {len(real)} real, {len(fake)} fake, {time.time() - start:.0f}s")
    print(f"  accuracy @0.5:         {acc:.1%}")
    print(f"  real flagged as fake:  {np.mean(real >= 0.5):.1%}")
    print(f"  fakes caught:          {np.mean(fake >= 0.5):.1%}")
    print(f"  EER:                   {eer:.1%}  (best threshold {thr:.2f})")
    print(f"  per-file scores:       {out}")


if __name__ == "__main__":
    main()
