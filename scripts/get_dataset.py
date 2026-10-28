"""Download a balanced subset of a Hugging Face real-vs-fake speech dataset.

Streams the dataset, so you never download the whole thing, and saves clips to
data/<dataset>/real/ and data/<dataset>/fake/.

    python scripts/get_dataset.py garystafford/deepfake-audio-detection --per-class 100
    python scripts/get_dataset.py mueller91/In-The-Wild --per-class 100

Then evaluate a detector on it:
    python scripts/evaluate_detector.py data/deepfake-audio-detection

If columns aren't detected automatically, pass --audio-col / --label-col, and
--fake-value to say which label value means "fake" (e.g. --fake-value 1 or --fake-value spoof).
Needs: pip install -r requirements-extra.txt
"""

import argparse
import shutil
from pathlib import Path

import soundfile as sf
from datasets import Audio, ClassLabel, load_dataset

REAL_WORDS = ("real", "bona", "human", "genuine", "authentic", "original")
FAKE_WORDS = ("fake", "spoof", "synth", "deepfake", "generated", "tts", "ai")
LABEL_COLS = ("label", "labels", "is_fake", "fake", "class", "target", "type")


def to_class(value, feature, fake_value):
    """Map a raw label to 'real' / 'fake' / None (unknown)."""
    if fake_value is not None:
        return "fake" if str(value).lower() == fake_value.lower() else "real"
    if isinstance(feature, ClassLabel):
        value = feature.int2str(int(value))
    if isinstance(value, bool):
        return "fake" if value else "real"
    if isinstance(value, (int, float)):
        return "fake" if int(value) == 1 else "real"  # assumption: 1 = fake (override with --fake-value)
    s = str(value).lower()
    if any(w in s for w in REAL_WORDS):
        return "real"
    if any(w in s for w in FAKE_WORDS):
        return "fake"
    return None


def save_audio(audio, out_base: Path) -> Path:
    """Write one example's audio, whatever form `datasets` handed it over in."""
    if isinstance(audio, dict) and audio.get("bytes"):
        ext = Path(audio.get("path") or "clip.wav").suffix or ".wav"
        path = out_base.with_suffix(ext)
        path.write_bytes(audio["bytes"])
        return path
    if isinstance(audio, dict) and audio.get("array") is not None:
        path = out_base.with_suffix(".wav")
        sf.write(path, audio["array"], audio["sampling_rate"])
        return path
    if isinstance(audio, dict) and audio.get("path") and Path(audio["path"]).exists():
        path = out_base.with_suffix(Path(audio["path"]).suffix or ".wav")
        shutil.copy(audio["path"], path)
        return path
    if hasattr(audio, "get_all_samples"):  # datasets>=4 AudioDecoder
        s = audio.get_all_samples()
        path = out_base.with_suffix(".wav")
        sf.write(path, s.data.numpy().mean(axis=0), s.sample_rate)
        return path
    raise ValueError(f"Don't know how to save audio of type {type(audio).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="Hugging Face dataset id")
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--audio-col")
    ap.add_argument("--label-col")
    ap.add_argument("--fake-value", help="label value that means fake")
    ap.add_argument("--max-scan", type=int, default=20000, help="give up after this many rows")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    ds = load_dataset(args.name, args.config, split=args.split, streaming=True)
    features = ds.features or {}

    audio_col = args.audio_col or next((c for c, f in features.items() if isinstance(f, Audio)), "audio")
    try:  # avoid decoding: we just want the original file bytes
        ds = ds.cast_column(audio_col, Audio(decode=False))
    except Exception:
        pass

    if args.label_col:
        label_col = args.label_col
    else:
        keys = list(features) or list(next(iter(ds)).keys())
        label_col = next((c for c in LABEL_COLS if c in keys), None)
        if label_col is None:
            raise SystemExit(f"Couldn't find a label column in {keys}. Pass --label-col.")
    label_feature = features.get(label_col)
    print(f"audio column: {audio_col!r}, label column: {label_col!r} ({label_feature})")

    out = Path(args.out) / args.name.split("/")[-1]
    for cls in ("real", "fake"):
        (out / cls).mkdir(parents=True, exist_ok=True)
    counts = {"real": 0, "fake": 0}
    seen_labels = {}

    for i, ex in enumerate(ds.shuffle(seed=0, buffer_size=500)):
        if i >= args.max_scan or min(counts.values()) >= args.per_class:
            break
        cls = to_class(ex[label_col], label_feature, args.fake_value)
        seen_labels.setdefault(str(ex[label_col]), cls)
        if cls is None or counts[cls] >= args.per_class:
            continue
        save_audio(ex[audio_col], out / cls / f"{cls}_{counts[cls]:04d}")
        counts[cls] += 1
        if sum(counts.values()) % 20 == 0:
            print(f"  saved {counts}")

    print(f"Done: {counts} in {out}")
    print(f"Label mapping used: {seen_labels}  (wrong? rerun with --fake-value)")


if __name__ == "__main__":
    main()
