"""Fake-voice score from the pretrained AASIST model bundled in this repo.

AASIST (NAVER, MIT licence) was trained on ASVspoof 2019 LA.
- Input: 16 kHz mono waveform, 64,600 samples (~4 s) per window.
- Output logits: class 1 = bonafide (real), class 0 = spoof (fake).

Check it on your own clips before relying on it:
    python -m scamcheck.voice samples/*.wav
"""

import json
import sys
from functools import lru_cache
from importlib import import_module
from pathlib import Path

import librosa
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:  # so `models.AASIST` imports from any working directory
    sys.path.insert(0, str(REPO))

SR = 16000
WIN = 64600        # AASIST's training input length (~4 s)
HOP = WIN // 2     # 50% overlap between windows
BATCH = 8          # windows per forward pass (keeps CPU memory low)
VARIANT = "AASIST"  # or "AASIST-L" (85k params, faster, slightly less accurate)


@lru_cache(maxsize=1)
def _load():
    cfg = json.loads((REPO / "config" / f"{VARIANT}.conf").read_text())
    model_cfg = cfg["model_config"]
    Model = import_module(f"models.{model_cfg['architecture']}").Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(model_cfg).to(device)
    state = torch.load(REPO / "models" / "weights" / f"{VARIANT}.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device


def _windows(audio: np.ndarray) -> list[np.ndarray]:
    """Cut audio into 4 s windows; short clips are tiled to length, as AASIST does in eval."""
    if len(audio) < WIN:
        reps = WIN // len(audio) + 1
        return [np.tile(audio, reps)[:WIN]]
    wins = [audio[s:s + WIN] for s in range(0, len(audio) - WIN + 1, HOP)]
    if (len(audio) - WIN) % HOP:  # make sure the tail is covered
        wins.append(audio[-WIN:])
    return wins


def fake_voice_score(path: str) -> dict:
    """Return {"fake_prob": mean over windows, "max_fake_prob": worst window, "windows": [...]}."""
    audio, _ = librosa.load(path, sr=SR, mono=True)
    if audio.size == 0:
        raise ValueError(f"No audio in {path}")

    model, device = _load()
    wins = _windows(audio)
    probs = []
    with torch.no_grad():
        for i in range(0, len(wins), BATCH):
            x = torch.tensor(np.stack(wins[i:i + BATCH]), dtype=torch.float32, device=device)
            _, logits = model(x)
            probs.extend(torch.softmax(logits, dim=-1)[:, 0].cpu().tolist())  # class 0 = spoof

    return {
        "fake_prob": float(np.mean(probs)),
        "max_fake_prob": float(np.max(probs)),
        "windows": [round(p, 3) for p in probs],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m scamcheck.voice clip1.wav [clip2.wav ...]")
    for p in sys.argv[1:]:
        r = fake_voice_score(p)
        print(f"{r['fake_prob']:.2f} fake (worst window {r['max_fake_prob']:.2f})  {p}")
