"""AASIST (NAVER, MIT licence), using the pretrained weights bundled in models/weights/.

Trained on ASVspoof 2019 LA. Input: 64,600 samples (~4 s) at 16 kHz.
Output logits: class 1 = bonafide (real), class 0 = spoof (fake).
"""

import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from .base import Detector

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:  # so `models.AASIST` imports from any working directory
    sys.path.insert(0, str(REPO))


class AasistDetector(Detector):
    window = 64600

    def __init__(self, variant: str = "AASIST"):  # or "AASIST-L" (85k params, faster)
        self.name = variant
        cfg = json.loads((REPO / "config" / f"{variant}.conf").read_text())["model_config"]
        Model = import_module(f"models.{cfg['architecture']}").Model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Model(cfg).to(self.device)
        state = torch.load(REPO / "models" / "weights" / f"{variant}.pth", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def fake_probs(self, windows: np.ndarray) -> np.ndarray:
        x = torch.tensor(windows, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, logits = self.model(x)
        return torch.softmax(logits, dim=-1)[:, 0].cpu().numpy()  # class 0 = spoof
