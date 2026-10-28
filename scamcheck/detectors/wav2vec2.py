"""Any Hugging Face wav2vec2-style audio classifier as a detector (needs `transformers`).

Default: garystafford/wav2vec2-deepfake-voice-detector, fine-tuned on real speech vs
modern TTS (ElevenLabs, Amazon Polly, Kokoro, Hume AI, Speechify, Luvvoice).
Use another model with W2V_MODEL=<hub id>.
"""

import os

import numpy as np
import torch

from .base import SR, Detector

DEFAULT_MODEL = "garystafford/wav2vec2-deepfake-voice-detector"
FAKE_WORDS = ("fake", "spoof", "synth", "deepfake", "generated", "ai")


class Wav2Vec2Detector(Detector):
    window = SR * 5  # 5 s windows (trained on 2.5-13 s chunks)
    batch = 4

    def __init__(self, model_id: str | None = None):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        model_id = model_id or os.getenv("W2V_MODEL", DEFAULT_MODEL)
        self.name = model_id.split("/")[-1]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fx = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id).to(self.device).eval()
        labels = {i: str(l).lower() for i, l in self.model.config.id2label.items()}
        self.fake_idx = next((i for i, l in labels.items() if any(w in l for w in FAKE_WORDS)), 1)

    def fake_probs(self, windows: np.ndarray) -> np.ndarray:
        inputs = self.fx(list(windows), sampling_rate=SR, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=-1)[:, self.fake_idx].cpu().numpy()
