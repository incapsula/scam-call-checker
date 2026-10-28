"""Shared logic for every fake-voice detector: loading audio, windowing, batching."""

import librosa
import numpy as np

SR = 16000  # every detector works on 16 kHz mono float32 audio


class Detector:
    name = "base"
    window = 64600  # samples per window the model expects (~4 s at 16 kHz)
    batch = 8       # windows per forward pass

    def fake_probs(self, windows: np.ndarray) -> np.ndarray:
        """windows: (n, self.window) float32 at 16 kHz -> (n,) probability each is fake."""
        raise NotImplementedError

    def _windows(self, audio: np.ndarray) -> list[np.ndarray]:
        """4 s windows with 50% overlap; short clips are tiled to length (as AASIST does)."""
        win, hop = self.window, self.window // 2
        if len(audio) < win:
            return [np.tile(audio, win // len(audio) + 1)[:win]]
        wins = [audio[s:s + win] for s in range(0, len(audio) - win + 1, hop)]
        if (len(audio) - win) % hop:  # make sure the tail is covered
            wins.append(audio[-win:])
        return wins

    def score(self, audio: np.ndarray) -> dict:
        """audio: 16 kHz mono float32. Returns average, worst window and per-window scores."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            raise ValueError("Empty audio")
        wins = self._windows(audio)
        probs = []
        for i in range(0, len(wins), self.batch):
            probs.extend(float(p) for p in self.fake_probs(np.stack(wins[i:i + self.batch])))
        return {
            "detector": self.name,
            "fake_prob": float(np.mean(probs)),
            "max_fake_prob": float(np.max(probs)),
            "windows": [round(p, 3) for p in probs],
        }

    def score_file(self, path: str) -> dict:
        audio, _ = librosa.load(path, sr=SR, mono=True)
        return self.score(audio)
