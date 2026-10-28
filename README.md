# Scam Call Checker

Upload or record a phone call and find out whether it's a scam, and whether the voice
is AI-generated. Everything runs locally on open models: nothing about the call leaves
the laptop.

Built at Hacktoberfest Hack Day Loughborough x DigiTech (28 Oct 2026), on top of
[AASIST](https://github.com/clovaai/aasist) and Google's Gemma 4.

## How it works

```
call audio ──► AASIST ──────────────► fake-voice score ──┐
          └──► Whisper ─► transcript ────────────────────┴──► Gemma 4 ──► verdict, quoted red flags, advice
```

1. **AASIST** (`scamcheck/voice.py`): pretrained anti-spoofing model scores each 4-second
   window of the call for synthetic speech.
2. **Whisper** (`scamcheck/transcribe.py`): turns the call into text.
3. **Gemma 4 via Ollama** (`scamcheck/analyse.py`): reads both, follows the instructions in
   `skills/scam-call-analyst/SKILL.md` (Agent Skills format), and returns structured JSON:
   verdict, risk score, the exact phrases that are red flags, and what to do.

## Setup

Python 3.10–3.12 recommended.

```bash
git clone <this repo> && cd <repo folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com), then pull Gemma 4:

```bash
ollama pull gemma4               # E4B, ~9.6 GB
# weaker laptop or slow wifi:
ollama pull gemma4:e2b           # ~7.2 GB, then set GEMMA_MODEL=gemma4:e2b
```

Run the app from the repo root and open the URL it prints:

```bash
python app.py
```

The first run also downloads the Whisper weights. AASIST's weights are already in
`models/weights/`.

**Event wifi tip:** only the demo laptop needs everything. If you're only working on
one piece, install just what it needs (see the table below).

## Test each piece on its own

```bash
python -m scamcheck.voice samples/*.wav          # AASIST scores per clip
python -m scamcheck.transcribe samples/real_scam.wav
python -m scamcheck.analyse "This is HMRC, you will be arrested unless you pay in gift cards"
```

## Who owns what

Each person owns different files, so you won't get merge conflicts.

| Task | Files | Needs installed | Owner |
|---|---|---|---|
| Voice detector: test AASIST on our clips, tune windows | `scamcheck/voice.py` | torch, librosa, soundfile | |
| Demo clips: record real voices, generate TTS fakes | `samples/` | a mic + any TTS | |
| Scam reasoning: tune the prompt, test edge cases | `skills/scam-call-analyst/SKILL.md` | Ollama + Gemma 4 | |
| UI and demo: polish `app.py`, pitch, backup recording | `app.py` | everything | |

## Git workflow (keep it simple)

```bash
git pull --rebase          # before you start, and before every push
git add <your files>
git commit -m "what you changed"
git push
```

Commit small and often. If a push is rejected, `git pull --rebase` and push again.

## Timeline

- **15:45** feature freeze, only fixes after this
- **16:00** record a backup screen capture of the demo
- **16:15** demos

## Known limitations

- AASIST was trained on ASVspoof 2019 LA (clean audio, 2019-era TTS and voice
  conversion). It can misjudge noisy phone or mic recordings and may miss modern voice
  cloning. That's why the verdict is driven mainly by what is said, with the voice score
  as an extra warning sign.
- No live phone integration: the app works on recorded or uploaded clips.

## Credits

- **AASIST**: Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
  Graph Attention Networks* (2021). Code and pretrained weights © NAVER Corp., MIT licence
  (see `LICENSE`, `NOTICE`). The original AASIST readme, including training instructions,
  is in [`AASIST_README.md`](AASIST_README.md).
- **Gemma 4**: Google, Apache 2.0.
- **Whisper** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper), MIT.
