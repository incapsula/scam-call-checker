# Scam Call Checker

Listens to a phone call and warns you, live, if it's a scam and whether the voice is
AI-generated. Everything runs locally on open models: nothing about the call leaves
the laptop.

Built at Hacktoberfest Hack Day Loughborough x DigiTech (28 Oct 2026), on top of
[AASIST](https://github.com/clovaai/aasist) and Google's Gemma 4.

## How it works

```
call audio ──► voice detector (AASIST) ───► fake-voice score ───┐
(mic or        every 5 s                                        ├──► Gemma 4 ──► verdict, red flags, advice
 loopback) ──► Whisper ──► running transcript ──────────────────┘    every 15 s
```

1. **Listen** (`scamcheck/live.py`): captures audio from a mic or loopback device in 5 s chunks.
2. **Voice check** (`scamcheck/detectors/`): AASIST scores each 4 s window for synthetic
   speech. Detectors are pluggable: swap in a better model without touching anything else.
3. **Transcribe** (`scamcheck/transcribe.py`): Whisper turns each chunk into text.
4. **Judge** (`scamcheck/analyse.py`): Gemma 4 via Ollama reads the transcript + voice score,
   following `skills/scam-call-analyst/SKILL.md` (Agent Skills format), and returns JSON:
   verdict, risk score, the exact phrases that are red flags, and what to do.
5. **Safety rules** (`scamcheck/rules.py`): a deterministic check Gemma can't override. It spots
   known scam lines (rate-reduction offers, "Card Member Services", safe accounts, one-time codes,
   gift cards, "press 1", read-out callback numbers…) and only ever raises the verdict:
   an AI voice is never "safe", and an AI voice plus any offer, money or pressure is a SCAM.
   It also replaces unsafe advice, and gives a verdict on its own if Gemma is unavailable.
   Accents, grammar and nationality are deliberately not used as signals.

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
ollama pull gemma4:e2b           # smaller (~7.2 GB); then set GEMMA_MODEL=gemma4:e2b
```

**Event wifi tip:** only the demo laptop needs everything. Install just what your piece needs.

## Run it

```bash
python app.py                    # web UI: "Live call" tab + "Check a recording" tab
python -m scamcheck.live         # same live check in the terminal
```

## Plugging it into a call

| Setup | How | Notes |
|---|---|---|
| **Phone on speaker** (easiest, best for the demo) | Put the phone next to the laptop, use the default mic | Works with any phone and any call app |
| **Call on the laptop** (WhatsApp desktop, Teams, Zoom, Discord) | Pick a loopback device that carries the laptop's sound output | Captures the caller's side, which is the side we're checking |
| Real phone lines | Twilio Media Streams / SIP streaming audio to this code | Future work: needs a public server |

Loopback devices:
- **Windows:** Sound settings → More sound settings → Recording → right-click → *Show disabled
  devices* → enable **Stereo Mix**. If it's not there, install VB-Cable.
- **macOS:** install BlackHole (2ch). In Audio MIDI Setup create a *Multi-Output Device*
  (speakers + BlackHole) and set it as output so you still hear the call, then listen on BlackHole.
- **Linux:** pick the **"Monitor of …"** device.

Find device numbers with `python -m scamcheck.live --list-devices`, then
`python -m scamcheck.live --device <number>` (or pick it in the app's dropdown).

## Alert a loved one

When a call is flagged (AI voice ≥ 80%, risk ≥ 80, or SCAM), the app can push an urgent
notification to a family member's phone via [ntfy](https://ntfy.sh): free, no account.

1. On their phone: install the **ntfy** app, tap **+**, subscribe to a hard-to-guess topic,
   e.g. `scamcheck-velox-4821` (anyone who knows the name can read it).
2. On the laptop, before starting the app:
   ```powershell
   $env:ALERT_NTFY_TOPIC="scamcheck-velox-4821"
   python -m scamcheck.alerts      # sends a test notification
   python app.py
   ```

Alerts are off unless `ALERT_NTFY_TOPIC` is set. Thresholds: `ALERT_AI_VOICE` (0.8), `ALERT_RISK` (80).

## Swapping the voice detector

```bash
VOICE_DETECTOR=aasist     python app.py   # default
VOICE_DETECTOR=aasist-l   python app.py   # smaller, faster AASIST
VOICE_DETECTOR=wav2vec2   python app.py   # wav2vec2 model trained on modern TTS (needs requirements-extra.txt)
```

(Windows PowerShell: `$env:VOICE_DETECTOR="wav2vec2"; python app.py`)

To add a new model: create `scamcheck/detectors/<name>.py` with a `Detector` subclass that
implements `fake_probs(windows)`, then register it in `scamcheck/detectors/__init__.py`.

## Datasets

| Dataset | What's in it | Use it for |
|---|---|---|
| [garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection) | 1,866 clips, balanced real/fake. Fakes from 6 modern TTS (ElevenLabs, Polly, Kokoro…). 16 kHz FLAC | **Start here.** Small, modern fakes |
| [In-the-Wild](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) ([HF](https://huggingface.co/datasets/mueller91/In-The-Wild)) | ~38 h of real and deepfake audio of 58 public figures, Apache 2.0 | Real-world, noisier audio |
| ASVspoof 2019 LA | What AASIST was trained on; several GB (`python download_dataset.py`) | Skip at the event |
| [BothBosu/Scammer-Conversation](https://huggingface.co/datasets/BothBosu/Scammer-Conversation) | ~1k generated conversations (CSV) | Testing Gemma's scam reasoning on text |

Get a small balanced subset and measure a detector on it:

```bash
pip install -r requirements-extra.txt
python scripts/get_dataset.py garystafford/deepfake-audio-detection --per-class 100
python scripts/evaluate_detector.py data/deepfake-audio-detection
python scripts/evaluate_detector.py data/deepfake-audio-detection --detector wav2vec2
```

The evaluation prints accuracy and EER (lower is better): use those numbers in the pitch.

## Test each piece on its own

```bash
python -m scamcheck.voice samples/*.wav                    # voice scores per clip
python -m scamcheck.transcribe samples/real_scam.wav
python -m scamcheck.analyse "This is HMRC, you will be arrested unless you pay in gift cards"
python -m scamcheck.analyse --voice 0.97 "Call Card Member Services at 1-800-694-0048 to lower your interest rate"
python -m scamcheck.rules --voice 0.97 "any transcript"     # safety rules only, no Gemma needed
```

## Who owns what

Each person owns different files, so you won't get merge conflicts.

| Task | Files | Owner |
|---|---|---|
| Live listening: get mic / loopback capture working on the demo laptop | `scamcheck/live.py` | |
| Voice detector + dataset: evaluate AASIST vs wav2vec2, get numbers for the pitch | `scamcheck/detectors/`, `scripts/` | |
| Scam reasoning: tune the prompt, test with the text dataset | `skills/scam-call-analyst/SKILL.md` | |
| Demo clips, UI polish, pitch, backup recording | `samples/`, `app.py` | |

## Git workflow (keep it simple)

```bash
git pull --rebase          # before you start, and before every push
git add <your files>
git commit -m "what you changed"
git push
```

## Timeline

- **15:45** feature freeze, only fixes after this
- **16:00** record a backup screen capture of the demo
- **16:15** demos

## Known limitations

- AASIST was trained on ASVspoof 2019 LA (clean audio, 2019-era TTS and voice conversion). It
  can misjudge noisy phone or mic audio and may miss modern voice cloning. That's why the
  verdict is driven mainly by what is said, with the voice score as an extra warning sign.
- Live mode processes 3 s chunks: the instant safety-rule verdict appears ~3–5 s after a scam
  line is spoken, and Gemma's fuller verdict a couple of seconds later. If it falls behind on a
  slower laptop, use the smaller Whisper (`$env:WHISPER_SIZE="base.en"`) or longer chunks
  (`$env:LIVE_CHUNK_S="5"`).

## Credits

- **AASIST**: Jung et al., *AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
  Graph Attention Networks* (2021). Code and pretrained weights © NAVER Corp., MIT licence
  (see `LICENSE`, `NOTICE`). Original readme with training instructions: [`AASIST_README.md`](AASIST_README.md).
- **Gemma 4**: Google, Apache 2.0.
- **Whisper** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper), MIT.
