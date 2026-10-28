# Demo clips

Put short audio clips (10–30 s) here. They show up as one-click examples in the app.

Aim for the 2×2 grid, named like this:

| File | Voice | Content |
|---|---|---|
| `real_safe.wav` | teammate | legit script |
| `real_scam.wav` | teammate | scam script |
| `fake_safe.wav` | open-source TTS | legit script |
| `fake_scam.wav` | open-source TTS | scam script |

Record the real ones on a laptop or phone mic. Make the fake ones with any open-source
TTS (e.g. Kokoro or Piper) reading the same scripts.

Then check what AASIST thinks of each:

```
python -m scamcheck.voice samples/*.wav
```

If AASIST gets your clips wrong, tell the team early. The demo story still works:
the content check (Gemma) catches the scam either way.

## Legit script

> Hi, this is Sarah from Loughborough Dental Practice calling to remind you about your
> check-up on Thursday at 2pm. If you need to rearrange, just ring the number on your
> appointment card. Thanks, bye.

## Scam script

> Hello, this is the fraud team at your bank. We've seen suspicious payments on your
> account and it's been frozen. To protect your money we need to move it to a safe
> account right now. I'll need the six-digit code we've just texted you. Please don't
> hang up and don't tell anyone, this is a live investigation.
