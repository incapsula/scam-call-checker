"""Push a notification to a loved one's phone when a call looks dangerous.

Uses ntfy (https://ntfy.sh): free, no account, no API keys.
On the loved one's phone:
  1. Install the "ntfy" app (Android / iPhone).
  2. Tap + and subscribe to a topic name only you know, e.g. scamcheck-velox-4821
     (anyone who knows the name can read it, so make it hard to guess).
On this laptop, before starting the app:
  $env:ALERT_NTFY_TOPIC="scamcheck-velox-4821"        (PowerShell)
Test it:
  python -m scamcheck.alerts

Alerts fire once per call when the voice is >= ALERT_AI_VOICE (default 80%) AI-generated,
the verdict is SCAM, or the risk score is >= ALERT_RISK (default 80).
"""

import os
import urllib.request

AI_ALERT = float(os.getenv("ALERT_AI_VOICE", "0.8"))
RISK_ALERT = int(os.getenv("ALERT_RISK", "80"))


def enabled() -> bool:
    return bool(os.getenv("ALERT_NTFY_TOPIC", "").strip())


def alert_reason(verdict: dict | None, voice: dict | None, voice_checks: int = 99) -> str | None:
    """Why this call deserves an alert, or None. Voice needs 2+ checks so one blip can't trigger it."""
    reasons = []
    if voice and voice_checks >= 2 and voice.get("fake_prob", 0) >= AI_ALERT:
        reasons.append(f"AI-generated voice ({voice['fake_prob']:.0%})")
    if verdict:
        try:
            risk = int(verdict.get("risk_score") or 0)
        except (TypeError, ValueError):
            risk = 0
        if verdict.get("verdict") == "SCAM":
            reasons.append(f"scam detected (risk {risk}/100)")
        elif risk >= RISK_ALERT:
            reasons.append(f"risk {risk}/100")
    return ", ".join(reasons) or None


def build_message(reason: str, verdict: dict | None) -> str:
    lines = [f"Possible scam call happening now: {reason}."]
    if verdict and verdict.get("red_flags"):
        quotes = [f'"{str(f.get("quote", ""))[:70]}"' for f in verdict["red_flags"][:2]]
        lines.append("Red flags: " + "; ".join(quotes))
    lines.append("Give them a ring to check they're OK, and remind them not to pay or share codes.")
    return "\n".join(lines)


def send(message: str, title: str = "Scam Call Checker alert") -> None:
    topic = os.getenv("ALERT_NTFY_TOPIC", "").strip()
    if not topic:
        raise RuntimeError("Set ALERT_NTFY_TOPIC to enable alerts")
    server = os.getenv("ALERT_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    req = urllib.request.Request(
        f"{server}/{topic}",
        data=message.encode("utf-8"),
        method="POST",
        headers={"Title": title, "Priority": "urgent", "Tags": "rotating_light,telephone_receiver"},
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"ntfy returned HTTP {resp.status}")


if __name__ == "__main__":
    if not enabled():
        raise SystemExit('Set the topic first, e.g.  $env:ALERT_NTFY_TOPIC="scamcheck-yourname-4821"')
    send("Test from Scam Call Checker: alerts are working.", title="Scam Call Checker test")
    print(f"Sent a test alert to topic {os.getenv('ALERT_NTFY_TOPIC')!r}. Check the phone.")
