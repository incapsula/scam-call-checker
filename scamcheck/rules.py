"""Deterministic safety net that runs alongside Gemma.

1. find_cues(): spots well-known scam phrases and read-out phone numbers in the transcript.
   They are passed to Gemma as hints, and added as red flags if Gemma missed them.
2. apply_floor(): after Gemma answers, the verdict can only go UP, never down:
   - an AI-generated voice is never "LIKELY SAFE"
   - an AI voice plus any money / offer / pressure cue is a SCAM
   - several classic scam phrases are SUSPICIOUS or a SCAM even with a human voice
   It also replaces unsafe advice (e.g. "call the number they gave you").

Accents, grammar and nationality are deliberately NOT signals: they say nothing about scams.

Try it without Gemma:
    python -m scamcheck.rules "Call Card Member Services at 1-800-694-0048" --voice 0.97
"""

import re

AI_VOICE_HIGH = 0.85    # voice score treated as "very likely AI-generated"
AI_VOICE_MEDIUM = 0.60  # "possibly AI-generated"

SEVERITY = {"LIKELY SAFE": 0, "SUSPICIOUS": 1, "SCAM": 2}
MIN_RISK = {"LIKELY SAFE": 0, "SUSPICIOUS": 45, "SCAM": 85}

SAFE_ADVICE = {
    "SCAM": ("Hang up. Don't call back any number they gave you, press any keys, pay anything or "
             "share codes or details. If it claimed to be your bank or card company, call the number "
             "on the back of your card. In the UK you can dial 159 to reach your bank safely."),
    "SUSPICIOUS": ("Be careful: don't pay anything or share codes or personal details. Hang up and "
                   "contact the organisation on a number you already trust (the back of your card, "
                   "their official website, or 159 in the UK for your bank)."),
}

# (name, regex, why it's a red flag, strong?)
_CUES = [
    # --- unsolicited offers about money ---
    ("rate_reduction",
     r"\binterest rates? (reduction|reduced|lowered)\b|\b(lower|reduce|cut) (your )?(interest|rate|apr)\b"
     r"|\brate reduction\b",
     "Unsolicited interest-rate reduction calls are a well-known robocall scam.", True),
    ("card_services",
     r"\bcard ?(member|holder) services\b|\bcredit card services\b",
     "'Card Member Services' is a generic name robocall scammers use instead of a real bank.", True),
    ("loan_debt_offer",
     r"\b(debt relief|debt consolidation|consolidate your debts?|pre-?approved|guaranteed loan|loan offer)\b",
     "Unsolicited loan or debt-relief offers are a common scam.", True),
    ("prize",
     r"\byou('ve| have) (just )?won\b|\bclaim your (prize|reward|gift)\b|\blottery\b|\bsweepstakes?\b"
     r"|\bfree (holiday|cruise|vacation|gift card)\b",
     "Unexpected prizes are a classic scam hook.", True),
    ("investment",
     r"\b(investment opportunity|guaranteed returns?|double your money|risk[- ]free (investment|returns?))\b",
     "Guaranteed or risk-free returns are a hallmark of investment scams.", True),
    ("warranty",
     r"\b(extended|car|vehicle|auto) warranty\b",
     "Car-warranty robocalls are a well-known scam.", True),
    ("refund",
     r"\b(refund|rebate|overpaid|overpayment|compensation payment)\b",
     "Unexpected refunds are often used to get bank details.", False),
    # --- money movement and secrets ---
    ("safe_account",
     r"\bsafe account\b|\bsecure account\b|\bmove (all )?(your|the) (money|savings|funds)\b",
     "Banks never ask you to move money to a 'safe account'.", True),
    ("gift_cards",
     r"\b(gift|itunes|google play|steam|amazon) cards?\b|\bvouchers?\b",
     "Legitimate organisations never ask for payment in gift cards or vouchers.", True),
    ("crypto",
     r"\b(bitcoin|crypto(currency)?|usdt|tether|ethereum|crypto wallet)\b",
     "Crypto payments are a common scam route because they can't be reversed.", True),
    ("transfer",
     r"\b(wire|bank) transfer\b|\bwestern union\b|\bmoneygram\b|\bcourier (will|to) collect\b",
     "Pressure to send money by transfer or courier is a common scam tactic.", False),
    ("one_time_code",
     r"\b(one[- ]time|verification|security|authori[sz]ation|six[- ]digit|6[- ]digit) (pass)?code\b"
     r"|\bcode (we|that we)('ve| have)? (just )?(texted|sent)\b|\bread (me )?(out )?the code\b",
     "Never share one-time codes: banks never ask for them.", True),
    ("credentials",
     r"\b(your|the) (pin|pin number|password|card number|long card number|cvv|cvc|security number)\b",
     "Requests for PINs, passwords or card details are a major red flag.", True),
    ("remote_access",
     r"\b(anydesk|teamviewer|remote access|remote desktop)\b|\bdownload (this|an|the) app\b"
     r"|\byour (computer|laptop|device) (has|is) (a virus|infected|been hacked|compromised)\b",
     "Remote-access requests are a classic tech-support scam.", True),
    # --- impersonation and threats ---
    ("tax_office",
     r"\b(hmrc|irs|tax office|tax authority)\b",
     "Tax-office impersonation is a common scam.", False),
    ("fraud_team",
     r"\bfraud (team|department|prevention team)\b",
     "Scammers often pose as a bank's fraud team.", False),
    ("threats",
     r"\b(arrest(ed)?|arrest warrant|police will|court action|legal action|lawsuit|bailiffs?)\b",
     "Threats of arrest or legal action are used to panic people.", True),
    ("account_alarm",
     r"\baccount (has been|is|will be) (frozen|suspended|locked|compromised|closed)\b"
     r"|\bsuspicious (activity|payments?|transactions?)\b|\bunusual activity\b",
     "Claims your account is compromised are used to create panic.", False),
    ("verify_details",
     r"\b(verify|confirm) your (identity|details|account|information)\b",
     "Being asked to 'verify your details' on an unexpected call is a red flag.", False),
    # --- robocall script and pressure ---
    ("final_notice",
     r"\b(the )?only notice\b|\b(final|last) notice\b|\bfinal attempt\b|\blast chance\b",
     "'Final notice' language is pressure from a scripted robocall.", True),
    ("press_key",
     r"\bpress (1|one|2|two|5|five|9|nine)\b",
     "'Press 1' prompts are typical of illegal robocalls.", True),
    ("deadline",
     r"\blimited[- ]time\b|\bact now\b|\btoday only\b|\bonly valid\b|\bvalid (only )?for\b"
     r"|\boffer (ends|expires)\b|\bwithin (the next )?\d+ (hours|business days|working days|days)\b",
     "Artificial deadlines pressure you not to check.", False),
    ("urgency",
     r"\bimmediately\b|\burgent(ly)?\b|\bas soon as possible\b",
     "Urgency is used to stop you thinking it through.", False),
    ("isolation",
     r"\b(don['’]?t|do not) (tell|hang up|speak to|contact|go to)\b|\bstay on the line\b"
     r"|\bkeep this (confidential|secret|between us)\b",
     "Being told to keep secrets or stay on the line is an isolation tactic.", True),
]
_COMPILED = [(name, re.compile(rx, re.I), why, strong) for name, rx, why, strong in _CUES]
GENERIC = {"urgency"}  # common in normal speech: never enough on its own to call a scam
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
_PHONE_WHY = ("They read out a number to call back. Real banks tell you to use the number on your "
              "card, never one from the call.")


def _quote(text: str, start: int, end: int) -> str:
    """The matched words plus a little context, for showing as a red flag."""
    before = text[:start].split()[-4:]
    after = text[end:].split()[:6]
    return " ".join(before + [text[start:end]] + after).strip(" ,.;")


def find_cues(transcript: str) -> list[dict]:
    """Scam cues found in the transcript (one per type), strongest first."""
    cues = []
    for name, rx, why, strong in _COMPILED:
        m = rx.search(transcript)
        if m:
            cues.append({"name": name, "match": m.group(0), "quote": _quote(transcript, *m.span()),
                         "why": why, "strong": strong})
    for m in _PHONE.finditer(transcript):
        digits = re.sub(r"\D", "", m.group(0))
        if 9 <= len(digits) <= 15:
            cues.append({"name": "callback_number", "match": m.group(0), "digits": digits,
                         "quote": _quote(transcript, *m.span()), "why": _PHONE_WHY, "strong": False})
            break
    return sorted(cues, key=lambda c: not c["strong"])


def _higher(a: str, b: str) -> str:
    return a if SEVERITY[a] >= SEVERITY[b] else b


def apply_floor(result: dict, voice: dict | None, cues: list[dict]) -> dict:
    """Raise the verdict to what the safety rules require. Never lowers it."""
    out = dict(result)
    verdict = str(out.get("verdict", "")).strip().upper()
    out["verdict"] = verdict if verdict in SEVERITY else "SUSPICIOUS"
    out["red_flags"] = [f for f in (out.get("red_flags") or []) if isinstance(f, dict)]
    out.setdefault("voice_note", "")
    out.setdefault("advice", "")

    p = (voice or {}).get("fake_prob")
    strong = [c for c in cues if c["strong"]]
    specific = [c for c in cues if c["name"] not in GENERIC]
    floor, reasons = "LIKELY SAFE", []

    if p is not None and p >= AI_VOICE_HIGH:
        floor = "SUSPICIOUS"
        reasons.append(f"the voice is {p:.0%} likely AI-generated")
        if specific:
            floor = "SCAM"
            reasons.append("an AI voice is pushing money, an offer, pressure or a callback number")
    elif p is not None and p >= AI_VOICE_MEDIUM and strong:
        floor = "SCAM"
        reasons.append(f"a possibly AI-generated voice ({p:.0%}) uses classic scam lines")
    if len(strong) >= 3:
        floor = "SCAM"
        reasons.append(f"{len(strong)} classic scam lines in one call")
    elif len(strong) >= 2:
        floor = _higher(floor, "SUSPICIOUS")
        reasons.append(f"{len(strong)} classic scam lines")

    if SEVERITY[floor] > SEVERITY[out["verdict"]]:
        out["escalated_from"] = out["verdict"]
        out["verdict"] = floor
        out["escalation"] = "Raised by safety rules: " + "; ".join(reasons) + "."
        out["advice"] = SAFE_ADVICE[floor]

    # Add rule-detected red flags Gemma didn't already quote (not on a "safe" verdict).
    if out["verdict"] != "LIKELY SAFE":
        quoted = " ".join(str(f.get("quote", "")).lower() for f in out["red_flags"])
        for c in cues:
            if c["match"].lower() not in quoted:
                out["red_flags"].append({"quote": c["quote"], "why": c["why"]})
    out["red_flags"] = out["red_flags"][:8]

    # Never tell someone to ring a number the caller read out.
    advice_digits = re.sub(r"\D", "", out["advice"])
    for c in cues:
        if c["name"] == "callback_number" and c["digits"][:7] in advice_digits:
            out["advice"] = SAFE_ADVICE.get(out["verdict"], SAFE_ADVICE["SUSPICIOUS"])

    try:
        risk = int(out.get("risk_score") or 0)
    except (TypeError, ValueError):
        risk = 0
    out["risk_score"] = max(0, min(100, max(risk, MIN_RISK[out["verdict"]])))
    return out


def rules_only_verdict(voice: dict | None, cues: list[dict]) -> dict:
    """Fallback when Gemma isn't available: judge on the rules alone."""
    p = (voice or {}).get("fake_prob")
    voice_note = (f"Voice scored {p:.0%} likely AI-generated." if p is not None
                  else "Voice detector unavailable.")
    base = {
        "verdict": "SUSPICIOUS" if any(c["strong"] for c in cues) else "LIKELY SAFE",
        "risk_score": 0,
        "red_flags": [],
        "voice_note": voice_note,
        "advice": "No obvious scam signs found, but never share codes or pay anyone who calls you unexpectedly.",
    }
    out = apply_floor(base, voice, cues)
    if out["verdict"] != "LIKELY SAFE" and "escalation" not in out:
        out["advice"] = SAFE_ADVICE[out["verdict"]]
    out["source"] = "rules"
    return out


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--voice", type=float, default=None, help="AI-voice score 0-1")
    args = ap.parse_args()
    v = None if args.voice is None else {"fake_prob": args.voice, "max_fake_prob": args.voice}
    found = find_cues(args.text)
    print("Cues:", json.dumps([{k: c[k] for k in ("name", "quote", "strong")} for c in found], indent=2))
    print("Rules-only verdict:", json.dumps(rules_only_verdict(v, found), indent=2))
