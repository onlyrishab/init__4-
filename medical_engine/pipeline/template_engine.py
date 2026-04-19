"""
SignSetu — Medical Sentence Grammar Engine

Converts an ordered list of sign-language tokens into a natural English
sentence suitable for doctor–patient communication.

The engine uses a token knowledge-base (category, display form, grammar role)
and composable grammar rules so that *any* combination of tokens produces
a coherent sentence — not just a handful of hardcoded patterns.
"""

# ─── TOKEN KNOWLEDGE BASE ──────────────────────────────────────────────────────
# Each token knows its category and the word(s) to use in a sentence.

TOKENS = {
    # ── Symptoms ──
    "PAIN":       {"cat": "symptom",  "word": "pain"},
    "FEVER":      {"cat": "symptom",  "word": "fever"},
    "NAUSEA":     {"cat": "symptom",  "word": "nausea"},
    "COUGH":      {"cat": "symptom",  "word": "a cough"},
    "DIZZY":      {"cat": "symptom",  "word": "dizziness"},
    "VOMIT":      {"cat": "symptom",  "word": "vomiting"},
    "BLEEDING":   {"cat": "symptom",  "word": "bleeding"},
    "SWELLING":   {"cat": "symptom",  "word": "swelling"},
    "BURNING":    {"cat": "symptom",  "word": "a burning sensation"},
    "ITCHING":    {"cat": "symptom",  "word": "itching"},
    "TIRED":      {"cat": "symptom",  "word": "tiredness"},
    "BREATHLESS": {"cat": "symptom",  "word": "difficulty breathing"},
    "WEAKNESS":   {"cat": "symptom",  "word": "weakness"},
    "NUMBNESS":   {"cat": "symptom",  "word": "numbness"},
    "RASH":       {"cat": "symptom",  "word": "a rash"},
    "CRAMP":      {"cat": "symptom",  "word": "cramps"},
    "STIFF":      {"cat": "symptom",  "word": "stiffness"},
    "SORE":       {"cat": "symptom",  "word": "soreness"},

    # ── Body locations ──
    "HEAD":       {"cat": "location", "word": "head"},
    "CHEST":      {"cat": "location", "word": "chest"},
    "STOMACH":    {"cat": "location", "word": "stomach"},
    "BACK":       {"cat": "location", "word": "back"},
    "THROAT":     {"cat": "location", "word": "throat"},
    "NECK":       {"cat": "location", "word": "neck"},
    "SHOULDER":   {"cat": "location", "word": "shoulder"},
    "ARM":        {"cat": "location", "word": "arm"},
    "HAND":       {"cat": "location", "word": "hand"},
    "LEG":        {"cat": "location", "word": "leg"},
    "KNEE":       {"cat": "location", "word": "knee"},
    "FOOT":       {"cat": "location", "word": "foot"},
    "EYE":        {"cat": "location", "word": "eye"},
    "EAR":        {"cat": "location", "word": "ear"},
    "TEETH":      {"cat": "location", "word": "teeth"},
    "MOUTH":      {"cat": "location", "word": "mouth"},
    "NOSE":       {"cat": "location", "word": "nose"},
    "SKIN":       {"cat": "location", "word": "skin"},
    "HEART":      {"cat": "location", "word": "heart"},
    "LUNG":       {"cat": "location", "word": "lungs"},

    # ── Severity / qualifiers ──
    "SEVERE":     {"cat": "severity", "word": "severe"},
    "MILD":       {"cat": "severity", "word": "mild"},
    "SHARP":      {"cat": "severity", "word": "sharp"},
    "CONSTANT":   {"cat": "severity", "word": "constant"},
    "SUDDEN":     {"cat": "severity", "word": "sudden"},
    "WORSE":      {"cat": "severity", "word": "getting worse"},
    "BETTER":     {"cat": "severity", "word": "getting better"},
    "LONG":       {"cat": "duration",  "word": "for a long time"},
    "TODAY":      {"cat": "duration",  "word": "since today"},
    "DAYS":       {"cat": "duration",  "word": "for a few days"},

    # ── Needs / requests ──
    "WATER":      {"cat": "need",     "word": "water"},
    "MEDICINE":   {"cat": "need",     "word": "medicine"},
    "DOCTOR":     {"cat": "need",     "word": "a doctor"},
    "BATHROOM":   {"cat": "need",     "word": "the bathroom"},
    "REST":       {"cat": "need",     "word": "rest"},
    "FOOD":       {"cat": "need",     "word": "food"},

    # ── Intents ──
    "YES":        {"cat": "intent",   "word": "yes"},
    "NO":         {"cat": "intent",   "word": "no"},
    "HELP":       {"cat": "intent",   "word": "help"},
    "STOP":       {"cat": "control",  "word": "stop"},
    "THANK":      {"cat": "intent",   "word": "thank you"},
    "SORRY":      {"cat": "intent",   "word": "sorry"},
    "AGAIN":      {"cat": "intent",   "word": "again"},
    "WHERE":      {"cat": "question", "word": "where"},
    "WHEN":       {"cat": "question", "word": "when"},
    "HOW_LONG":   {"cat": "question", "word": "how long"},
}


def _unique_ordered(items):
    """Remove duplicates while preserving insertion order."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _join_english(words):
    """Join a list of words with commas and 'and'."""
    if len(words) == 0:
        return ""
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " and " + words[-1]


def build_sentence(token_names):
    """
    Convert an ordered list of token name strings into a natural English
    sentence.  Handles arbitrary combinations of symptoms, locations,
    severity qualifiers, duration, needs, intents, and questions.
    """
    names = [t.upper() for t in token_names]
    if not names:
        return ""

    # ── Categorise ──────────────────────────────────────────────────────────
    symptoms  = _unique_ordered([TOKENS[t]["word"] for t in names if TOKENS.get(t, {}).get("cat") == "symptom"])
    locations = _unique_ordered([TOKENS[t]["word"] for t in names if TOKENS.get(t, {}).get("cat") == "location"])
    severities = _unique_ordered([TOKENS[t]["word"] for t in names if TOKENS.get(t, {}).get("cat") == "severity"])
    durations = _unique_ordered([TOKENS[t]["word"] for t in names if TOKENS.get(t, {}).get("cat") == "duration"])
    needs     = _unique_ordered([TOKENS[t]["word"] for t in names if TOKENS.get(t, {}).get("cat") == "need"])
    intents   = _unique_ordered([TOKENS[t]["word"] for t in names if TOKENS.get(t, {}).get("cat") == "intent"])
    questions = _unique_ordered([TOKENS[t]["word"] for t in names if TOKENS.get(t, {}).get("cat") == "question"])

    has_medical = bool(symptoms or locations)

    # ── 1.  Pure intent (no medical context) ────────────────────────────────
    if not has_medical and not needs and not questions:
        if "help" in intents and len(intents) == 1:
            return "Emergency — I need help immediately."
        if "yes" in intents:
            return "Yes."
        if "no" in intents:
            return "No."
        if "thank you" in intents:
            return "Thank you."
        if "sorry" in intents:
            return "Sorry."
        if intents:
            return intents[0].capitalize() + "."
        # Unknown tokens — fallback
        return ""

    parts = []

    # ── 2.  Medical complaint ───────────────────────────────────────────────
    if symptoms:
        # Separate adjective qualifiers ("severe") from progressive ones ("getting worse")
        adj_quals = [s for s in severities if not s.startswith("getting")]
        prog_quals = [s for s in severities if s.startswith("getting")]

        if adj_quals:
            qual = " ".join(adj_quals)
            # Strip leading article when prepending qualifier: "a cough" → "mild cough"
            def _qualify(s):
                for article in ("a ", "an "):
                    if s.startswith(article):
                        return f"{qual} {s[len(article):]}"
                return f"{qual} {s}"
        else:
            def _qualify(s): return s

        systemic_words = ["fever", "nausea", "dizziness", "vomiting", "tiredness", "difficulty breathing", "weakness", "a cough"]
        sys_symptoms = [s for s in symptoms if s in systemic_words]
        loc_symptoms = [s for s in symptoms if s not in systemic_words]

        if locations:
            loc_str = _join_english(locations)
            if loc_symptoms:
                loc_sym_str = _join_english([_qualify(s) for s in loc_symptoms])
                loc_part = f"{loc_sym_str} in my {loc_str}"
            else:
                loc_part = f"discomfort in my {loc_str}"

            if sys_symptoms:
                sys_sym_str = _join_english([_qualify(s) for s in sys_symptoms])
                parts.append(f"I have {sys_sym_str}, and {loc_part}")
            else:
                parts.append(f"I have {loc_part}")
        else:
            symptom_str = _join_english([_qualify(s) for s in symptoms])
            parts.append(f"I have {symptom_str}")

        if prog_quals:
            parts[-1] += ", and it is " + " and ".join(prog_quals)

    elif locations and not symptoms:
        # Location only (no symptom yet)
        loc_str = _join_english(locations)
        if severities:
            qual = " ".join(severities)
            parts.append(f"I feel {qual} discomfort in my {loc_str}")
        else:
            parts.append(f"Something is wrong with my {loc_str}")

    # ── 3.  Duration ────────────────────────────────────────────────────────
    if durations:
        dur = durations[-1]  # use last duration token
        if parts:
            parts[-1] += f" {dur}"

    # ── 4.  Needs ───────────────────────────────────────────────────────────
    if needs:
        need_str = _join_english(needs)
        if parts:
            parts.append(f"I need {need_str}")
        else:
            parts.append(f"I need {need_str}")

    # ── 5.  Questions ───────────────────────────────────────────────────────
    if questions:
        q = questions[0]
        if q == "where":
            parts.append("Where should I go?")
        elif q == "when":
            parts.append("When will I get better?")
        elif q == "how long":
            parts.append("How long will this take?")

    # ── 6.  Urgency from HELP mixed with medical ───────────────────────────
    if "help" in intents and has_medical:
        parts.append("Please help me")

    # ── Assemble ────────────────────────────────────────────────────────────
    if not parts:
        return ""

    sentence = ". ".join(parts) + "."

    # Clean up punctuation
    sentence = sentence.replace("?.", "?").replace("..", ".").replace(". .", ".")

    return sentence
