"""Categorize tokens into semantic slots for the grammar engine."""

from pipeline.template_engine import TOKENS


class SlotFiller:
    def fill(self, token_names):
        slots = {
            "symptoms": [],
            "locations": [],
            "severity": [],
            "duration": [],
            "needs": [],
            "intent": None,
            "question": None,
        }

        for token in token_names:
            t = token.upper()
            info = TOKENS.get(t)
            if not info:
                continue

            cat = info["cat"]
            word = info["word"]

            if cat == "symptom":
                if word not in slots["symptoms"]:
                    slots["symptoms"].append(word)
            elif cat == "location":
                if word not in slots["locations"]:
                    slots["locations"].append(word)
            elif cat == "severity":
                if word not in slots["severity"]:
                    slots["severity"].append(word)
            elif cat == "duration":
                slots["duration"] = [word]  # last one wins
            elif cat == "need":
                if word not in slots["needs"]:
                    slots["needs"].append(word)
            elif cat == "intent":
                if t == "HELP" and not slots["symptoms"] and not slots["locations"]:
                    return {"intent": "emergency"}
                slots["intent"] = word
            elif cat == "question":
                slots["question"] = word

        return slots
