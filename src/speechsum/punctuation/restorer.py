from __future__ import annotations

import re
import structlog

logger = structlog.get_logger()

# Try to load the ML model lazily; if it fails, fall back to heuristic
_ml_pipeline = None
_ml_model_failed = False


def _get_ml_pipeline():
    global _ml_pipeline, _ml_model_failed
    if _ml_model_failed:
        return None
    if _ml_pipeline is None:
        try:
            from transformers import pipeline
            logger.info("punctuation_model_loading", model="felflare/bert-restore-punctuation")
            _ml_pipeline = pipeline(
                "token-classification",
                model="felflare/bert-restore-punctuation",
                aggregation_strategy="simple",
            )
            logger.info("punctuation_model_loaded")
        except Exception as e:
            logger.warning("punctuation_model_load_failed", error=str(e))
            _ml_model_failed = True
            return None
    return _ml_pipeline


def restore_punctuation(text: str) -> str:
    """Restore punctuation and capitalization to raw transcript text."""
    if not text or not text.strip():
        return text

    # Try ML model first
    pipe = _get_ml_pipeline()
    if pipe is not None:
        try:
            return _restore_with_ml(text, pipe)
        except Exception as e:
            logger.warning("ml_punctuation_failed", error=str(e))

    # Fallback: heuristic punctuation
    return _restore_heuristic(text)


def _restore_with_ml(text: str, pipe) -> str:
    entities = pipe(text)
    result = []
    last_end = 0
    punct_map = {
        "PERIOD": ".",
        "COMMA": ",",
        "QUESTION": "?",
        "EXCLAMATION": "!",
        "COLON": ":",
        "SEMICOLON": ";",
    }

    for ent in entities:
        if ent["start"] > last_end:
            result.append(text[last_end : ent["start"]])
        word = ent["word"]
        punct = punct_map.get(ent["entity_group"], "")
        result.append(word + punct)
        last_end = ent["end"]

    if last_end < len(text):
        result.append(text[last_end:])

    return "".join(result)


def _restore_heuristic(text: str) -> str:
    """Heuristic: split on pause words, capitalize, add periods."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text.strip())

    # Protect verb phrases that contain discourse markers
    # e.g., "let you know" -> "LET_YOU_KNOW" temporarily
    protections = {
        r"\blet you know\b": "LET_YOU_KNOW",
        r"\btell you know\b": "TELL_YOU_KNOW",
        r"\blet me know\b": "LET_ME_KNOW",
        r"\blet us know\b": "LET_US_KNOW",
    }
    for pattern, token in protections.items():
        text = re.sub(pattern, token, text, flags=re.IGNORECASE)

    # Common discourse markers that start new sentences
    # Note: we only split when these appear at clause boundaries, not mid-phrase
    starters = [
        "i think", "i believe", "i feel", "i know", "i guess",
        "you know", "you see", "look", "listen", "well", "so", "anyway",
        "basically", "actually", "essentially", "literally", "obviously",
        "clearly", "certainly", "definitely", "probably", "maybe",
        "however", "therefore", "consequently", "furthermore", "moreover",
        "nevertheless", "meanwhile", "afterwards", "subsequently",
        "finally", "eventually", "first", "second", "third", "lastly",
        "in conclusion", "to summarize", "for example", "in fact",
        "on the other hand", "in addition", "what about", "by the way",
    ]

    # Question phrases that typically end a sentence (longer, specific ones)
    question_phrases = [
        "what do you think about",
        "how do you feel about",
        "what are your thoughts on",
        "what is your opinion on",
        "can you tell me",
        "could you explain",
        "would you mind",
        "do you know",
        "did you know",
        "have you heard",
        "have you seen",
        "what do you think",
        "what do you mean",
        "why not",
    ]

    # Sort by length descending to match longest first
    starters.sort(key=len, reverse=True)

    for s in starters:
        # Match word boundary + starter at start of sentence or after space
        pattern = r"(?<=\s)" + re.escape(s) + r"\b"
        text = re.sub(pattern, r". " + s, text, flags=re.IGNORECASE)

    # Also handle sentence-initial starters
    for s in starters:
        pattern = r"^" + re.escape(s) + r"\b"
        text = re.sub(pattern, s.capitalize(), text, flags=re.IGNORECASE)

    # Question phrases: add question mark before them
    question_phrases.sort(key=len, reverse=True)
    for q in question_phrases:
        pattern = r"(?<=\s)" + re.escape(q) + r"\b"
        text = re.sub(pattern, r"? " + q, text, flags=re.IGNORECASE)

    for q in question_phrases:
        pattern = r"^" + re.escape(q) + r"\b"
        text = re.sub(pattern, q.capitalize(), text, flags=re.IGNORECASE)

    # Split on long comma sequences (indicates pause)
    text = re.sub(r"\s*,\s*,\s*", ". ", text)
    text = re.sub(r"\s+and\s+and\s+", ". ", text)

    # Capitalize first letter
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]

    # Capitalize after sentence endings
    text = re.sub(r"([.!?])\s+([a-z])", lambda m: m.group(1) + " " + m.group(2).upper(), text)

    # Ensure ends with period
    if text and text[-1] not in ".!?":
        text += "."

    # Fix spacing around punctuation
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = re.sub(r"([.,!?])([A-Za-z])", r"\1 \2", text)

    # Fix double spaces
    text = re.sub(r"\s{2,}", " ", text)

    # Restore protected phrases
    restorations = {
        "LET_YOU_KNOW": "let you know",
        "TELL_YOU_KNOW": "tell you know",
        "LET_ME_KNOW": "let me know",
        "LET_US_KNOW": "let us know",
    }
    for token, phrase in restorations.items():
        text = text.replace(token, phrase)

    return text