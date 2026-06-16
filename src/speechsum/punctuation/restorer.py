from __future__ import annotations

import re
from functools import partial

import structlog

logger = structlog.get_logger()


def restore_punctuation(text: str) -> str:
    if not text or not text.strip():
        return text
    try:
        return _restore_heuristic(text)
    except Exception:
        return text


def _restore_heuristic(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())

    # --- Protect phrases that should stay together ---
    protections = [
        (r"\blet you know\b", "PROTECT_LYK"),
        (r"\blet me know\b", "PROTECT_LMK"),
        (r"\blet us know\b", "PROTECT_LUK"),
    ]
    for p, t in protections:
        text = re.sub(p, t, text, flags=re.IGNORECASE)

    # --- Split on discourse markers that start new sentences ---
    # Only split when the marker follows a word (not at start of text)
    starters = [
        r" " + w + r"\b"
        for w in [
            "well",
            "so",
            "now",
            "anyway",
            "alright",
            "okay",
            "basically",
            "actually",
            "essentially",
            "honestly",
            "frankly",
            "obviously",
            "clearly",
            "certainly",
            "definitely",
            "probably",
            "however",
            "therefore",
            "consequently",
            "furthermore",
            "moreover",
            "nevertheless",
            "meanwhile",
            "afterwards",
            "subsequently",
            "finally",
            "eventually",
            "lastly",
            "in conclusion",
            "to summarize",
            "for example",
            "for instance",
            "in fact",
            "in addition",
            "on the other hand",
            "you know",
            "you see",
            "listen",
            "by the way",
            "incidentally",
        ]
    ]

    for s in starters:
        text = re.sub(s, lambda m: ". " + m.group(0).strip(), text, flags=re.IGNORECASE)

    # Split on "i think/believe/feel/know/mean/guess" when mid-sentence
    i_phrases = r" (i think|i believe|i feel|i know|i mean|i guess|i suppose) "
    text = re.sub(i_phrases, lambda m: ". " + m.group(1).strip() + " ", text, flags=re.IGNORECASE)

    # --- Split on "and/but/or/so" + article/pronoun (independent clause) ---
    connectors = [
        (
            r"\band\s+(the|a|an|this|that|these|those|it|they|he|she|we|i|you|there|here|one|some|many|all|every|no|any|what|which|who)",
            ". And ",
        ),
        (
            r"\bbut\s+(the|a|an|this|that|these|those|it|they|he|she|we|i|you|there|here|one|some|many|all|every)",
            ". But ",
        ),
        (
            r"\bor\s+(the|a|an|this|that|these|those|it|they|he|she|we|i|you|there|here|one|some|many|all)",
            ". Or ",
        ),
        (r"\bso\s+(the|a|an|this|that|these|those|it|they|he|she|we|i|you|there|here)", ". So "),
    ]

    def _prepend_replacement(r: str, m: re.Match[str]) -> str:
        return r + m.group(1)

    for pattern, replacement in connectors:
        text = re.sub(
            pattern, partial(_prepend_replacement, replacement), text, flags=re.IGNORECASE
        )

    # --- Split on "then" as a temporal connector ---
    text = re.sub(
        r" then (the|a|an|this|that|these|those|it|they|he|she|we|i|you|there|we)\b",
        lambda m: ". Then " + m.group(1),
        text,
        flags=re.IGNORECASE,
    )

    # --- Split after temporal references when followed by pronoun ---
    text = re.sub(
        r" (today|yesterday|tomorrow|now|soon|immediately|at first|at last|after that) (i|you|he|she|it|we|they|this|that|these|those|there)\b",
        lambda m: ". " + m.group(1).capitalize() + " " + m.group(2),
        text,
        flags=re.IGNORECASE,
    )

    # --- Split on appositive "this/that" after a noun ---
    text = re.sub(
        r"([a-z]+) (this|that|these|those) (is|are|was|were|has|have|had|will|would|refers|means|shows)",
        lambda m: m.group(1) + ". " + m.group(2).capitalize() + " " + m.group(3),
        text,
        flags=re.IGNORECASE,
    )

    # Split on "in other words", "that is to say", "what this means"
    text = re.sub(
        r" (in other words|that is to say)\b",
        lambda m: ". " + m.group(1).capitalize(),
        text,
        flags=re.IGNORECASE,
    )

    # --- Comma after introductory word at sentence start ---
    text = re.sub(
        r"(?:^|\.)\s*(well|so|now|yes|no|sure|okay|alright|right|first(?:ly)?|second(?:ly)?|third(?:ly)?|finally|lastly)\s+([A-Za-z])",
        lambda m: m.group(0).split(m.group(1))[0] + m.group(1) + ", " + m.group(2),
        text,
        flags=re.IGNORECASE,
    )

    # --- Capitalization ---
    # Capitalize "i" to "I"
    text = re.sub(r"\bi\b", "I", text)

    # Capitalize first word of text and after sentence-ending punctuation
    pieces = re.split(r"([.!?])\s*", text)
    result_parts = []
    for i, piece in enumerate(pieces):
        if i == 0:
            # First piece - capitalize first letter
            result_parts.append(piece[0].upper() + piece[1:] if piece else "")
        elif piece in ".!?":
            result_parts.append(piece)
        elif piece:
            result_parts.append(piece[0].upper() + piece[1:])
        else:
            result_parts.append("")
    text = "".join(result_parts)

    # --- Question detection: check each sentence ---
    sentences = re.split(r"(?<=[.!?])\s+", text)
    question_words = r"\b(what|who|where|when|why|how|which|whose|whom)\b"
    for i, sent in enumerate(sentences):
        if (
            sent
            and not sent.rstrip().endswith("?")
            and re.search(question_words, sent, re.IGNORECASE)
            and re.search(
                r"\b(do|does|did|is|are|was|were|can|could|will|would|shall|should|may|might|have|has|had)\s",
                sent,
                re.IGNORECASE,
            )
        ):
            sentences[i] = sent.rstrip() + "?"
    text = " ".join(sentences)

    # --- Cleanup ---
    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    text = re.sub(r"([.,!?:;])([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\?\.", "?", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\s{2,}", " ", text)

    # Ensure ends with period
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."

    # Fix contractions after capitalization
    contractions = {r"\bIm\b": "I'm", r"\bIll\b": "I'll", r"\bIve\b": "I've", r"\bId\b": "I'd"}
    for p, r in contractions.items():
        text = re.sub(p, r, text)

    # Restore protected phrases
    restorations = {
        "PROTECT_LYK": "let you know",
        "PROTECT_LMK": "let me know",
        "PROTECT_LUK": "let us know",
    }
    for t, p in restorations.items():
        text = text.replace(t, p)

    return text
