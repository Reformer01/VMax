from __future__ import annotations

import re
from functools import partial
from typing import Any

import structlog

logger = structlog.get_logger()

# Label mapping from rpunct: (punctuation, uppercase_this_word)
# Index matches the model's LABEL_X outputs
_LABEL_MAP: list[tuple[str, bool]] = [
    ("", True),  # 0: OU - no punct, uppercase word
    ("", False),  # 1: OO - no punct, lowercase word
    (".", False),  # 2: .O - period, no uppercase
    ("!", False),  # 3: !O - exclamation, no uppercase
    (",", False),  # 4: ,O - comma, no uppercase
    (".", True),  # 5: .U - period, uppercase word
    ("!", True),  # 6: !U - exclamation, uppercase word
    (",", True),  # 7: ,U - comma, uppercase word
    (":", False),  # 8: :O - colon, no uppercase
    (";", False),  # 9: ;O - semicolon, no uppercase
    (":", True),  # 10: :U - colon, uppercase word
    ("'", False),  # 11: 'O - apostrophe, no uppercase
    ("-", False),  # 12: -O - dash, no uppercase
    ("?", False),  # 13: ?O - question mark, no uppercase
    ("?", True),  # 14: ?U - question mark, uppercase word
]

_ML_PIPELINE: Any = None


def _get_ml_pipeline() -> Any:
    global _ML_PIPELINE
    if _ML_PIPELINE is not None:
        return _ML_PIPELINE
    try:
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        model_name = "felflare/bert-restore-punctuation"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            id2label={i: f"LABEL_{i}" for i in range(15)},
            label2id={f"LABEL_{i}": i for i in range(15)},
        )
        _ML_PIPELINE = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="none",
        )
        logger.info("ml_punctuation_model_loaded", model=model_name)
        return _ML_PIPELINE
    except Exception as exc:
        logger.warning("ml_punctuation_model_unavailable", error=str(exc))
        return None


def restore_punctuation(text: str) -> str:
    if not text or not text.strip():
        return text

    ml = _get_ml_pipeline()
    if ml is not None:
        try:
            return _restore_with_ml(text, ml)
        except Exception as exc:
            logger.warning("ml_punctuation_failed_falling_back", error=str(exc))

    try:
        return _restore_heuristic(text)
    except Exception:
        return text


def _predict_words(text: str, ml_pipeline: Any) -> list[tuple[str, int]]:
    """Predict a label index for each word in text. Returns [(word, label_idx), ...]."""
    words = text.split()
    if not words:
        return []

    chunk_size = 200
    overlap = 30
    all_results: list[tuple[str, int]] = []
    emitted = 0

    for chunk_start in range(0, len(words), chunk_size):
        chunk_end = min(chunk_start + chunk_size + overlap, len(words))
        chunk_words = words[chunk_start:chunk_end]
        chunk_text = " ".join(chunk_words)

        predictions = ml_pipeline(chunk_text)

        word_labels: list[int] = []
        pred_idx = 0
        n = len(predictions)

        for word in chunk_words:
            prefix_len = sum(len(w) + 1 for w in chunk_words[: chunk_words.index(word)])
            word_start = (
                chunk_text.find(word, prefix_len) if prefix_len > 0 else chunk_text.find(word)
            )

            best_label = 1
            best_score = 0.0

            while pred_idx < n:
                pred = predictions[pred_idx]
                if pred["start"] >= word_start and pred["start"] < word_start + len(word):
                    if pred["score"] > best_score:
                        label_num = int(pred["entity"].replace("LABEL_", ""))
                        best_label = label_num
                        best_score = pred["score"]
                    pred_idx += 1
                elif pred["start"] < word_start:
                    pred_idx += 1
                else:
                    break

            word_labels.append(best_label)

        # Emit only the non-overlapping portion: first chunk_size words of this chunk
        emit_end = min(chunk_size, len(chunk_words))
        for w, lbl in zip(chunk_words[:emit_end], word_labels[:emit_end], strict=True):
            if emitted < len(words):
                all_results.append((w, lbl))
                emitted += 1

    return all_results


def _restore_with_ml(text: str, ml_pipeline: Any) -> str:
    text = re.sub(r"\s+", " ", text.strip())

    # Protect phrases
    protections = [
        (r"\blet you know\b", "PROTECT_LYK"),
        (r"\blet me know\b", "PROTECT_LMK"),
        (r"\blet us know\b", "PROTECT_LUK"),
    ]
    for p, t in protections:
        text = re.sub(p, t, text, flags=re.IGNORECASE)

    word_labels = _predict_words(text, ml_pipeline)
    if not word_labels:
        return text

    result_parts: list[str] = []
    for _, (word, label_idx) in enumerate(word_labels):
        punct, uppercase = _LABEL_MAP[label_idx]
        out_word = word.capitalize() if uppercase else word
        result_parts.append(out_word + punct)

    result = " ".join(result_parts)

    # --- Post-process ML output for common missed patterns ---
    # Split on "then/thus/hence" and "in conclusion", "for example" etc. (case-insensitive)
    result = re.sub(
        r" (\w+)\s+(then|thus|hence) (the|a|an|this|that|these|those|it|they|he|she|we|i|you|there)\b",
        lambda m: f" {m.group(1)}. {m.group(2).capitalize()} {m.group(3)}",
        result,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r" (\w+)\s+(in conclusion|to summarize|for example|for instance|in addition|on the other hand)\b",
        lambda m: f" {m.group(1)}. {m.group(2).capitalize()}",
        result,
        flags=re.IGNORECASE,
    )
    # Split on "this word/these questions/that fact" etc. starting a new sentence
    result = re.sub(
        r" (\w+)\s+(this|that|these|those)\s+(word|question|fact|idea|concept|notion|point|issue|matter)",
        lambda m: f" {m.group(1)}. {m.group(2).capitalize()} {m.group(3)}",
        result,
        flags=re.IGNORECASE,
    )
    # Ensure capitalization after sentence-ending punctuation (ML model sometimes misses)
    result = re.sub(r"(\.|!|\?)\s+([a-z])", lambda m: m.group(1) + " " + m.group(2).upper(), result)

    # Cleanup spacing around punctuation
    result = re.sub(r"\s+([.,!?:;])", r"\1", result)
    result = re.sub(r"([.,!?:;])([a-zA-Z])", r"\1 \2", result)
    result = re.sub(r"[,;]+\.", ".", result)
    result = re.sub(r"[,;]+\?", "?", result)
    result = re.sub(r"\.{2,}", ".", result)
    result = re.sub(r"\?\.", "?", result)
    result = re.sub(r",{2,}", ",", result)
    result = re.sub(r"\?{2,}", "?", result)
    result = re.sub(r"\s{2,}", " ", result)

    # Ensure ends with sentence-ending punctuation
    result = result.strip()
    if result:
        last = result[-1]
        if last == "," or last == ";" or last == ":":
            result = result[:-1].rstrip() + "."
        elif last not in ".!?":
            result += "."

    # Fix contractions
    contractions = {r"\bIm\b": "I'm", r"\bIll\b": "I'll", r"\bIve\b": "I've", r"\bId\b": "I'd"}
    for p, r in contractions.items():
        result = re.sub(p, r, result)

    # Restore protected
    restorations = {
        "PROTECT_LYK": "let you know",
        "PROTECT_LMK": "let me know",
        "PROTECT_LUK": "let us know",
    }
    for t, p in restorations.items():
        result = result.replace(t, p)

    return result


def _restore_heuristic(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())

    protections = [
        (r"\blet you know\b", "PROTECT_LYK"),
        (r"\blet me know\b", "PROTECT_LMK"),
        (r"\blet us know\b", "PROTECT_LUK"),
    ]
    for p, t in protections:
        text = re.sub(p, t, text, flags=re.IGNORECASE)

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

    i_phrases = r" (i think|i believe|i feel|i know|i mean|i guess|i suppose) "
    text = re.sub(i_phrases, lambda m: ". " + m.group(1).strip() + " ", text, flags=re.IGNORECASE)

    connector_words = (
        r"\b(the|a|an|this|that|these|those|it|they|he|she|we|i|you|"
        r"there|here|one|some|many|all|every|no|any|what|which|who)\b"
    )
    connectors = [
        (r"\band\s+" + connector_words, ". And "),
        (r"\bbut\s+" + connector_words, ". But "),
        (r"\bor\s+" + connector_words, ". Or "),
        (r"\bso\s+" + connector_words, ". So "),
    ]

    def _prepend(r: str, m: re.Match[str]) -> str:
        return r + m.group(1)

    for pattern, replacement in connectors:
        text = re.sub(pattern, partial(_prepend, replacement), text, flags=re.IGNORECASE)

    text = re.sub(
        r" then (the|a|an|this|that|these|those|it|they|he|she|we|i|you|there|we)\b",
        lambda m: ". Then " + m.group(1),
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r" (today|yesterday|tomorrow|now|soon|immediately|at first|at last|after that) "
        r"(i|you|he|she|it|we|they|this|that|these|those|there)\b",
        lambda m: ". " + m.group(1).capitalize() + " " + m.group(2),
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"([a-z]+) (this|that|these|those) (is|are|was|were|has|have|had|will|would|refers|means|shows)",
        lambda m: m.group(1) + ". " + m.group(2).capitalize() + " " + m.group(3),
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r" (in other words|that is to say)\b",
        lambda m: ". " + m.group(1).capitalize(),
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(?:^|\.)\s*(well|so|now|yes|no|sure|okay|alright|right|"
        r"first(?:ly)?|second(?:ly)?|third(?:ly)?|finally|lastly)\s+([A-Za-z])",
        lambda m: m.group(0).split(m.group(1))[0] + m.group(1) + ", " + m.group(2),
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\bi\b", "I", text)

    pieces = re.split(r"([.!?])\s*", text)
    result_parts = []
    for i, piece in enumerate(pieces):
        if i == 0:
            result_parts.append(piece[0].upper() + piece[1:] if piece else "")
        elif piece in ".!?":
            result_parts.append(piece)
        elif piece:
            result_parts.append(piece[0].upper() + piece[1:])
        else:
            result_parts.append("")
    text = "".join(result_parts)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    question_words = r"\b(what|who|where|when|why|how|which|whose|whom)\b"
    for i, sent in enumerate(sentences):
        if (
            sent
            and not sent.rstrip().endswith("?")
            and re.search(question_words, sent, re.IGNORECASE)
            and re.search(
                r"\b(do|does|did|is|are|was|were|can|could|will|would|"
                r"shall|should|may|might|have|has|had)\s",
                sent,
                re.IGNORECASE,
            )
        ):
            sentences[i] = sent.rstrip() + "?"
    text = " ".join(sentences)

    text = re.sub(r"\s+([.,!?:;])", r"\1", text)
    text = re.sub(r"([.,!?:;])([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\?\.", "?", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."

    contractions = {r"\bIm\b": "I'm", r"\bIll\b": "I'll", r"\bIve\b": "I've", r"\bId\b": "I'd"}
    for p, r in contractions.items():
        text = re.sub(p, r, text)

    restorations = {
        "PROTECT_LYK": "let you know",
        "PROTECT_LMK": "let me know",
        "PROTECT_LUK": "let us know",
    }
    for t, p in restorations.items():
        text = text.replace(t, p)

    return text
