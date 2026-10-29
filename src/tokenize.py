"""Unicode-safe tokenization for code-mixed SND/URD/ROM/ENG text (SPEC §5.1)."""

from __future__ import annotations

import re
import unicodedata

_TERMINATORS = ".?!۔؟"
_TERMINATOR_SPLIT = re.compile(rf"(?<=[{_TERMINATORS}])\s+")
_ZERO_WIDTH = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u2060\ufeff\ufe0e\ufe0f")
)

_ARABIC_START = (
    r"\u0620-\u063F\u0641-\u064A\u066E-\u066F\u0671-\u06D3\u06D5"
    r"\u06FA-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF"
)
_ARABIC_CONTINUE = _ARABIC_START + r"\u0640\u064B-\u065F\u0670\u06D6-\u06ED"
_ARABIC_WORD = f"[{_ARABIC_START}][{_ARABIC_CONTINUE}]*"

_LATIN_WORD = r"[^\W\d_]+(?:['’][^\W\d_]+)*"
_NUMBER = r"[\d\u0660-\u0669]+(?:[.,][\d\u0660-\u0669]+)*"
_EMOJI = (
    r"[\U0001F000-\U0001FAFF\u2190-\u21FF\u2300-\u23FF"
    r"\u25A0-\u27BF\u2B00-\u2BFF]+"
)

_TOKEN_RE = re.compile(
    "|".join(
        [
            r"https?://[^\s<>\"']+",
            r"www\.[^\s<>\"']+",
            r"@\w+(?:\.\w+)*",
            r"#\w+",
            _ARABIC_WORD,
            _LATIN_WORD,
            _NUMBER,
            _EMOJI,
            r"\S",
        ]
    )
)


def normalize(text: str) -> str:
    """Apply NFC and remove zero-width characters."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text).translate(_ZERO_WIDTH)


def split_sentences(text: str, split_on_terminators: bool = False) -> list[str]:
    """Split text into sentences.

    Sentences are newline-delimited by default. When ``split_on_terminators`` is
    true, also split after ``.``, ``?``, ``!``, ``۔`` or ``؟`` followed by
    whitespace; the terminator stays with the preceding sentence.
    """
    text = normalize(text)
    if not text.strip():
        return []

    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if split_on_terminators:
            sentences.extend(
                part.strip()
                for part in _TERMINATOR_SPLIT.split(line)
                if part.strip()
            )
        else:
            sentences.append(line)
    return sentences


def tokenize(sentence: str) -> list[str]:
    """Tokenize one sentence, keeping words intact and punctuation separate.

    URLs, mentions, and hashtags are emitted as single tokens so later tagging
    can classify them as ``OTHER``.
    """
    if not sentence:
        return []
    return _TOKEN_RE.findall(normalize(sentence))
