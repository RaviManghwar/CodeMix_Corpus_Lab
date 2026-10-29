"""Rule-based language identification for SND/URD/ROM/ENG text (SPEC §5.2)."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.tokenize import normalize, split_sentences, tokenize

LANGS = ("SND", "URD", "ROM", "ENG", "OTHER")
OTHER = "OTHER"
FALLBACK = "ENG"
COLUMNS = [
    "sent_id",
    "tok_id",
    "token",
    "norm",
    "auto_lang",
    "manual_lang",
    "final_lang",
]

LEXICON_DIR = Path(__file__).resolve().parent.parent / "data" / "lexicons"
LEXICON_FILES = {
    "english": "english.txt",
    "roman_urdu": "roman_urdu.txt",
    "urdu": "urdu.txt",
    "sindhi": "sindhi.txt",
    "ambiguous": "ambiguous.txt",
}

SINDHI_LETTERS = frozenset("ڄڃڇڏڊڍڌڙڻٺٽٿڀڦڪڳڱھ")

_URL = re.compile(r"^(?:https?://|www\.)", re.IGNORECASE)
_MENTION = re.compile(r"^@\w")
_HASHTAG = re.compile(r"^#\w")
_NUMBER = re.compile(r"^[0-9\u0660-\u0669]+(?:[.,][0-9\u0660-\u0669]+)*$")
_WORD_CHAR = re.compile(r"\w")
_ARABIC_CHAR = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_LATIN_CHAR = re.compile(r"[A-Za-z\u00C0-\u024F]")


@lru_cache(maxsize=None)
def _load(directory: str) -> dict[str, frozenset[str]]:
    base = Path(directory)
    return {
        name: frozenset(
            line.strip().lower()
            for line in (base / filename).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        for name, filename in LEXICON_FILES.items()
    }


def load_lexicons(path: str | Path | None = None) -> dict[str, frozenset[str]]:
    """Load lexicons from a directory, defaulting to ``data/lexicons``."""
    return _load(str(path) if path is not None else str(LEXICON_DIR))


def _tag_arabic(token: str) -> str:
    if any(char in SINDHI_LETTERS for char in token):
        return "SND"
    if token in load_lexicons()["sindhi"]:
        return "SND"
    return "URD"


def _tag_latin(token: str) -> str | None:
    word = token.lower()
    if word in load_lexicons()["ambiguous"]:
        return None
    if word in load_lexicons()["roman_urdu"]:
        return "ROM"
    if word in load_lexicons()["english"]:
        return "ENG"
    return OTHER


def _base_tag(token: str) -> str | None:
    text = normalize(token)
    if not text:
        return OTHER
    if _URL.match(text) or _MENTION.match(text) or _HASHTAG.match(text):
        return OTHER
    if _NUMBER.fullmatch(text):
        return OTHER
    if not _WORD_CHAR.search(text):
        return OTHER
    if _ARABIC_CHAR.search(text):
        return _tag_arabic(text)
    if _LATIN_CHAR.search(text):
        return _tag_latin(text)
    return OTHER


def _is_context_tag(tag: str | None) -> bool:
    return tag is not None and tag != OTHER


def tag_token(token: str, prev_tag: str | None = None) -> str:
    """Tag a single token, optionally resolving ambiguity from ``prev_tag``."""
    tag = _base_tag(token)
    if tag is not None:
        return tag
    if prev_tag in LANGS and prev_tag != OTHER:
        return prev_tag
    return FALLBACK


def tag_sentence(tokens: list[str]) -> list[str]:
    """Tag a sequence of tokens, resolving ambiguous words from neighbours."""
    base_tags = [_base_tag(token) for token in tokens]
    tags: list[str] = []
    for index, tag in enumerate(base_tags):
        if tag is not None:
            tags.append(tag)
            continue
        left = next(
            (
                base_tags[i]
                for i in range(index - 1, -1, -1)
                if _is_context_tag(base_tags[i])
            ),
            None,
        )
        right = next(
            (
                base_tags[i]
                for i in range(index + 1, len(base_tags))
                if _is_context_tag(base_tags[i])
            ),
            None,
        )
        tags.append(left or right or FALLBACK)
    return tags


def _norm(token: str) -> str:
    text = normalize(token)
    if _LATIN_CHAR.search(text):
        text = text.lower()
        text = re.sub(r"([A-Za-z])\1{2,}", r"\1", text)
    return text


def tag_corpus(text: str) -> pd.DataFrame:
    """Tag a corpus into the SPEC §4 token DataFrame."""
    rows: list[dict[str, object]] = []
    for sent_id, sentence in enumerate(split_sentences(text)):
        tokens = tokenize(sentence)
        for tok_id, (token, lang) in enumerate(zip(tokens, tag_sentence(tokens))):
            rows.append(
                {
                    "sent_id": sent_id,
                    "tok_id": tok_id,
                    "token": token,
                    "norm": _norm(token),
                    "auto_lang": lang,
                    "manual_lang": "",
                    "final_lang": lang,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)
