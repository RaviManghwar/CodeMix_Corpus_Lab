"""Code-switching analysis for tagged token DataFrames (SPEC §5.3)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator

import pandas as pd

from src.langid import LANGS, OTHER

LANGUAGES = tuple(lang for lang in LANGS if lang != OTHER)


def _tags_by_sentence(df: pd.DataFrame) -> list[tuple[int, list[str]]]:
    if df.empty:
        return []
    return [
        (int(sent_id), group["final_lang"].tolist())
        for sent_id, group in df.groupby("sent_id", sort=True)
    ]


def _iter_switches(tags: list[str]) -> Iterator[tuple[int, str, str]]:
    previous: str | None = None
    for position, tag in enumerate(tags):
        if tag == OTHER:
            continue
        if previous is not None and tag != previous:
            yield position, previous, tag
        previous = tag


def find_switches(tags: list[str]) -> list[int]:
    """Return positions where the language changes, ignoring ``OTHER`` tokens.

    Each position is the index of the token that starts the new language;
    comparison is against the previous non-``OTHER`` token.
    """
    return [position for position, _, _ in _iter_switches(tags)]


def classify_sentence(tags: list[str]) -> str:
    """Classify a sentence as ``"intra"`` if it switches, else ``"single-lang"``."""
    return "intra" if find_switches(tags) else "single-lang"


def _dominant_language(tags: list[str]) -> str | None:
    counts = Counter(tag for tag in tags if tag != OTHER)
    if not counts:
        return None
    return max(counts, key=counts.get)


def inter_sentential(df: pd.DataFrame) -> list[int]:
    """Return sent_ids whose dominant language differs from the previous one.

    Dominant language is the most frequent non-``OTHER`` tag, with ties broken
    by first appearance. Sentences without a dominant language are skipped.
    """
    switched: list[int] = []
    previous: str | None = None
    for sent_id, tags in _tags_by_sentence(df):
        dominant = _dominant_language(tags)
        if dominant is None:
            continue
        if previous is not None and dominant != previous:
            switched.append(sent_id)
        previous = dominant
    return switched


def switch_rate(df: pd.DataFrame) -> float:
    """Return intra-sentential switches per 100 non-``OTHER`` tokens."""
    switches = 0
    tokens = 0
    for _, tags in _tags_by_sentence(df):
        switches += len(find_switches(tags))
        tokens += sum(1 for tag in tags if tag != OTHER)
    return (switches / tokens * 100) if tokens else 0.0


def summary(df: pd.DataFrame) -> dict:
    """Return distribution, switch counts/rate, and the direction matrix."""
    sentences = _tags_by_sentence(df)

    distribution = {lang: 0 for lang in LANGS}
    for _, tags in sentences:
        for tag in tags:
            distribution[tag] = distribution.get(tag, 0) + 1

    intra = sum(len(find_switches(tags)) for _, tags in sentences)
    inter = len(inter_sentential(df))
    matrix = {source: {target: 0 for target in LANGUAGES} for source in LANGUAGES}
    for _, tags in sentences:
        for _, source, target in _iter_switches(tags):
            matrix[source][target] += 1

    return {
        "distribution": distribution,
        "switch_rate": switch_rate(df),
        "intra_sentential": intra,
        "inter_sentential": inter,
        "direction_matrix": matrix,
    }
