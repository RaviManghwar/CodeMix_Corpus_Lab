"""Corpus tools for tagged token DataFrames (SPEC §5.4)."""

from __future__ import annotations

import math
from collections import Counter

import pandas as pd

from src.langid import OTHER

FREQUENCY_COLUMNS = ["word", "lang", "count", "relative_freq_per_1000"]
CONCORDANCE_COLUMNS = ["sent_id", "left", "keyword", "right", "keyword_lang"]
NGRAM_COLUMNS = ["ngram", "count"]
COLLOCATION_COLUMNS = ["collocate", "co_freq", "collocate_freq", "pmi"]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _tokens(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["final_lang"] != OTHER]


def _matches(norm: pd.Series, keyword: str) -> pd.Series:
    return norm.str.casefold() == keyword.strip().casefold()


def frequency(df: pd.DataFrame, lang: str | None = None) -> pd.DataFrame:
    """Return word frequencies per language, sorted by count descending."""
    if df.empty:
        return _empty(FREQUENCY_COLUMNS)

    data = _tokens(df)
    if lang is not None:
        data = data[data["final_lang"] == lang]
    if data.empty:
        return _empty(FREQUENCY_COLUMNS)

    counts = (
        data.groupby(["norm", "final_lang"], sort=False)
        .size()
        .reset_index(name="count")
    )
    counts = counts.rename(columns={"norm": "word", "final_lang": "lang"})
    counts["relative_freq_per_1000"] = counts["count"] / len(data) * 1000
    return counts.sort_values(
        ["count", "word", "lang"], ascending=[False, True, True]
    ).reset_index(drop=True)


def concordance(
    df: pd.DataFrame,
    keyword: str,
    window: int = 5,
    lang: str | None = None,
) -> pd.DataFrame:
    """Return KWIC lines for exact keyword matches, within sentence boundaries."""
    if df.empty or not keyword.strip():
        return _empty(CONCORDANCE_COLUMNS)

    window = max(int(window), 0)
    key = keyword.strip().casefold()
    rows = []
    for sent_id, group in _tokens(df).groupby("sent_id", sort=True):
        norms = group["norm"].tolist()
        tags = group["final_lang"].tolist()
        for position, norm in enumerate(norms):
            if norm.casefold() != key:
                continue
            if lang is not None and tags[position] != lang:
                continue
            rows.append(
                {
                    "sent_id": int(sent_id),
                    "left": " ".join(norms[max(0, position - window) : position]),
                    "keyword": norm,
                    "right": " ".join(norms[position + 1 : position + 1 + window]),
                    "keyword_lang": tags[position],
                }
            )
    return pd.DataFrame(rows, columns=CONCORDANCE_COLUMNS)


def ngrams(
    df: pd.DataFrame,
    n: int = 2,
    lang: str | None = None,
    min_freq: int = 1,
) -> pd.DataFrame:
    """Return within-sentence n-grams, optionally limited to one language."""
    if n < 1:
        raise ValueError("n must be at least 1")
    if df.empty:
        return _empty(NGRAM_COLUMNS)

    counts: Counter[str] = Counter()
    for _, group in _tokens(df).groupby("sent_id", sort=True):
        norms = group["norm"].tolist()
        tags = group["final_lang"].tolist()
        for start in range(len(norms) - n + 1):
            window_tags = tags[start : start + n]
            if lang is not None and any(tag != lang for tag in window_tags):
                continue
            counts[" ".join(norms[start : start + n])] += 1

    rows = [
        {"ngram": ngram, "count": count}
        for ngram, count in counts.items()
        if count >= min_freq
    ]
    result = pd.DataFrame(rows, columns=NGRAM_COLUMNS)
    return result.sort_values(
        ["count", "ngram"], ascending=[False, True]
    ).reset_index(drop=True)


def collocations(
    df: pd.DataFrame,
    keyword: str,
    window: int = 4,
    min_freq: int = 2,
) -> pd.DataFrame:
    """Return PMI-scored collocates within ``window`` tokens of ``keyword``."""
    if df.empty or not keyword.strip():
        return _empty(COLLOCATION_COLUMNS)

    key = keyword.strip().casefold()
    data = _tokens(df)
    matched = _matches(data["norm"], key)
    keyword_freq = int(matched.sum())
    if keyword_freq == 0:
        return _empty(COLLOCATION_COLUMNS)

    window = max(int(window), 0)
    total_tokens = len(data)
    collocate_counts: Counter[str] = Counter()
    for _, group in data.groupby("sent_id", sort=True):
        norms = group["norm"].tolist()
        positions = [
            position
            for position, norm in enumerate(norms)
            if norm.casefold() == key
        ]
        for position in positions:
            start = max(0, position - window)
            end = min(len(norms), position + window + 1)
            for neighbour in range(start, end):
                if neighbour == position:
                    continue
                collocate = norms[neighbour]
                if collocate.casefold() == key:
                    continue
                collocate_counts[collocate] += 1

    corpus_counts = Counter(norm.casefold() for norm in data["norm"])
    rows = []
    for collocate, co_freq in collocate_counts.items():
        if co_freq < min_freq:
            continue
        collocate_freq = corpus_counts[collocate.casefold()]
        pmi = math.log2(
            (co_freq * total_tokens) / (collocate_freq * keyword_freq)
        )
        rows.append(
            {
                "collocate": collocate,
                "co_freq": co_freq,
                "collocate_freq": collocate_freq,
                "pmi": pmi,
            }
        )

    result = pd.DataFrame(rows, columns=COLLOCATION_COLUMNS)
    return result.sort_values(
        ["pmi", "co_freq", "collocate"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
