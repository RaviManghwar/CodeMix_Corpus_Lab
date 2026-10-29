"""Tests for src.corpus_tools (SPEC §5.4)."""

import math

import pandas as pd
import pytest

from src.langid import COLUMNS
from src.corpus_tools import (
    COLLOCATION_COLUMNS,
    CONCORDANCE_COLUMNS,
    FREQUENCY_COLUMNS,
    NGRAM_COLUMNS,
    collocations,
    concordance,
    frequency,
    ngrams,
)


def make_df(sentences: list[list[tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    for sent_id, tokens in enumerate(sentences):
        for tok_id, (token, tag) in enumerate(tokens):
            rows.append(
                {
                    "sent_id": sent_id,
                    "tok_id": tok_id,
                    "token": token,
                    "norm": token.lower(),
                    "auto_lang": tag,
                    "manual_lang": "",
                    "final_lang": tag,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def test_frequency_exact_counts_and_relative_frequency():
    df = make_df(
        [
            [
                ("Mera", "ROM"),
                ("laptop", "ENG"),
                ("kharab", "ROM"),
                ("hai", "ROM"),
                (",", "OTHER"),
            ],
            [
                ("Mera", "ROM"),
                ("laptop", "ENG"),
                ("theek", "ROM"),
                ("hai", "ROM"),
            ],
        ]
    )
    result = frequency(df)

    assert list(result.columns) == FREQUENCY_COLUMNS
    assert result["word"].tolist() == ["hai", "laptop", "mera", "kharab", "theek"]
    assert result["lang"].tolist() == ["ROM", "ENG", "ROM", "ROM", "ROM"]
    assert result["count"].tolist() == [2, 2, 2, 1, 1]
    assert result["relative_freq_per_1000"].tolist() == pytest.approx(
        [250, 250, 250, 125, 125]
    )


def test_frequency_language_filter_uses_filtered_denominator():
    df = make_df(
        [
            [("Mera", "ROM"), ("laptop", "ENG"), ("kharab", "ROM")],
            [("Mera", "ROM"), ("laptop", "ENG"), ("hai", "ROM")],
        ]
    )
    result = frequency(df, lang="ROM")

    assert result["word"].tolist() == ["mera", "hai", "kharab"]
    assert result["count"].tolist() == [2, 1, 1]
    assert result["relative_freq_per_1000"].tolist() == pytest.approx(
        [2 / 4 * 1000, 1 / 4 * 1000, 1 / 4 * 1000]
    )


def test_frequency_uses_final_lang_and_skips_other():
    df = make_df([ [("Mera", "ROM"), ("laptop", "ENG"), (",", "OTHER")] ])
    df["auto_lang"] = "SND"
    result = frequency(df)

    assert "OTHER" not in result["lang"].tolist()
    assert "SND" not in result["lang"].tolist()
    assert result["lang"].tolist() == ["ENG", "ROM"]


def test_frequency_empty():
    assert frequency(empty_df()).empty
    assert list(frequency(empty_df()).columns) == FREQUENCY_COLUMNS


def test_concordance_windows_and_sentence_boundaries():
    df = make_df(
        [
            [
                ("Start", "ENG"),
                ("middle", "ROM"),
                (",", "OTHER"),
                ("end", "ENG"),
            ],
            [("next", "ENG"), ("MIDDLE", "ENG"), ("last", "ENG")],
        ]
    )
    result = concordance(df, "MIDDLE", window=5)

    assert list(result.columns) == CONCORDANCE_COLUMNS
    assert result["sent_id"].tolist() == [0, 1]
    assert result["left"].tolist() == ["start", "next"]
    assert result["keyword"].tolist() == ["middle", "middle"]
    assert result["right"].tolist() == ["end", "last"]
    assert result["keyword_lang"].tolist() == ["ROM", "ENG"]


def test_concordance_sentence_edge_windows():
    df = make_df([ [("start", "ENG"), ("middle", "ROM"), ("end", "ENG")] ])

    first = concordance(df, "start", window=1)
    last = concordance(df, "end", window=1)

    assert first["left"].tolist() == [""]
    assert first["right"].tolist() == ["middle"]
    assert last["left"].tolist() == ["middle"]
    assert last["right"].tolist() == [""]


def test_concordance_language_filter():
    df = make_df(
        [
            [("middle", "ROM")],
            [("middle", "ENG")],
        ]
    )
    result = concordance(df, "middle", lang="ENG")

    assert result["sent_id"].tolist() == [1]
    assert result["keyword_lang"].tolist() == ["ENG"]


def test_concordance_no_match_and_empty():
    df = make_df([ [("hello", "ENG")] ])
    assert concordance(df, "missing").empty
    assert list(concordance(df, "missing").columns) == CONCORDANCE_COLUMNS
    assert concordance(empty_df(), "hello").empty


def test_ngrams_do_not_cross_sentences():
    df = make_df(
        [
            [("a", "ROM"), ("b", "ROM"), ("c", "ENG")],
            [("d", "ROM"), ("e", "ROM")],
        ]
    )
    result = ngrams(df, n=2)

    assert list(result.columns) == NGRAM_COLUMNS
    assert result["ngram"].tolist() == ["a b", "b c", "d e"]
    assert result["count"].tolist() == [1, 1, 1]
    assert "c d" not in result["ngram"].tolist()


def test_ngrams_counts_and_min_frequency():
    df = make_df(
        [
            [("a", "ROM"), ("b", "ROM"), ("a", "ROM"), ("b", "ROM")],
        ]
    )
    result = ngrams(df, n=2)

    assert result["ngram"].tolist() == ["a b", "b a"]
    assert result["count"].tolist() == [2, 1]
    assert ngrams(df, n=2, min_freq=2)["ngram"].tolist() == ["a b"]


def test_ngrams_language_filter_requires_every_token():
    df = make_df([ [("a", "ROM"), ("b", "ENG"), ("c", "ROM")] ])

    assert ngrams(df, n=2, lang="ROM").empty
    assert ngrams(df, n=1, lang="ROM")["ngram"].tolist() == ["a", "c"]


def test_ngrams_empty_and_invalid_n():
    assert ngrams(empty_df()).empty
    with pytest.raises(ValueError):
        ngrams(make_df([ [("a", "ROM")] ]), n=0)


def test_collocations_exact_pmi():
    df = make_df(
        [
            [
                ("chai", "ROM"),
                ("acha", "ROM"),
                ("chai", "ROM"),
                ("bura", "ROM"),
            ]
        ]
    )
    result = collocations(df, "CHAI", window=1, min_freq=1)

    assert list(result.columns) == COLLOCATION_COLUMNS
    assert result["collocate"].tolist() == ["acha", "bura"]
    assert result["co_freq"].tolist() == [2, 1]
    assert result["collocate_freq"].tolist() == [1, 1]
    assert result["pmi"].tolist() == pytest.approx([2.0, 1.0])


def test_collocations_min_frequency():
    df = make_df(
        [
            [
                ("chai", "ROM"),
                ("acha", "ROM"),
                ("chai", "ROM"),
                ("bura", "ROM"),
            ]
        ]
    )
    result = collocations(df, "chai", window=1, min_freq=2)

    assert result["collocate"].tolist() == ["acha"]


def test_collocations_do_not_cross_sentences():
    df = make_df(
        [
            [("chai", "ROM"), ("acha", "ROM")],
            [("bura", "ROM")],
        ]
    )
    result = collocations(df, "chai", window=5, min_freq=1)

    assert "bura" not in result["collocate"].tolist()
    assert result["collocate"].tolist() == ["acha"]


def test_collocations_no_keyword_and_empty():
    df = make_df([ [("chai", "ROM")] ])
    assert collocations(df, "missing").empty
    assert list(collocations(df, "missing").columns) == COLLOCATION_COLUMNS
    assert collocations(empty_df(), "chai").empty


def test_pmi_matches_formula():
    df = make_df(
        [
            [
                ("chai", "ROM"),
                ("acha", "ROM"),
                ("chai", "ROM"),
                ("bura", "ROM"),
            ]
        ]
    )
    result = collocations(df, "chai", window=1, min_freq=1)
    acha = result[result["collocate"] == "acha"].iloc[0]

    expected = math.log2(
        (acha["co_freq"] * 4) / (acha["collocate_freq"] * 2)
    )
    assert acha["pmi"] == pytest.approx(expected)
