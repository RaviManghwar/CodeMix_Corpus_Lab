"""Tests for src.switching (SPEC §5.3)."""

import pandas as pd
import pytest

from src.langid import COLUMNS
from src.switching import (
    classify_sentence,
    find_switches,
    inter_sentential,
    summary,
    switch_rate,
)


def make_df(sentences: list[list[str]]) -> pd.DataFrame:
    rows = []
    for sent_id, tags in enumerate(sentences):
        for tok_id, tag in enumerate(tags):
            rows.append(
                {
                    "sent_id": sent_id,
                    "tok_id": tok_id,
                    "token": f"t{tok_id}",
                    "norm": f"t{tok_id}",
                    "auto_lang": tag,
                    "manual_lang": "",
                    "final_lang": tag,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def test_find_switches_monolingual():
    assert find_switches([]) == []
    assert find_switches(["ENG", "ENG", "ENG"]) == []
    assert find_switches(["OTHER", "OTHER"]) == []


def test_find_switches_one_switch():
    assert find_switches(["ENG", "ENG", "ROM"]) == [2]


def test_find_switches_ignores_other_between_languages():
    assert find_switches(["ENG", "OTHER", "ROM", "OTHER", "ENG"]) == [2, 4]


def test_find_switches_ignores_other_for_same_language():
    assert find_switches(["ENG", "OTHER", "ENG"]) == []


def test_classify_sentence():
    assert classify_sentence(["ENG", "ENG", "OTHER"]) == "single-lang"
    assert classify_sentence(["ENG", "ROM"]) == "intra"
    assert classify_sentence([]) == "single-lang"


def test_inter_sentential():
    df = make_df([["ENG", "ENG"], ["ROM", "ROM"], ["ROM", "ENG"]])
    assert inter_sentential(df) == [1]


def test_inter_sentential_ties_go_to_first_language():
    df = make_df([["ROM", "ROM"], ["ROM", "ENG"]])
    assert inter_sentential(df) == []


def test_inter_sentential_uses_dominant_language():
    df = make_df([["ENG", "ENG", "ENG"], ["ENG", "ROM", "ROM", "ROM"]])
    assert inter_sentential(df) == [1]


def test_inter_sentential_skips_other_only_sentences():
    df = make_df([["ENG"], ["OTHER", "OTHER"], ["ROM"]])
    assert inter_sentential(df) == [2]


def test_inter_sentential_empty():
    assert inter_sentential(empty_df()) == []


def test_switch_rate_one_switch():
    df = make_df([["ENG", "ENG", "ROM"]])
    assert switch_rate(df) == pytest.approx(100 / 3)


def test_switch_rate_excludes_other_tokens():
    df = make_df([["ENG", "OTHER", "ROM"]])
    assert switch_rate(df) == pytest.approx(50.0)


def test_switch_rate_empty_and_all_other():
    assert switch_rate(empty_df()) == 0.0
    assert switch_rate(make_df([["OTHER", "OTHER"]])) == 0.0


def test_summary():
    df = make_df(
        [
            ["ENG", "ROM", "ENG"],
            ["ENG", "ENG"],
            ["URD"],
        ]
    )
    result = summary(df)

    assert result["distribution"] == {
        "SND": 0,
        "URD": 1,
        "ROM": 1,
        "ENG": 4,
        "OTHER": 0,
    }
    assert result["intra_sentential"] == 2
    assert result["inter_sentential"] == 1
    assert result["switch_rate"] == pytest.approx(2 / 6 * 100)
    assert result["direction_matrix"]["ENG"]["ROM"] == 1
    assert result["direction_matrix"]["ROM"]["ENG"] == 1
    assert result["direction_matrix"]["ENG"]["ENG"] == 0


def test_summary_empty():
    result = summary(empty_df())
    assert result["distribution"] == {
        "SND": 0,
        "URD": 0,
        "ROM": 0,
        "ENG": 0,
        "OTHER": 0,
    }
    assert result["switch_rate"] == 0.0
    assert result["intra_sentential"] == 0
    assert result["inter_sentential"] == 0
    assert all(
        count == 0
        for targets in result["direction_matrix"].values()
        for count in targets.values()
    )


def test_functions_use_final_lang():
    df = make_df([["ENG", "ROM"]])
    df["auto_lang"] = "SND"
    assert switch_rate(df) == pytest.approx(50.0)
    assert summary(df)["distribution"]["ENG"] == 1
    assert summary(df)["direction_matrix"]["ENG"]["ROM"] == 1
