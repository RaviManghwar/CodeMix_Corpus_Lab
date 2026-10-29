"""Tests for src.evaluate (SPEC §5.5)."""

import pandas as pd
import pytest

from src.langid import COLUMNS, LANGS
from src.evaluate import ERROR_COLUMNS, accuracy_report, list_errors


def make_df(rows: list[tuple[str, str, str | None]]) -> pd.DataFrame:
    data = []
    for tok_id, (token, auto_lang, manual_lang) in enumerate(rows):
        data.append(
            {
                "sent_id": 0,
                "tok_id": tok_id,
                "token": token,
                "norm": token.lower(),
                "auto_lang": auto_lang,
                "manual_lang": manual_lang,
                "final_lang": auto_lang,
            }
        )
    return pd.DataFrame(data, columns=COLUMNS)


def empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def test_perfect_match():
    df = make_df(
        [
            ("a", "ENG", "ENG"),
            ("b", "ROM", "ROM"),
            ("c", "OTHER", "OTHER"),
        ]
    )
    report = accuracy_report(df)

    assert report["accuracy"] == 1.0
    assert report["annotated_tokens"] == 3
    assert report["total_tokens"] == 3
    assert report["annotated_pct"] == 100.0

    per = report["per_language"].set_index("lang")
    assert per.loc["ENG", "precision"] == 1.0
    assert per.loc["ENG", "recall"] == 1.0
    assert per.loc["ENG", "f1"] == 1.0
    assert per.loc["ENG", "support"] == 1
    assert per.loc["OTHER", "support"] == 1

    confusion = report["confusion_matrix"]
    assert confusion.loc["ENG", "ENG"] == 1
    assert confusion.loc["ROM", "ROM"] == 1
    assert confusion.loc["OTHER", "OTHER"] == 1
    assert list(confusion.index) == list(LANGS)
    assert list(confusion.columns) == list(LANGS)
    assert list_errors(df).empty


def test_known_errors_and_metrics():
    df = make_df(
        [
            ("a", "ENG", "ENG"),
            ("b", "ROM", "ENG"),
            ("c", "ENG", "ROM"),
            ("d", "URD", "URD"),
        ]
    )
    report = accuracy_report(df)
    per = report["per_language"].set_index("lang")
    confusion = report["confusion_matrix"]

    assert report["accuracy"] == pytest.approx(0.5)
    assert report["annotated_tokens"] == 4
    assert report["annotated_pct"] == 100.0

    assert per.loc["ENG", "support"] == 2
    assert per.loc["ENG", "precision"] == pytest.approx(0.5)
    assert per.loc["ENG", "recall"] == pytest.approx(0.5)
    assert per.loc["ENG", "f1"] == pytest.approx(0.5)
    assert per.loc["ROM", "support"] == 1
    assert per.loc["ROM", "precision"] == 0.0
    assert per.loc["ROM", "recall"] == 0.0
    assert per.loc["ROM", "f1"] == 0.0
    assert per.loc["URD", "f1"] == 1.0

    assert confusion.loc["ENG", "ENG"] == 1
    assert confusion.loc["ENG", "ROM"] == 1
    assert confusion.loc["ROM", "ENG"] == 1
    assert confusion.loc["URD", "URD"] == 1


def test_list_errors_includes_token_and_tags():
    df = make_df(
        [
            ("a", "ENG", "ENG"),
            ("b", "ROM", "ENG"),
            ("c", "ENG", "ROM"),
            ("d", "URD", "URD"),
        ]
    )
    errors = list_errors(df)

    assert list(errors.columns) == ERROR_COLUMNS
    assert errors["token"].tolist() == ["b", "c"]
    assert errors["auto_lang"].tolist() == ["ROM", "ENG"]
    assert errors["gold_lang"].tolist() == ["ENG", "ROM"]


def test_no_annotations():
    df = make_df(
        [
            ("a", "ENG", ""),
            ("b", "ROM", None),
            ("c", "ENG", "   "),
            ("d", "URD", "BOGUS"),
        ]
    )
    report = accuracy_report(df)

    assert report["accuracy"] == 0.0
    assert report["annotated_tokens"] == 0
    assert report["total_tokens"] == 4
    assert report["annotated_pct"] == 0.0
    assert report["per_language"]["support"].tolist() == [0, 0, 0, 0, 0]
    assert (report["confusion_matrix"].to_numpy() == 0).all()
    assert list_errors(df).empty
    assert list(list_errors(df).columns) == ERROR_COLUMNS


def test_blank_manual_lang_ignored():
    df = make_df(
        [
            ("a", "ENG", ""),
            ("b", "ROM", "ROM"),
            ("c", "ENG", "  "),
        ]
    )
    report = accuracy_report(df)

    assert report["annotated_tokens"] == 1
    assert report["annotated_pct"] == pytest.approx(1 / 3 * 100)
    assert report["accuracy"] == 1.0
    assert report["per_language"].set_index("lang").loc["ROM", "support"] == 1


def test_custom_gold_column():
    df = make_df(
        [
            ("a", "ENG", ""),
            ("b", "ROM", ""),
        ]
    )
    df["gold_lang"] = ["ENG", "ENG"]

    report = accuracy_report(df, gold_col="gold_lang")
    errors = list_errors(df, gold_col="gold_lang")

    assert report["accuracy"] == 0.5
    assert report["annotated_tokens"] == 2
    assert errors["token"].tolist() == ["b"]
    assert errors["gold_lang"].tolist() == ["ENG"]


def test_missing_gold_column_raises():
    with pytest.raises(ValueError):
        accuracy_report(make_df([("a", "ENG", "ENG")]), gold_col="missing")


def test_empty_dataframe():
    report = accuracy_report(empty_df())

    assert report["accuracy"] == 0.0
    assert report["annotated_tokens"] == 0
    assert report["total_tokens"] == 0
    assert report["annotated_pct"] == 0.0
    assert len(report["per_language"]) == len(LANGS)
    assert report["confusion_matrix"].shape == (len(LANGS), len(LANGS))
    assert (report["confusion_matrix"].to_numpy() == 0).all()
    assert list_errors(empty_df()).empty
