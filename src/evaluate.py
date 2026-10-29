"""Accuracy evaluation against manual or gold tags (SPEC §5.5)."""

from __future__ import annotations

import pandas as pd

from src.langid import LANGS

ERROR_COLUMNS = ["sent_id", "tok_id", "token", "norm", "auto_lang", "gold_lang"]
PER_LANGUAGE_COLUMNS = ["lang", "precision", "recall", "f1", "support"]


def _gold_tags(df: pd.DataFrame, gold_col: str) -> tuple[pd.Series, pd.Series]:
    if gold_col not in df.columns:
        raise ValueError(f"Gold column '{gold_col}' not found")
    gold = df[gold_col].fillna("").astype(str).str.strip()
    return gold, gold.isin(LANGS)


def _auto_tags(df: pd.DataFrame) -> pd.Series:
    return df["auto_lang"].fillna("").astype(str).str.strip()


def accuracy_report(
    df: pd.DataFrame, gold_col: str = "manual_lang"
) -> dict[str, object]:
    """Compare ``auto_lang`` with the gold tags on annotated rows.

    Returns overall accuracy (0–1), annotated/total token counts, a
    per-language precision/recall/F1 table, and a gold-by-predicted confusion
    matrix. Blank or invalid gold tags are ignored.
    """
    total = len(df)
    gold, annotated = _gold_tags(df, gold_col)
    auto = _auto_tags(df)
    gold = gold[annotated]
    auto = auto[annotated]
    count = len(gold)

    correct = int((gold == auto).sum())
    accuracy = correct / count if count else 0.0

    per_language = []
    for lang in LANGS:
        true_positive = int(((gold == lang) & (auto == lang)).sum())
        predicted = int((auto == lang).sum())
        actual = int((gold == lang).sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_language.append(
            {
                "lang": lang,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": actual,
            }
        )

    confusion = pd.crosstab(gold, auto)
    confusion = confusion.reindex(
        index=list(LANGS), columns=list(LANGS), fill_value=0
    ).fillna(0).astype(int)
    confusion.index.name = "gold"
    confusion.columns.name = "auto"

    return {
        "accuracy": accuracy,
        "annotated_tokens": count,
        "total_tokens": total,
        "annotated_pct": (count / total * 100) if total else 0.0,
        "per_language": pd.DataFrame(
            per_language, columns=PER_LANGUAGE_COLUMNS
        ),
        "confusion_matrix": confusion,
    }


def list_errors(
    df: pd.DataFrame, gold_col: str = "manual_lang"
) -> pd.DataFrame:
    """Return annotated rows where ``auto_lang`` differs from the gold tag."""
    if df.empty:
        return pd.DataFrame(columns=ERROR_COLUMNS)

    gold, annotated = _gold_tags(df, gold_col)
    auto = _auto_tags(df)
    mismatched = annotated & (gold != auto)
    if not mismatched.any():
        return pd.DataFrame(columns=ERROR_COLUMNS)

    result = df.loc[
        mismatched,
        ["sent_id", "tok_id", "token", "norm", "auto_lang"],
    ].copy()
    result["gold_lang"] = gold[mismatched].to_numpy()
    return result[ERROR_COLUMNS].reset_index(drop=True)
