"""CSV and JSON export for annotated token DataFrames (SPEC §5.6)."""

from __future__ import annotations

import json

import pandas as pd

from src.switching import find_switches, summary
from src.tokenize import split_sentences

EXPORT_COLUMNS = [
    "sent_id",
    "tok_id",
    "token",
    "norm",
    "auto_lang",
    "manual_lang",
    "final_lang",
]
def to_csv(df: pd.DataFrame) -> bytes:
    """Return annotated CSV bytes (UTF-8 with BOM for Excel)."""
    export = df.reindex(columns=EXPORT_COLUMNS).copy()
    export["manual_lang"] = export["manual_lang"].fillna("")
    return export.to_csv(index=False).encode("utf-8-sig")


def to_json(df: pd.DataFrame, text: str | None = None) -> bytes:
    """Return sentence-nested JSON bytes with tokens, switches, and summary."""
    sentence_texts = split_sentences(text) if text else []
    sentences = []
    for sent_id, group in df.groupby("sent_id", sort=True):
        sent_id = int(sent_id)
        tokens = []
        for row in group.itertuples(index=False):
            tokens.append(
                {
                    "tok_id": int(row.tok_id),
                    "token": row.token,
                    "norm": row.norm,
                    "auto_lang": row.auto_lang,
                    "manual_lang": (
                        "" if pd.isna(row.manual_lang) else row.manual_lang
                    ),
                    "final_lang": row.final_lang,
                }
            )
        original = (
            sentence_texts[sent_id]
            if sent_id < len(sentence_texts)
            else " ".join(token["token"] for token in tokens)
        )
        sentences.append(
            {
                "sent_id": sent_id,
                "text": original,
                "switch_count": len(find_switches(group["final_lang"].tolist())),
                "tokens": tokens,
            }
        )

    payload = {"summary": summary(df), "sentences": sentences}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
