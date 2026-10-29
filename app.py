"""CodeMix Corpus Lab — Streamlit entrypoint."""

import html
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.langid import tag_corpus

SAMPLE_PATH = Path(__file__).parent / "data" / "sample" / "sample_corpus.txt"

PAGES = [
    "Input",
    "Tagging",
    "Switching",
    "Corpus Tools",
    "Correct & Evaluate",
    "Export",
]

LANG_COLORS = {
    "SND": "#FFD6A5",
    "URD": "#B7E4C7",
    "ROM": "#FDFFB6",
    "ENG": "#A2D2FF",
    "OTHER": "#DEE2E6",
}
LANG_LABELS = {
    "SND": "Sindhi",
    "URD": "Urdu",
    "ROM": "Roman Urdu",
    "ENG": "English",
    "OTHER": "Other",
}
_ARABIC_SCRIPT = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]"
)


@st.cache_data(show_spinner=False)
def build_token_df(text: str) -> pd.DataFrame:
    return tag_corpus(text)


def _is_rtl(token: str) -> bool:
    return bool(_ARABIC_SCRIPT.search(token))


def render_input() -> None:
    st.title("Input")
    st.caption("Paste text, upload a .txt/.csv file, or load the sample corpus.")

    if st.button("Load sample corpus"):
        if SAMPLE_PATH.is_file():
            st.session_state["raw_text"] = SAMPLE_PATH.read_text(encoding="utf-8")
        else:
            st.error(f"Sample corpus not found at {SAMPLE_PATH}")

    if st.session_state.get("raw_text"):
        st.text_area("Corpus", st.session_state["raw_text"], height=300)


def render_tagging() -> None:
    st.title("Tagging")

    text = st.session_state.get("raw_text", "")
    if not text.strip():
        st.info("No corpus loaded yet. Go to Input to paste text or load the sample.")
        return

    df = build_token_df(text)
    st.session_state["tokens_df"] = df

    legend = "".join(
        f'<span style="background:{color};color:#212529;padding:2px 8px;'
        'border-radius:6px;margin-right:6px;font-size:0.85rem;">'
        f"{code} — {LANG_LABELS[code]}</span>"
        for code, color in LANG_COLORS.items()
    )
    st.markdown(f'<div style="margin-bottom:0.75rem;">{legend}</div>', unsafe_allow_html=True)

    for _, group in df.groupby("sent_id", sort=True):
        spans = []
        for row in group.itertuples(index=False):
            rtl = ";direction:rtl;unicode-bidi:isolate" if _is_rtl(row.token) else ""
            color = LANG_COLORS.get(row.final_lang, LANG_COLORS["OTHER"])
            spans.append(
                f'<span title="{row.final_lang}" style="background:{color};'
                "color:#212529;padding:1px 5px;border-radius:4px;margin:0 2px;"
                f'display:inline-block{rtl};">{html.escape(row.token)}</span>'
            )
        st.markdown(
            f'<div style="line-height:2;margin-bottom:0.5rem;">{"".join(spans)}</div>',
            unsafe_allow_html=True,
        )

    st.subheader("Language distribution")
    counts = df["final_lang"].value_counts().reindex(LANG_COLORS, fill_value=0)
    chart_df = counts.rename_axis("Language").reset_index(name="Tokens")
    fig = px.bar(
        chart_df,
        x="Language",
        y="Tokens",
        color="Language",
        color_discrete_map=LANG_COLORS,
        text="Tokens",
    )
    fig.update_layout(
        showlegend=False,
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, width="stretch")


def render_switching() -> None:
    st.title("Switching")
    st.info("Placeholder: switch rate, switch types, and direction heatmap.")


def render_corpus_tools() -> None:
    st.title("Corpus Tools")
    st.info("Placeholder: Frequency, Concordance, N-grams, and Collocations.")


def render_correct_evaluate() -> None:
    st.title("Correct & Evaluate")
    st.info("Placeholder: editable tag table and accuracy report.")


def render_export() -> None:
    st.title("Export")
    st.info("Placeholder: download annotated CSV and JSON.")


RENDERERS = {
    "Input": render_input,
    "Tagging": render_tagging,
    "Switching": render_switching,
    "Corpus Tools": render_corpus_tools,
    "Correct & Evaluate": render_correct_evaluate,
    "Export": render_export,
}


def main() -> None:
    st.set_page_config(page_title="CodeMix Corpus Lab", layout="wide")
    st.sidebar.title("CodeMix Corpus Lab")
    page = st.sidebar.radio("Pages", PAGES)
    RENDERERS[page]()


if __name__ == "__main__":
    main()
