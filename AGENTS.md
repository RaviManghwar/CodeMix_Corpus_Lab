# AGENTS.md

## Project state
- Partially scaffolded: `app.py`, `src/`, and `tests/` exist, but only `tokenize.py` and `langid.py` are implemented. Other modules/pages are placeholders. There is no `requirements.txt`, `README.md`, or git repo yet.
- `SPEC (2).md` is the authoritative technical design; `PRD (1).md` defines scope/goals. Both filenames contain spaces and parentheses — quote them in shell.
- Do not build V1 non-goals: ML training/fine-tuning, DB/accounts/cloud storage, audio, or languages beyond SND/URD/ROM/ENG.

## Stack and commands
- Python 3.11+, Streamlit, pandas, Plotly, NLTK, pytest. No database; state lives in `st.session_state` and exported files.
- No `requirements.txt` yet. Verified checks:
  - `python3 -m pytest` (single test: `python3 -m pytest tests/test_langid.py::test_case -q`)
  - `streamlit run app.py`
- The `pytest` console script is not on PATH here; use `python3 -m pytest`.
- Pin dependency versions in `requirements.txt` once created; Streamlit Community Cloud deployment depends on it.

## Expected layout (SPEC §3)
- `app.py` — single Streamlit entrypoint with 6 sidebar pages: Input, Tagging, Switching, Corpus Tools, Correct & Evaluate, Export.
- `src/` — `tokenize.py`, `langid.py`, `switching.py`, `corpus_tools.py`, `evaluate.py`, `export.py`.
- `data/lexicons/` — `english.txt`, `roman_urdu.txt`, `urdu.txt`, `sindhi.txt`, plus the `ambiguous.txt` fixture.
- `data/sample/sample_corpus.txt`.
- `tests/` — one test file per module.

## Data gotchas
- Lexicons are UTF-8, one word per line, lowercase; SND/URD in native script, ROM/ENG in Latin.
- Lexicons live only in `data/lexicons/`; do not fork duplicate copies at the repo root.
- `data/lexicons/ambiguous.txt` is a fixture for words present in **both** `english.txt` and `roman_urdu.txt` (`do hi in is the to us`), not a lexicon. Resolve via nearest non-`OTHER` neighbour tag, else fall back to `ENG`.
- `sample_corpus.txt` is the demo corpus: one sentence per line, mixing Roman Urdu, Urdu script, and Sindhi script. It is not a gold/annotated set.
- Tags are exactly `SND`, `URD`, `ROM`, `ENG`, `OTHER`.
- Sindhi vs Urdu script is ambiguous: check Sindhi-only letters first (ڄ ڃ ڇ ڏ ڊ ڍ ڌ ڙ ڻ ٺ ٽ ٿ ڀ ڦ ڪ ڳ ڱ ھ), then the Sindhi lexicon, then default to `URD` (SPEC §5.2).

## Modeling conventions
- Token DataFrame columns/order: `sent_id, tok_id, token, norm, auto_lang, manual_lang, final_lang` (SPEC §4).
- Tagging order: number/punctuation/emoji/URL/mention/hashtag → `OTHER`; Arabic script → SND/URD; Latin lexicon → ROM/ENG, unknown Latin → `OTHER`.
- `norm` lowercases Latin tokens and collapses runs of 3+ identical Latin letters; lexicon lookups use lowercase without that collapse.
- `switching.py` ignores `OTHER` when finding switch points.
- Use `final_lang` (manual if set, else auto) for corpus tools, visualisation, switching stats, and export. Compare `auto_lang` vs `manual_lang` only in `evaluate.py`.
- The Tagging page caches `tag_corpus(text)` with `st.cache_data` and stores the token DataFrame in `st.session_state["tokens_df"]`; later pages should read that key.
- Preserve Unicode with NFC normalisation and strip zero-width characters; render Sindhi/Urdu RTL.
- Required test coverage (SPEC §8): pure SND, URD, ROM, ENG; mixed sentence; emoji/URL/number; empty input.
