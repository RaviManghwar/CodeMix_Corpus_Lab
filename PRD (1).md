# PRD: CodeMix Corpus Lab

**Owner:** Ravi Manghwar
**Status:** Draft v1
**Target V1:** 1-2 weeks
**Stack:** Python, Streamlit, pandas, NLTK, Plotly (built with opencode in VS Code)

---

## 1. Problem

Code-switching between Sindhi, Urdu, Roman Urdu and English is everywhere in Pakistani digital communication (WhatsApp, social media, classrooms), but it is hard to study computationally:

- Standard corpus tools (e.g. AntConc) have no language tagging, so mixed-language text cannot be analysed at word level.
- Sindhi and Roman Urdu are low-resource: few lexicons, no ready-made taggers.
- Researchers must annotate manually in spreadsheets, which is slow and not reproducible.

## 2. Vision

A lightweight, open, browser-based lab where a linguist can paste or upload code-mixed text and, within minutes, get word-level language tags, code-switching statistics, and standard corpus-linguistic outputs, with the ability to correct tags manually and measure accuracy.

## 3. Target Users

| User | Need |
|---|---|
| Linguistics students and researchers (primary) | Quickly analyse code-mixed data without writing code |
| Sociolinguists studying South Asian multilingualism | Switch-rate and pattern statistics for papers |
| Supervisors / reviewers (secondary) | A live demo that shows the researcher's skills and a reproducible method |

## 4. Goals

1. Tag words in mixed Sindhi / Urdu / Roman Urdu / English text with usable accuracy.
2. Detect and quantify code-switching (inter- vs intra-sentential, switch rate).
3. Provide core corpus tools: frequency, concordance (KWIC), N-grams, collocations.
4. Let the user correct tags and see accuracy against the automatic output.
5. Export annotated data for further research (CSV/JSON).
6. Serve as a portfolio piece that supports Master's applications.

## 5. Non-Goals (V1)

- No machine-learning model training or fine-tuning
- No user accounts, database or cloud storage
- No audio or speech processing
- No support for languages beyond the four listed

## 6. User Stories

1. As a researcher, I paste mixed text and see each word tagged by language so I can inspect the mix.
2. As a researcher, I upload a .txt/.csv file of sentences to analyse a whole dataset.
3. As a sociolinguist, I see the switch rate and switch types so I can report them.
4. As a corpus linguist, I run concordance, N-grams and collocations on tagged text.
5. As an annotator, I fix wrong tags in an editable table and see accuracy of the auto-tagger.
6. As a researcher, I export annotated data to CSV/JSON for use in R, Python or a paper.

## 7. Functional Requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-1 | Accept pasted text and .txt/.csv upload (one sentence per line) | Must |
| FR-2 | Unicode-safe tokenization | Must |
| FR-3 | Word-level tags: SND, URD, ROM, ENG, OTHER | Must |
| FR-4 | Switch-point detection, inter/intra-sentential classification, switch rate | Must |
| FR-5 | Frequency list, KWIC concordance, N-grams, collocations (PMI) | Must |
| FR-6 | Language distribution chart and per-sentence switch visualisation | Should |
| FR-7 | Editable tag table and accuracy report (precision, recall, F1, accuracy) | Must |
| FR-8 | Export annotated CSV and JSON | Must |
| FR-9 | Built-in sample corpus for instant demo | Should |
| FR-10 | Custom lexicon upload by the user | Could |

## 8. Non-Functional Requirements

- Runs locally with one command (`streamlit run app.py`)
- Handles up to about 10,000 tokens without lag
- RTL scripts (Sindhi, Urdu) display correctly
- Clean, documented, tested code (pytest) with a clear README

## 9. Success Metrics

| Metric | Target for V1 |
|---|---|
| Tagging accuracy on own annotated set (200-500 sentences) | 85% or higher token-level |
| Time from paste to results | Under 10 seconds |
| Deployed live demo | Streamlit Community Cloud link works |
| Research note | 2-page note in the README with a real finding |

## 10. Research Value (why it matters)

- Addresses a genuine gap: tools for Sindhi and Roman Urdu code-mixing.
- Produces an annotated mini-dataset and open lexicons, which are reusable contributions.
- Opens concrete research questions, e.g. *where and why do Pakistani university students code-switch in WhatsApp text?*

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Roman Urdu spelling is highly variable | Normalisation rules plus a growing lexicon; report accuracy honestly |
| Sindhi vs Urdu script overlap | Use Sindhi-specific letters and a lexicon; show confidence |
| Small lexicons cause low accuracy | Build lexicons from your own annotated data; iterate |
| Scope creep | Freeze V1 to the Must items above |
| Privacy of real chat data | Anonymise all data; do not publish identifiable messages |

## 12. Milestones

| Day | Deliverable |
|---|---|
| 1 | Setup, SPEC and PRD finalised, project skeleton |
| 2-3 | Tokenizer, script detection, Latin-script tagging |
| 4-5 | Switching analysis and corpus tools |
| 6 | Manual correction and accuracy report |
| 7 | Export, sample corpus, README |
| 8-10 | Annotated dataset (200-500 sentences), lexicon refinement |
| 11-12 | Research note, deployment, polish |
| 13-14 | Buffer, demo recording, link in emails and CV |

## 13. Open Questions

1. Which data source for the annotated set (WhatsApp, Twitter/X, classroom data)?
2. Should Roman Urdu and Roman Sindhi be separated in V2?
3. Which research question will the 2-page note answer?
