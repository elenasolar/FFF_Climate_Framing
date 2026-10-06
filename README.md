# Crisis and Framing of the Fridays for Future Movement on Instagram — An Exploratory Analysis

*Course project for* Current Debates on Political Inequality *(Summer Semester 2026)*

## Overview

This project studies how Fridays for Future (FFF) Germany's climate communication on Instagram evolved between 2018 and 2025, across five compounding crises (COVID-19, the Ukraine War/energy crisis, the cost-of-living crisis, the Gaza War, and the 2024/25 German government crisis and snap election). Each post is scored along four framing dimensions by an LLM, then read against a hand-curated periodization and key-event calendar to test whether — and in which direction — framing shifted under pressure, and whether climate framing extended into broader inequality and justice discourse. The hypotheses are stated in [`Project_Proposal.md`](Project_Proposal.md) §7.

## Data

- **Source**: Instagram, pulled via Meta's Content Library (academic research access), filtered to posts mentioning `fridaysforfuture` / `fridays for future`.
- **Sample**: 3,240 posts published by the five official Germany-based FFF accounts (the national account plus Berlin, Köln, München and Hamburg) between December 2018 and December 2025 — as opposed to third-party posts merely mentioning FFF. The Austrian chapters (827 posts) are excluded because the crisis periodization and event calendar are German; 2026 is excluded because the pull's January–July slice holds only ~50 posts; 22 posts with no caption left after stripping hashtags/mentions are dropped as a data-quality rule. See [`notes/data_collection.md`](notes/data_collection.md) for the sampling rationale and a known data-access limitation (Meta's follower/verification thresholds plausibly exclude smaller volunteer-run chapter accounts).
- Raw post text lives in `data/raw/` and is not intended for publication (Instagram content, not ours to redistribute); only aggregated scores are meant to be shared.

## Method

### Four framing axes

Every post is rated 0–4 on each concept within four axes (definitions in [`concepts/`](concepts/)):

| Axis | Concepts | Framework |
|---|---|---|
| **A — DPM** | diagnostic, prognostic, motivational | Benford & Snow's (1988) core framing functions |
| **B — DPM Subcategories** | blame_attribution, problem_identification, solutions, tactics, rationale, call_to_action | Finer-grained split of each DPM function |
| **C — Master Frame** | climate_change, climate_justice | della Porta & Parks — ecological/technocratic vs. justice-and-responsibility framing |
| **D — Inequality Arena** | oben_unten (top–bottom), innen_aussen (inside–outside), wir_sie (us–them), heute_morgen (today–tomorrow) | Mau, Lux & Westheuser's (2023) societal inequality arenas |

### Crisis-window periodization & key events

A six-window periodization (baseline, COVID-19, Ukraine War/energy crisis, cost-of-living crisis, Gaza War, and the German government crisis from the failed Vertrauensfrage in December 2024) and a curated calendar of ~100 FFF-relevant events (global strikes, elections, COP conferences, campaigns) are defined once in [`src/crisis_windows.py`](src/crisis_windows.py) and [`src/key_events.py`](src/key_events.py), and reused across every plot, the regressions and the interactive dashboard. Windows partition the timeline; end dates are inclusive of the whole day.

### LLM-based scoring

Each axis's concepts are scored by an LLM (Qwen3-4B-Instruct, served locally via vLLM) against explicit definitions and cue examples (`concepts/*.json`), using [`scripts/score_axis_llm_ddr.py`](scripts/score_axis_llm_ddr.py). The full-corpus runs use a single, fixed concept ordering; a counterbalanced pilot on a stratified ~1,000-post sample (`01_robustness_checks.ipynb`) found order effects that were statistically detectable but substantively negligible (< 0.1 points on the 0–4 scale), which is what justifies that choice.

### Validation & robustness

- `01_robustness_checks.ipynb` — stratified pilot sample and ordering-effect diagnostics.
- `04_llm_agreement_validation.ipynb` — checks that the hierarchical relationship between Axis A (DPM) and Axis B (subcategories) holds empirically, even though the LLM is never told about that hierarchy.
- `06_crisis_window_regression.ipynb` — tests the hypotheses H1–H4 with per-post regressions on the crisis windows, controlling for caption length and account (both confounds shown in the notebook's pre-checks), with HAC standard errors.
- Human-vs-LLM validation (a hand-coded sample) is planned but not yet integrated.

## Repository Structure

```
notebooks/    Analysis pipeline, run roughly in numeric order (00 → 05)
src/          Shared code: crisis windows, key events, plot styling
scripts/      LLM scoring, topic classification, dashboard build
concepts/     Per-axis concept definitions (also used as LLM prompts)
data/         raw/ (not published), processed/, llm_scores/
figures/      Publication-ready static plots (PDF only)
docs/         Self-contained interactive dashboard (see below)
notes/        Working methodology notes (data collection, environment)
paper/        Appendix snippets
```

## Exploring the Results

- **Interactive dashboard** — `docs/index.html`, published via GitHub Pages once enabled for this repo (Settings → Pages → branch `main`, folder `/docs`). Lets you switch axes, compare two axes on dual y-axes, change the time-aggregation level (weekly through yearly), and toggle crisis-window shading and event markers. Regenerate it after any change to the scores or event/window definitions with:
  ```
  python scripts/build_dashboard.py
  ```
- **Static figures** — `figures/FFF/`, generated by `05_llm_scoring_analysis.ipynb` (per-axis intensity/composition plots, cross-axis overlays, the full concept correlation matrix).
- **Notebooks** — read top-to-bottom for the full analysis narrative and working notes; `05_llm_scoring_analysis.ipynb`'s "Key Takeaways" section summarizes cross-axis patterns found so far.

## Reproducing the Analysis

```
python -m venv .venv
.venv/Scripts/activate     # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt   # if present, otherwise see notes/environment_constraints.md
```

Run the notebooks in order; `05_llm_scoring_analysis.ipynb`'s `## Config` cell switches which axis/aggregation level is active and needs to be re-run per axis.

## Status & Known Limitations

- Caption length strongly predicts the LLM intensity scores and changed over the period; every regression controls for it, and the descriptive intensity plots should be read with that in mind.
- Human-vs-LLM validation sample is planned, not yet scored.
- The sampling frame is bounded by Meta Content Library's account-eligibility rules, which plausibly and systematically under-represent smaller, lower-follower regional chapters.
