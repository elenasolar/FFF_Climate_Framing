# Project Review: Research Design, Analysis, Interpretation

*2026-09-22. Scope: everything inside `FFF_Climate_Framing/` except `references/` (external
ddr-library copy, gitignored) and `.venv/`. Read: all eight notebooks (source cells), `scripts/`,
`src/`, `concepts/`, `notes/`, `README.md`, `Project_Update.md`. Every number below comes from
checks run on the current data and score files during this review, not from memory of the
notebooks' saved outputs.*

---

## 1. Bottom line

The project is well above the "descriptive course project" bar in infrastructure, measurement
care, and transparency. Its weakness is inference, not effort: as the analysis stands, the
headline result -- *frame intensity jumped at each crisis onset and then declined* -- is to a
large degree an artifact of two things not yet controlled: **how long the captions are** and
**who is posting**. Once both are controlled, the DPM story mostly disappears, while the
justice / inequality-arena story survives. That is a better and more interesting paper than the
current one, but it is a different paper, and the write-up has to be built on the controlled
results.

Three things must happen before this is solid at PhD level:

1. Control for caption length and corpus composition (or switch to share-based outcomes).
2. Validate the LLM scores against human coding, with more than 50 posts.
3. Put the theory -> hypotheses -> test chain on paper, so that the regressions in
   `06_crisis_window_regression.ipynb` answer stated hypotheses instead of describing periods.

Everything else is refinement.

---

## 2. Strengths (keep these)

- **Theory-anchored measurement, not ad hoc coding.** Four axes, each traceable to a named
  framework (Benford & Snow DPM; Mendelsohn et al.'s subcategories; della Porta & Parks master
  frames; Mau/Lux/Westheuser arenas), each with an explicit German definition *and* cue
  examples in `concepts/*.json`. This is a real codebook, reusable and inspectable.
- **Validation work already exceeds most course projects.** The ordering pilot
  (`01_robustness_checks.ipynb`: ICC(2,1) .88-.95, Friedman significant but gaps < 0.1 points,
  and the correct "statistically real, substantively negligible" call) and the hierarchical
  convergent/discriminant check (`04_llm_agreement_validation.ipynb`, with the strict criterion
  that caught `rationale` and `tactics`) are exactly the kind of checks reviewers ask for.
- **Full corpus, not a sample.** 4,138 posts over eight years, scored on every axis, with
  per-post scores retained -- enough for post-level inference.
- **Single source of truth for design choices.** Windows (`src/crisis_windows.py`), events
  (`src/key_events.py`), and styling are defined once and imported everywhere; the dashboard
  and static figures cannot drift apart.
- **Per-concept analysis instead of axis means** -- the right call, and it already paid off
  (`rationale` behaving as diagnostic; `heute_morgen` moving against the other arenas).
- **Honest documentation of data limits** (`notes/data_collection.md` on Meta Content Library
  coverage, hashtag-hijacking accounts, the unresolved Germany/Austria scope).
- **Moving beyond description was the right instinct**: the segmented regression with HAC
  errors and FDR correction is the appropriate design family for "did the trajectory break
  at known dates".

---

## 3. Critical issues

Ordered by how much they threaten the conclusions.

### 3.1 Caption length drives the intensity scores  (severity: high)

**Evidence**

- Correlation of log(clean word count) with each concept's score: 0.26-0.60. Highest:
  `rationale` .60, `problem_identification` .58, `diagnostic` .57, `prognostic` .55,
  `heute_morgen` .49, `solutions` .49, `climate_change` .47. Lowest: `call_to_action` .26,
  `wir_sie` .26, `innen_aussen` .28, `oben_unten` .31.
- Variance explained by log(words) alone: 17-34% (`motivational` .19, `diagnostic` .34).
  Variance explained by the five crisis windows alone: 1-4%.
- Median clean word count by year: 2018 13, 2019 41, **2020 68**, 2021 58, 2022 50, 2023 44,
  2024 44, 2025 62, 2026 42. The baseline -> covid jump in caption length has the same shape as
  the baseline -> covid jump in `diagnostic`/`prognostic`.
- Level shift vs. baseline, before -> after adding log(words) (OLS, no other controls):

  | concept | window | before | after |
  |---|---|---|---|
  | diagnostic | covid | +0.49 | +0.14 |
  | diagnostic | ukraine_energy | +0.44 | +0.23 |
  | diagnostic | gaza | +0.28 | +0.19 |
  | motivational | covid | +0.16 | **-0.05 (sign flips)** |
  | motivational | gaza | +0.21 | +0.16 |
  | climate_justice | ukraine_energy | +0.71 | +0.56 |
  | climate_justice | gaza | +0.42 | +0.36 |
  | oben_unten | gaza | +0.39 | +0.37 |
  | oben_unten | ukraine_energy | +0.34 | +0.28 |
  | heute_morgen | gaza | -0.28 | **-0.35 (more negative)** |

- The first principal component explains **43%** of the variance across all 15 concepts
  (second: 15%). A general "how much substantive text is there" factor sits under every axis
  and inflates every cross-axis correlation -- including the three "clusters" in
  `05_llm_scoring_analysis.ipynb`'s Key Takeaways.

**Why it matters.** An LLM asked to rate a caption 0-4 on fifteen functions is partly rating
"how much does this caption say". Longer captions give more opportunities for every function
to appear. Caption-length norms changed over the period (short 2019 strike calls; long 2020
lockdown-era statements), and that change is confounded with the windows.

**What to do** (one as primary specification, the rest as robustness):

- (a) Add `log(clean_word_count)` as a covariate to every model in 06 and report the
  before/after table. The table is a finding in itself.
- (b) For DPM and the subcategories, switch the outcome to **relative composition** (a
  concept's share of the post's total across its axis). `05`'s Relative Frame Composition
  plot already does this descriptively; using it inferentially removes the common length
  factor by construction, and "did the *mix* shift" is closer to the concept of a frame
  shift than raw intensity is.
- (c) Before the cross-axis correlation matrix, residualize on length (or partial out PC1)
  and check whether the three clusters survive. Report the cluster structure only if they do.
- (d) Report the length trend as a communication-style finding in its own right.

### 3.2 Who is posting changed more than how they frame  (severity: high)

**Evidence**

- Posts per account per year: the national account `fridaysforfuture.de` goes 278 (2019) ->
  12 (2025) -> 6 (2026). Hamburg dominates 2023-24 (196, 136). Berlin is 91 of 164 posts in
  2025 (55%). Vienna posts 296 in 2019 (24% of that year) and almost nothing after 2021.
- **Austrian accounts (`fridaysforfuture.at` + `fridaysforfuturevienna`) are 827 posts, 20% of
  the corpus**, and their share falls monotonically across exactly the windows being compared:
  baseline 29%, covid 20%, ukraine_energy 19%, cost_of_living 12%, gaza 7%. The periodization
  and the key events are German (Bundestag elections, Heizungsgesetz, German lockdown timing);
  they do not apply to Vienna the same way. `notes/data_collection.md` lists this as
  unresolved -- it has to be resolved before anything is interpreted.
- Content type: videos are 11% of 2019 posts, 28% of 2023, 53% of 2026 (n=51). A video
  caption is not the message.
- Posts flagged by the logistics heuristic (`has_logistics`): 27% (2019) -> 36% (2022-23) ->
  50% (2024) -> 55% (2026). A growing part of the late corpus is event announcements, which
  score high on `call_to_action`/`tactics` by construction.
- Volume: 1,213 posts (2019) -> 579 (2023) -> 164 (2025) -> 51 (2026 H1). Quarterly means for
  2025-26 rest on ~40 posts; the late "peaks" in the arena axis are largely noise.
- Adding account fixed effects on top of the length control changed the estimates very little
  (e.g. `diagnostic` covid 0.14 -> 0.13), so composition is less damaging than length -- but
  Austria is a *scope* problem, not only a control problem.

**What to do**

- Decide Germany-only for the main analysis (my recommendation, given German-defined windows
  and events). Keep Austria as a comparison case (see 4, item 12) or justify pooling explicitly.
- Add account fixed effects, `content_type`, and `has_logistics` to the 06 models.
- Put a posting-volume figure and an account-share-over-time figure in the report so the
  reader sees the composition change before seeing the framing change.
- Either end the analysis where quarterly n is still reasonable (e.g. end 2025) or aggregate
  the tail; do not let 40-post quarters drive the narrative.

### 3.3 The periodization does most of the causal work and cannot bear it  (severity: high, interpretation)

- **One movement, one timeline: window = time.** Any secular trend -- movement maturation,
  professionalization, decline, platform norms -- loads onto the window dummies. The design
  identifies *periods*, not *crisis effects*. Write accordingly: "in the period after the
  invasion", "coincides with", never "caused by" or "in response to".
- **Baseline as reference contains the founding ramp-up.** 2018 has 40 posts with a median of
  13 words; 2019 is explosive growth. Almost every negative slope coefficient in 06 means
  "grew less steeply than the ramp-up", not "declined during the crisis". Options: use covid
  as the reference; define a "mature baseline" (2019-03-15, first global strike, to 2020-02);
  or drop 2018.
- **The `gaza` window is 2.75 years and 602 posts** (2023-10 -> 2026-07). It contains the 2024
  EU election, the 2025 Bundestag election, the movement's international split over Gaza, and
  FFF's post-2023 decline. Labelling all of that "Gaza War" attributes the whole late period
  to one crisis. Rename it (e.g. "post-2023 / overlapping crises") or split it at 2025-01 or at
  the Bundestag election.
- **End dates are by fiat.** `cost_of_living` = calendar 2023; `ukraine_energy` ends
  2022-12-31. Start dates have citable anchors, end dates do not. Add anchors where possible
  and run a boundary sensitivity check (shift each boundary +/-3 months; the 06 code makes this
  cheap). If a result depends on where a fiat boundary sits, say so.
- **Boundary bug.** `resolve_windows()` turns `end="2020-02-29"` into midnight at the *start*
  of that day, so posts published later that day fall into a gap between windows. Six posts
  (five on 2020-02-29, one on 2022-12-31) currently belong to no window. Fix by making the end
  inclusive to end-of-day, or by defining windows half-open `[start, next_start)`. Related:
  `05`'s per-window summary uses `.loc[start:end]`, which double-counts a boundary day now
  that `cost_of_living` ends exactly at `gaza`'s start.

### 3.4 Measurement validity of the LLM scores  (severity: high for a PhD-level claim)

- **No human validation exists yet.** The planned 50 posts is too few: with 15 concepts and
  floor-heavy arenas (below), you would have a handful of non-zero cases per concept. Aim for
  150-200 posts, stratified by window *and* by LLM score (oversample non-zero arena posts),
  two coders, Krippendorff's alpha (ordinal) or ICC(2,1) per concept, plus agreement on
  presence (0 vs. >0). Report per concept and expect the arenas to be worst. If an arena
  validates poorly, demote it to exploratory rather than dropping it silently.
- **Model and parameters are under-documented.** Qwen3-4B-Instruct-2507 is a small model --
  fine for explicit cues, weaker for pragmatics (irony, implicit blame). Report model,
  quantization, temperature (not visible anywhere in this repo; it lives in the ddr package's
  defaults), `enable_thinking=false`, the exact rendered prompt (appendix), and run dates.
  Test-retest reliability (rerun ~200 posts, report agreement) is listed in
  `Project_Update.md` as "if time" -- promote it; it is cheap and reviewers will ask.
- **Scale behaviour.** Ceiling: `motivational` 54% at 4, `call_to_action` 54% at 4. Floor:
  `oben_unten` 83% zero, `innen_aussen` 86%, `wir_sie` 74%, `blame_attribution` 59%. Means of
  the arenas are means of mostly zeros; a "+0.4 shift" is really a change in the share of posts
  that mention the arena at all. Model the arenas as presence (score >= 1, or >= 2) with
  logit/probit and report predicted probabilities. Treat `motivational` as saturated: there is
  almost no headroom, so "motivational rose" is both hard to detect and easy to over-read.
- **Ordinal treated as interval** -- an accepted convention; state it, and run an
  ordered-logit robustness check for one headline concept.
- **Ordering.** The pilot justified a single fixed ordering for the full runs -- a good,
  well-documented decision. But README's method section said scores are "debiased by averaging
  across counterbalanced orderings"; that is only true of the pilot file. Corrected in this
  review pass. Also: `Project_Update.md` says the definitions were revised and all axes rerun.
  Was the ordering pilot (and the 04 hierarchy check) run on the *final* definitions? If not,
  state it or rerun the pilot.
- **Hierarchical misfit** (`rationale` correlates .80 with `diagnostic` vs. .41 with its parent
  `motivational`; `tactics` .65 with `motivational` vs. .51 with `prognostic`) is documented --
  good. Consequence: in this corpus `rationale` *is* diagnosis; do not narrate it as
  mobilization, and consider reporting axis B's parents by empirical rather than theoretical
  grouping.
- **Construct overlap.** `oben_unten` <-> `climate_justice` is already caveated.
  `heute_morgen` is the bigger problem: it is defined as generational climate framing,
  correlates .61 with `climate_change` and .49 with length, and 37% of posts score zero on it
  while 30% score zero on `climate_change`. It may mostly register "this post is about climate
  at all". Two checks: (i) among posts with `climate_change >= 2`, does `heute_morgen` still
  vary across windows? (ii) Its negative Gaza shift could be topic displacement (fewer
  climate posts) rather than reframing -- needs the topic control in 3.6.
- **Which text was scored?** `score_axis_llm_ddr.py` recommends `clean_text` (hashtags
  stripped). Confirm which column the full runs actually used and state it. Hashtags carry
  frame cues (`#klimagerechtigkeit`), so the choice changes what the scores mean.

### 3.5 The theory -> hypotheses -> results chain is not on paper  (severity: high for the rubric)

`Project_Proposal.md` states RQ1–RQ5 (correction: it *is* in the repository -- my file listing
was truncated), but no directional hypotheses. Nothing states what *should* happen under which
theory, so 06 currently tests 60 coefficients without a prior. Section 5 sketches a hypothesis
set that the existing data and code can test; the essential property is that each hypothesis
names a direction, names the coefficient that tests it, and at least one is written so the data
can refute it.

### 3.6 Modelling details in 06  (severity: medium)

- HAC standard errors on posts sorted by date treat rank order as time, but posts are
  irregularly spaced (bursts on strike days). Alternatives: cluster SEs by ISO week or month;
  or aggregate to weekly means and apply HAC to the regular series. Show that conclusions do
  not hinge on the choice.
- A linear slope inside a 2-year (covid) or 2.75-year (gaza) window is a strong assumption;
  the small-multiples plot shows non-monotone paths within windows. Report slope tests only
  for short windows, or overlay a within-window LOESS/spline to show the linear fit is
  adequate.
- With n ~ 4,100, 48 of 60 level shifts are "significant". Lead with effect sizes on the 0-4
  scale (and as % of the baseline mean) with CIs; add a minimum-meaningful-effect line (e.g.
  0.25 points) to the forest plots.
- **Add topic as covariate or mediator** (axis E labels or the BERTopic topic): is the framing
  shift a change *within* topics, or a change in *what* is posted about? This is the single
  most informative extension for "substantive interpretation beyond description".
- Fifteen separate models are defensible with FDR correction (done). A joint long-format
  model with concept x window interactions would let you test *differences between concepts*
  directly (e.g. "did `climate_justice` shift more than `climate_change`?"), which is what
  several of your Key Takeaways actually claim.

### 3.7 What the data cannot see  (severity: medium; must be stated)

- Captions only. Sharepics, videos, and stories carry much of FFF's messaging, and the
  video share rises from 11% to 53% across the period -- so the text is a shrinking part of
  the message, and that trend is correlated with the windows.
- One platform. Instagram's caption norms and hashtag culture shape what is measured; the
  object is "FFF's Instagram captions", not "FFF's framing".
- Meta Content Library coverage (documented -- good), deleted posts and renamed accounts
  (survivorship), and the 24 posts in `FFF_german.csv` that have no scores (4,138 in the corpus
  vs. 4,114 scored on A/C/D and 4,104 on B) -- presumably empty `clean_text` and parse
  failures; say so.

### 3.8 Reproducibility and data governance  (severity: medium; high before the repo goes public)

- `data/processed/FFF_german.csv` (raw captions, usernames, URLs) and several processed topic
  files are git-tracked; the `.gitignore` lines for `data/processed/` and `data/llm_scores/`
  are commented out; notebooks `00*`, `03`, `xx_test` have raw post text in saved outputs.
  Check Meta Content Library's data-sharing terms -- redistribution of raw content is
  typically not permitted -- and strip these (including history, via `git filter-repo`) before
  publishing.
- Notebooks 00/02/03 read from an sshfs mount with a local key file; note in README that they
  run on the lab server, not locally.
- Record LLM run metadata (model, parameters, prompt, date, parse-failure counts -- 10 on axis
  B) in a small `data/llm_scores/README.md` or run log.
- Keep `Project_Proposal.md` current as the design record (updated 2026-09-22 with scope,
  final windows, analysis strategy and hypotheses).

---

## 4. What to change to make this a solid PhD-level course project (prioritized)

**Must**

1. Resolve scope: Germany-only main analysis; Austria as a comparison case or an explicit
   justification for pooling.
2. Add controls to 06 -- `log(clean_word_count)`, account fixed effects, `content_type`,
   `has_logistics` -- and/or switch DPM outcomes to shares. Re-run. Report the before/after
   table honestly; it is itself a result.
3. Human validation: 150-200 posts, two coders, alpha/ICC per concept; demote what does not
   validate.
4. Write the hypotheses (Section 5); pre-specify the reference window and boundaries; add a
   boundary sensitivity check.
5. Fix the window definitions (end-of-day inclusive or half-open intervals) and rename or split
   the `gaza` window.
6. Interpretation language: periods and coincidence, not causes.

**Should**

7. Presence-based (logit) models for the arenas; ordered-logit robustness for one DPM concept.
8. Topic controls / within-topic analysis.
9. Test-retest of the LLM scoring on a subsample; document all parameters and the rendered
   prompt.
10. Volume, account-share, and content-type composition figures in the report.

**Nice**

11. Event study around global strikes and elections (+/-2 weeks): do mobilization frames spike
    before participation events, and did the spike weaken over time? This is the closest link
    to *political participation* the data offer, and it uses variation *within* windows rather
    than between them.
12. Austria as a control for Germany-specific events (Bundestag elections, Heizungsgesetz
    debate): a difference-in-differences sketch, with the obvious caveats (n = 827, different
    movement dynamics, shared COVID/Ukraine/Gaza exposure).
13. Report caption length itself as a communication-style finding.

---

## 5. A hypothesis set that fits the data you have

Anchors (verify citations before use): Snow & Benford's frame alignment processes --
*amplification* (intensifying existing frames), *extension* (reaching into adjacent concerns),
*bridging*; della Porta & Parks on the climate-justice master frame; Mau, Lux & Westheuser's
arenas as the *content* extension can reach into. Crisis side: issue competition / agenda
crowding (Downs' issue-attention cycle) predicts that exogenous crises crowd climate out of
public attention and push a movement either to link its issue to the crisis (extension /
bridging) or to intensify mobilization (amplification). Sorce & Dumitrica's work on FFF's
COVID-era frame shifts and the Haunss & Sommer volume on FFF Germany are the closest
movement-specific literature.

- **H1 -- crowding produces extension into the matching arena.** During an exogenous crisis,
  FFF's climate framing extends into the inequality arena connected to *that* crisis:
  `innen_aussen` (belonging, solidarity, refugees) in the Ukraine window; `oben_unten`
  (distribution) in the cost-of-living window; `wir_sie` (identity/solidarity conflicts) in
  the Gaza window. *Test*: window-specific level shifts in 06 for the matching arena, with
  length/topic controls. *Falsification*: the mismatched cells -- `oben_unten` should not
  shift more in the Ukraine window than in cost-of-living. *Current data*: `innen_aussen`
  ukraine +0.75 (largest arena shift) supports; `wir_sie` gaza +0.67 (largest) supports;
  `oben_unten` cost_of_living +0.52 vs. ukraine +0.48 does **not** differentiate -- H1 is
  partly falsified as it stands, which is fine and worth writing.
- **H2 -- master-frame shift toward justice.** `climate_justice` rises relative to
  `climate_change` over the period, and the gap widens in crisis windows (justice framing as
  the bridge to non-climate crises). *Test*: model the difference or share
  (justice - change). *Current data*: the gap widens in ukraine and gaza and the justice shifts
  survive length control (+0.56, +0.36).
- **H3 -- amplification: crises shift the DPM mix toward mobilization.** `motivational` /
  `call_to_action` gain share at the expense of diagnosis in crisis windows. *Test*: share-based
  DPM outcomes with controls. *Current data*: after length control the DPM level shifts are
  small or sign-unstable -- H3 is likely rejected. "Extension without amplification" is a
  genuine, interpretable result and is the "beyond description" content the rubric asks for.
- **H4 -- participation link.** Mobilization framing peaks around participation events
  (global strikes, elections) and those peaks shrink across the period as the movement
  declines. *Test*: event study (Section 4, item 11).

Write at least H3 so that it can fail. It probably will.

---

## 6. How to read the current 06 output correctly (until it is re-run with controls)

- "48 of 60 level shifts positive and significant" does not mean crises intensified framing.
  It mostly means captions got longer and every later window is later than the ramp-up.
- Robust to the length control: `climate_justice` (ukraine, cost_of_living, gaza), `oben_unten`
  (ukraine, gaza), `innen_aussen` (ukraine), `wir_sie` (gaza), and `heute_morgen`'s *negative*
  shifts (all windows). Not robust: the DPM levels.
- Slopes are almost uniformly negative relative to baseline; that is the ramp-up artifact, not
  decline during crises. Only narrate a slope if it survives a change of reference window.
- The three clusters in 05's Key Takeaways (mobilization; diagnostic/problem; status/justice)
  are partly the general length factor; keep them as hypotheses until 3.1(c) is done.

---

## 7. Smaller fixes noticed along the way

- `resolve_windows()`: end-of-day inclusive ends or half-open intervals (3.3).
- README method sentence on debiasing across orderings -- corrected in this pass.
- `05`'s per-window summary (`.loc[start:end]`) double-counts boundary days.
- Confirm the ordering pilot and the 04 hierarchy check used the *final* concept definitions.
- `Quaterly Analysis.txt` and `Project_Update.md` are untracked working notes at the repo root
  -- move to `notes/` or ignore them.
- The 10 axis-B parse failures and the 24 unscored posts are silently dropped downstream;
  state the final analytic n per axis in the methods section.
- `notes/environment_constraints.md` links to a `02_framing_feasibility.ipynb` that no longer
  exists.
