# Data Collection Notes

Working notes for the methods section — documents how the analysis sample was pulled and restricted, and why. Numbers reproduced in [`notebooks/00_data_inspection.ipynb`](../notebooks/00_data_inspection.ipynb).

## Source & Query

- **Platform / tool**: Instagram, via Meta Content Library (Meta's academic research data-access tool).
- **Text filter** (OR, case-insensitive): `"fridaysforfuture"` (catches the hashtag and unspaced mentions) and `"fridays for future"` (spaced phrase form).
- **Language filter**: German only, applied at query time.
- **Raw pull**: 44,778 posts, 4,599 unique accounts, 2018-09-03 to 2026-07-29 (query window requested from 2015, but FFF's own posting history only starts in Sept. 2018 — consistent with the movement's founding period).
- API-side language filter verified: 44,753/44,778 (99.9%) actually tagged `lang == "de"`.

## Sample Restrictions

1. **Language — German only.** Justified independently: in the earlier, broader (non-language-restricted, global) sample, the identifiable German FFF chapter accounts already posted 94.9–99.5% in German (96.8% combined), so restricting to German costs negligible data and avoids a translation step / domain-mismatch problem for later classification.
2. **Account restriction — official FFF accounts only.** Kept only posts published *by* an account whose handle matches the FFF pattern (`fridaysforfuture` / `fridays_for_future`, case-insensitive), not third-party posts merely mentioning or hashtagging FFF. This operationalizes "FFF's own communication" (per the project's framing focus) rather than public discourse *about* FFF. Result: **4,138 posts across 8 accounts** (`df_FFF` in the notebook).
3. **Noise precedent (why account-restriction matters).** In the earlier, unrestricted sample, one non-FFF account — an artist reposting his own work under `#fridaysforfuture` alongside dozens of unrelated art-world hashtags — accounted for 11,000+ posts and dominated raw post/hashtag frequency counts despite having nothing to do with the movement. He was dropped explicitly before computing the FFF-account restriction; restricting to official FFF-handle accounts (step 2) would also have excluded this kind of hashtag-hijacking structurally, without needing to name individual noise accounts.
4. **Scope decision — Germany only (resolved 2026-09-22).** The 4,138-post sample splits as:
   - Germany-based chapters (`fridaysforfuture.de`, `.berlin`, `.koeln`, `.muenchen`, `_hh`): **3,310 posts**
   - Austria-based chapters (`fridaysforfuture.at`, `fridaysforfuturevienna`): **827 posts**
   - 1 stray match (`fridaysforfutureusa`)

   The Austrian chapters are dropped. The crisis periodization and the key-event calendar are German (Bundestag elections, Heizungsgesetz, German lockdown timing) and do not apply to Vienna the same way; and the Austrian share is not constant — 29% of the baseline window's posts vs. 7% of the Gaza window's — so pooling would confound the period comparison with a composition change. Austria is kept in reserve as a possible comparison case.
5. **Analysis period ends 2025-12-31.** The pull runs to July 2026, but the 2026 slice holds only 48 Germany-based posts — too few for the quarterly and per-window comparisons. Germany-based, through 2025: 3,262 posts.
6. **0-word posts dropped (resolved 2026-09-23, see "Data Quality Check" above).** 22 posts have no caption left after stripping hashtags/mentions — dropped, since the LLM has nothing to score. **Final analytic sample: 3,240 posts** (Dec 2018 – Dec 2025) across five accounts: Hamburg 705, national 706, Berlin 679, Köln 675, München 475. Per year: 2018 28, 2019 847, 2020 553, 2021 411, 2022 429, 2023 513, 2024 310, 2025 149 — note the steep decline, which makes late-period estimates noisier.

## Known Data-Access Limitation: Incomplete Account Coverage

Several expected regional FFF chapter accounts do not appear in the downloadable export at all, even though they plausibly post matching German-language content. Checked and ruled out: verification status is *not* the differentiator.

Per Meta's own developer documentation for the Content Library API ([IG accounts guide](https://developers.facebook.com/docs/content-library-and-api/content-library-api/guides/ig-accounts/)), inclusion is gated differently by account type:
- **Business / Creator accounts**: included with few or no follower-count restrictions.
- **Personal accounts**: must be public **and** either verified **or** meet a minimum follower threshold (documented as 100–1,000+ followers depending on API version; broader platform documentation elsewhere cites a 25,000-follower bar for personal accounts, recently lowered to 1,000 — [Meta Transparency Center](https://transparency.meta.com/researchtools/meta-content-library/)).

**Most likely explanation**: smaller, volunteer-run city chapters that run their Instagram as a *personal*-type account with a modest following fall below this threshold and are excluded, while larger chapters (Hamburg, Berlin, Cologne, Munich, Vienna) — which clear the threshold and/or use business/creator-type accounts — remain included.

**Caveat**: this is Meta's documented general eligibility rule, not a confirmed diagnosis of *which specific* accounts are missing from this pull — Meta does not publish a full account-level allow-list. Report as a stated data-access limitation, e.g.: *"The sampling frame is bounded by Meta Content Library's account-eligibility criteria (account type and follower/verification thresholds), which plausibly and systematically exclude smaller, lower-follower regional chapter accounts."*

## Data Quality Check: Substantive Text Content

Concern: some posts may be mostly or entirely hashtags/mentions, leaving no room for actual diagnostic/prognostic/motivational content once stripped.

**Method**: strip `#hashtag` tokens, `@mention` tokens, and the anonymized `<redacted_mention>` placeholder from `text`, then count remaining words.

**Result** (n = 4,138):
- 0 words remaining: 34 posts (0.8%)
- ≥ 10 words remaining: 3,906 posts (94.4%)
- ≥ 20 words remaining: 3,507 posts (84.8%)

**Takeaway**: the sample is not dominated by hashtag-only/low-content posts — the large majority of posts carry substantive text well beyond hashtags and mentions, supporting the sample's suitability for frame-scoring (DPM / master-frame / inequality-arena classification per the proposal's §6.3 method).

**Resolved (2026-09-23)**: the 1–9 word posts are kept — they are genuine short mobilization calls ("Freitag = Streiktag!"), not noise, and their DPM scores reflect that (mean `motivational` 1.50 vs. 3.31 for longer posts, but 31% still score ≥ 3). The 0-word posts are dropped as a data-quality rule instead: there is no caption left for the LLM to score, so they carry no framing information at all. Dropped in `00_data_inspection_FFF.ipynb`, right after `clean_word_count` is computed — see §"Scope decision" below for the resulting analytic sample size.
