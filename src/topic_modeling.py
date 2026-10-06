"""Reusable BERTopic pipeline, matching the two-pass methodology validated in
notebooks/00_data_inspection_FFF.ipynb §5, so the overall-corpus and
per-crisis-window notebooks share one implementation instead of copy-pasted
cells drifting apart (and out of sync with what's actually validated).

Two-pass fit (`fit_topic_model_two_pass`): a deliberately over-aggressive
"broad" fit first (small `min_cluster_size` -- over-fragments on purpose,
kept only as a diagnostic baseline you can compare against), then a cheap
`reduce_topics()` exploration toward a target count (diagnostic only, doesn't
feed anything downstream), then a "refined" re-fit with `min_cluster_size`
re-tuned toward that target via real re-clustering rather than post-hoc
merging -- that refined fit is what gets returned and used. Embeddings are
computed once and reused across all three fits, since only the clustering
step changes between them.

Shared preprocessing: hashtags stripped before embedding (boilerplate
branding tags otherwise dominate the pooled embedding), cluster_selection_method
"leaf" rather than the HDBSCAN default "eom" (predictable granularity control),
min_df/max_df left at (1, 1.0) on the vectorizer since those filter BERTopic's
per-TOPIC pseudo-documents, not raw posts.

**Not runnable locally** -- torch/sentence-transformers/bertopic are blocked by
Smart App Control on this machine (notes/environment_constraints.md). Run on
the GPU server, same as notebooks/00_data_inspection_FFF.ipynb §5.
"""

from __future__ import annotations

import re

import pandas as pd


def clean_for_topic_model(text: str) -> str:
    """Strip hashtags, mentions, and URLs -- embed on prose only.

    Hashtags are kept as a SEPARATE analysis (the hashtag co-occurrence heatmap
    and manual codebook in 00_data_inspection_FFF.ipynb) precisely because keeping
    them in the embedding text was what caused the original topic-collapse bug.
    """
    if pd.isna(text):
        return ""
    text = re.sub(r"<redacted_mention>", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prepare_topic_docs(
    df: pd.DataFrame, text_col: str = "text", date_col: str = "creation_time",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Clean texts and drop empty docs, returning (docs, dates, non_empty_mask).

    `non_empty_mask` is boolean, aligned to `df`'s original index -- NOT to the
    returned (reset-index) docs/dates. Callers merging per-doc results (e.g.
    topic assignments) back onto the original, unfiltered df MUST use this mask
    (`df.loc[non_empty_mask, "topic"] = topics`), not a separately recomputed
    "is text non-empty" check -- a post can have non-empty RAW text but become
    empty after clean_for_topic_model strips its hashtags/mentions/URLs (e.g. a
    post that's just "#fridaysforfuture #klima"), so the two masks disagree and
    recomputing one independently silently drops the wrong rows (this caused a
    real "Length of values (4104) does not match length of index (4138)" bug
    in 02_topic_model_ner_overall.ipynb's Save Results cell).

    Both the overall-corpus and by-crisis-window notebooks call this on
    whatever subset of df they're currently working with.
    """
    docs = df[text_col].apply(clean_for_topic_model)
    non_empty = docs.str.len() > 0
    return (
        docs[non_empty].reset_index(drop=True),
        df.loc[non_empty, date_col].reset_index(drop=True),
        non_empty,
    )


def fit_topic_model_two_pass(
    docs: list[str],
    target_n_topics: int = 30,
    embedding_model_name: str = "paraphrase-multilingual-mpnet-base-v2",
    initial_min_cluster_size: int = 10,
    min_samples: int = 5,
    n_neighbors: int = 10,
    seed_topic_list: list[list[str]] | None = None,
    refined_min_cluster_size: int | None = None,
    device: str | None = None,
) -> dict:
    """Two-pass BERTopic fit -- broad diagnostic fit, then a refined re-fit
    that's the one actually returned. See module docstring for the reasoning.

    The refined fit's `min_cluster_size` defaults to the "~1% of corpus"
    heuristic validated in 00_data_inspection_FFF.ipynb on the FULL corpus
    (`max(10, len(docs) // 100)`) -- pass `refined_min_cluster_size` explicitly
    to override this for a much smaller corpus (e.g. a single crisis window),
    where the flat 1%-of-corpus rule floors out at 10 and ends up barely
    different from the broad pass's own `initial_min_cluster_size=10`, i.e. no
    real consolidation happens. Either way there's no closed-form guarantee
    `target_n_topics` is hit exactly -- `target_n_topics` itself only drives the
    (diagnostic-only) `reduce_topics()` exploration, not the refined fit
    directly; check the printed counts and adjust if a result is far off.

    Import torch/sentence_transformers/bertopic/spacy/sklearn/umap/hdbscan
    lazily (inside the function) so importing this module doesn't require a
    GPU environment -- only calling this function does.

    Returns a dict:
        topic_model, topics, topic_info -- the REFINED fit; use these downstream.
        broad_n_topics, broad_outlier_frac -- diagnostic baseline from pass 1.
        refined_min_cluster_size, refined_outlier_frac -- diagnostic for pass 3.
    """
    import torch
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS
    from umap import UMAP

    docs = list(docs)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    embedding_model = SentenceTransformer(embedding_model_name, device=device)
    embeddings = embedding_model.encode(docs, show_progress_bar=True)

    vectorizer_model = CountVectorizer(
        stop_words=list(DE_STOP_WORDS), min_df=1, max_df=1.0, ngram_range=(1, 2),
    )

    def _build_model(min_cluster_size: int) -> BERTopic:
        umap_model = UMAP(
            n_neighbors=n_neighbors, n_components=5, min_dist=0.0, metric="cosine", random_state=42,
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
            cluster_selection_method="leaf", prediction_data=True,
        )
        return BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            seed_topic_list=seed_topic_list,
            calculate_probabilities=False,
            verbose=True,
        )

    # ── Pass 1: broad, deliberately over-aggressive fit (diagnostic baseline only) ──
    broad_model = _build_model(initial_min_cluster_size)
    broad_topics, _ = broad_model.fit_transform(docs, embeddings=embeddings)
    broad_info = broad_model.get_topic_info()
    broad_n_topics = int((broad_info["Topic"] != -1).sum())
    broad_outlier_frac = sum(1 for t in broad_topics if t == -1) / len(broad_topics)

    # ── Pass 2: reduce_topics() exploration toward target_n_topics (diagnostic only,
    # mutates broad_model in place -- fine, broad_model isn't returned or reused after this) ──
    broad_model.reduce_topics(docs, nr_topics=target_n_topics)

    # ── Pass 3: refined re-fit, real re-clustering aimed at target_n_topics ──
    if refined_min_cluster_size is None:
        refined_min_cluster_size = max(10, len(docs) // 100)
    refined_model = _build_model(refined_min_cluster_size)
    refined_topics, _ = refined_model.fit_transform(docs, embeddings=embeddings)
    refined_info = refined_model.get_topic_info()
    refined_outlier_frac = sum(1 for t in refined_topics if t == -1) / len(refined_topics)

    return {
        "topic_model": refined_model,
        "topics": refined_topics,
        "topic_info": refined_info,
        "broad_n_topics": broad_n_topics,
        "broad_outlier_frac": broad_outlier_frac,
        "refined_min_cluster_size": refined_min_cluster_size,
        "refined_outlier_frac": refined_outlier_frac,
    }


def outlier_summary(topics: list[int]) -> tuple[int, float]:
    """(n_outliers, outlier_fraction) -- topic_model's own -1 "noise" bucket."""
    n = len(topics)
    n_outliers = sum(1 for t in topics if t == -1)
    return n_outliers, n_outliers / n if n else float("nan")
