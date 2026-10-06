"""German NER extraction (Project_Proposal.md §6.2) -- new, no prior implementation existed.

Tracks which actors, places, and events get named over time: a cheap,
interpretable signal of which crisis the movement is explicitly engaging with
at any given point, complementing BERTopic's broader thematic signal.

**Not runnable locally** -- spaCy's compiled pipeline components are blocked by
Smart App Control on this machine, same as the other ML packages (see
notes/environment_constraints.md). Run on the GPU server (or any machine
without that restriction; unlike BERTopic this doesn't need a GPU -- spaCy's
de_core_news_lg runs fine on CPU, just slower than a transformer pipeline).
"""

from __future__ import annotations

import re

import pandas as pd

# spaCy's own label scheme (de_core_news_lg) -- kept explicit here rather than
# scattered as string literals through the code, so a mismatch is one place to check.
ENTITY_LABELS = {
    "PER": "person",
    "LOC": "location",
    "ORG": "organization",
    "MISC": "miscellaneous",
}


def clean_for_ner(text: str) -> str:
    """Strip mentions/URLs but KEEP hashtags (spaCy tags hashtag words fine, and
    the # itself is harmless) -- unlike topic_modeling.clean_for_topic_model,
    NER doesn't need aggressive hashtag stripping since it's not embedding-based.
    """
    if pd.isna(text):
        return ""
    text = re.sub(r"<redacted_mention>", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_entities(
    texts: list[str],
    ids: list[str],
    model_name: str = "de_core_news_lg",
    batch_size: int = 128,
) -> pd.DataFrame:
    """Run spaCy NER over a corpus, one row per (doc, entity) mention.

    Returns a long-format DataFrame: text_id, entity_text, entity_label
    (spaCy's raw PER/LOC/ORG/MISC), entity_label_readable (via ENTITY_LABELS).
    Aggregate with entity_counts_over_time() below, or group by entity_text
    directly for a simple frequency table.

    Requires: python -m spacy download de_core_news_lg (once, on the machine
    this runs on).
    """
    import spacy

    nlp = spacy.load(model_name, disable=["tagger", "parser", "lemmatizer"])

    rows = []
    for doc_id, doc in zip(ids, nlp.pipe((clean_for_ner(t) for t in texts), batch_size=batch_size)):
        for ent in doc.ents:
            rows.append({
                "text_id": doc_id,
                "entity_text": ent.text,
                "entity_label": ent.label_,
                "entity_label_readable": ENTITY_LABELS.get(ent.label_, ent.label_.lower()),
            })
    return pd.DataFrame(rows, columns=["text_id", "entity_text", "entity_label", "entity_label_readable"])


def entity_counts_over_time(
    entities: pd.DataFrame,
    dates: pd.DataFrame,
    date_col: str = "creation_time",
    freq: str = "M",
    top_n: int = 15,
) -> pd.DataFrame:
    """Pivot to (time bin) x (entity) counts for the top-N most frequent entities overall.

    `dates` must have a `text_id` column aligning with `entities.text_id` and
    the date column -- typically just the same df the texts were pulled from.
    """
    merged = entities.merge(dates[["text_id", date_col]], on="text_id", how="left")
    top_entities = merged["entity_text"].value_counts().head(top_n).index
    merged = merged[merged["entity_text"].isin(top_entities)]
    merged["time_bin"] = merged[date_col].dt.to_period(freq).dt.to_timestamp()
    return (
        merged.groupby(["time_bin", "entity_text"]).size()
        .unstack(fill_value=0)
        .reindex(columns=top_entities)
    )
