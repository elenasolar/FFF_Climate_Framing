#!/usr/bin/env python3
"""Score a text corpus on one classification axis, via the ddr package's LLM-DDR
scorer, against an already-running vLLM OpenAI-compatible server (GPU pod).

One axis = one JSON concept file = one invocation of this script. Reuse the same
script for axis A (DPM), axis C (master frame), axis D (justice arena), ... by
pointing --concepts at a different file each time; the pod stays resident across
runs (see Project_Proposal.md §6.3.1 -- separate calls per axis, not one shared
call). Run each axis's Round-1 pass on the full corpus, then re-run with
--ordering-scheme full-factorial (or latin-square for 4+-item axes) on a
subsample to pilot the counterbalancing design (§6.3.2) before committing that
to the full corpus too.

Modeled on the ddr-library's own scripts/score_corpus_llm_ddr.py (sharding +
checkpointing so a tmux session can be killed/resumed safely), extended with:
  - backend="api" (score_corpus_llm_ddr.py only exposes vllm/hf)
  - counterbalanced orderings of the SAME concepts dict, to test the
    primacy/recency position effects called out in the proposal's order-effect
    design, rather than a single fixed prompt order
  - --workers: concurrent (ordering, shard) requests against the pod, since
    score_texts() itself makes one blocking HTTP call per batch -- see the
    PREREQUISITES section below for why this only makes sense for backend=api

Typical invocation, on the GPU server / a host with network access to the pod
(NOT this repo's local Windows venv -- see notes/environment_constraints.md;
score_texts() loads a HuggingFace tokenizer locally even for backend="api", and
that pulls in `transformers`, which is blocked here by Smart App Control):

    python score_axis_llm_ddr.py \\
        --input             data/processed/fff_de_topics.csv \\
        --text-col          clean_text \\
        --id-col            id \\
        --concepts          concepts/axis_a_dpm_de.json \\
        --output            data/llm_scores/axis_a_dpm_de.csv \\
        --model             Qwen/Qwen3-4B-Instruct-2507 \\
        --backend           api \\
        --api-base-url      http://<pod-service>:8000/v1 \\
        --language          German \\
        --scale-min 0 --scale-max 4 \\
        --ordering-scheme   full-factorial \\
        --batch-size        64 \\
        --shard-size        2000 \\
        --workers           4

Output: one CSV, LONG format -- one row per (document, ordering). Columns:
text_id, ordering_id, ordering (comma-joined concept order used for that run),
one column per concept (integer rating, or NaN if unparseable), parse_error.
No text column -- this is long-format (one row per doc PER ORDERING), so
including it would duplicate the same text once per ordering; text_id already
matches the source corpus's id column, so join it back there when you want to
inspect (see notebooks/05_llm_scoring_analysis.ipynb) rather than carrying a
redundant copy in every row here. Separately: this also does NOT include the
model's raw response text even on parse failure -- the ddr package's own
output_parser.py discards it during parsing, keeping only the parsed ratings
and (on failure) which concept names failed to parse.
Aggregating across orderings (mean/SD for the Likert scores, agreement stats)
is deliberately NOT done here -- that's analysis, and belongs in a notebook
(see notebooks/05_llm_scoring_analysis.ipynb), not in this script. Order-effect
diagnostics on a counterbalanced pilot run (--ordering-scheme full-factorial/
latin-square) belong in notebooks/01_robustness_checks.ipynb instead.


PREREQUISITES (checklist before running for real)
───────────────────────────────────────────────────────────────────────────
1. GPU pod running the model, reachable from wherever this script runs:
       vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000
   (add --api-key <key> there only if you want to require one; otherwise the
   package's default of "EMPTY" -- vLLM's own default -- is fine).
   Note the pod's address as seen from the *calling* machine, not from inside
   the pod itself -- e.g. a Kubernetes service DNS name / internal IP, not
   "localhost", unless you're port-forwarding.

2. The calling environment (the lab's Linux server, in a tmux session -- NOT
   the pod, and not this repo's local Windows venv) needs:
       pip install -e /path/to/ddr-library transformers openai pandas tqdm
   No torch/vllm/GPU stack required *there* -- inference happens on the pod.
   `transformers` is still needed locally because score_texts() always loads
   a tokenizer via AutoTokenizer.from_pretrained() to build the chat-templated
   prompt, regardless of backend. If that tokenizer load fails with something
   torch-shaped, install a CPU-only torch build too (small, no CUDA needed):
       pip install torch --index-url https://download.pytorch.org/whl/cpu

3. Know, before typing the command:
     - the pod's URL as --api-base-url, e.g. http://<pod-service>:8000/v1
     - --api-key, only if the pod was started with --api-key
     - --model must match EXACTLY what the pod was started with (vllm serve's
       first argument) -- it's used both for the tokenizer and to address the
       model on the server's /v1/completions endpoint
     - your input file + which text column to score. Recommend --text-col
       clean_text (hashtags/mentions already stripped, from
       00_data_inspection.ipynb's "Substantive Text Content" section) rather
       than raw text, for the same reason topic modeling needed it: hashtag
       blocks are boilerplate, not framing content.

4. Smoke test BEFORE pointing this at the full corpus (see --smoke-test below)
   -- confirm the pod responds, the concepts parse, and ratings look sane on
   a handful of documents before spending real time/tokens on thousands.

Exact command for a 50-document smoke test of axis A, German definitions:

    python score_axis_llm_ddr.py \\
        --input        data/processed/fff_de_topics.csv \\
        --text-col     clean_text \\
        --id-col       id \\
        --concepts     concepts/axis_a_dpm_de.json \\
        --output       data/llm_scores/_smoke_axis_a_dpm_de.csv \\
        --model        Qwen/Qwen3-4B-Instruct-2507 \\
        --backend      api \\
        --api-base-url http://<pod-service>:8000/v1 \\
        --language     German \\
        --smoke-test   50

Once that looks right, drop --smoke-test, point --output at the real path, and
add --workers (api backend only) and/or --ordering-scheme for the real run.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ddr.llm_ddr import LLMDDRConfig, score_texts
from ddr.utils.logging_config import configure_logging

# Eager, main-thread-only import: ddr's own score_texts() does `from transformers import
# AutoTokenizer` internally, lazily, on first call -- transformers resolves its submodules
# lazily too, and that first resolution isn't thread-safe. With --workers > 1, multiple
# worker threads can hit that first import at nearly the same moment and race on
# transformers' internal lazy-module cache: one thread wins and finishes cleanly, another
# gets a spurious "cannot import name 'AutoTokenizer' from 'transformers'" ImportError that
# kills the whole run even though the environment and install are fine. Forcing the
# resolution once here, before any ThreadPoolExecutor exists, makes every later import (from
# any thread) a cheap cache hit instead of a race.
from transformers import AutoTokenizer  # noqa: F401

logger = logging.getLogger(__name__)


# ── Orderings ────────────────────────────────────────────────────────────────

def generate_orderings(concept_names: list[str], scheme: str) -> list[list[str]]:
    """Return the list of concept-key orderings to run, per Project_Proposal.md §6.3.2.

    - "single":         one run, the concepts file's own order (no counterbalancing).
    - "full-factorial": every permutation (n!). Only sane for small n (DPM: 3! = 6;
                         master frame: 2! = 2). Raises for n >= 4 (4! = 24 is already
                         the point the proposal switches to a Latin square instead).
    - "latin-square":   n orderings (n = number of concepts), each concept appearing
                         exactly once in each prompt position, via cyclic rotation.
                         Use for larger axes (e.g. the 4-item justice arena) where
                         full factorial is too expensive.
    """
    n = len(concept_names)

    if scheme == "single":
        return [list(concept_names)]

    if scheme == "full-factorial":
        if n > 3:
            raise ValueError(
                f"full-factorial with {n} concepts means {n}! = {_factorial(n)} runs -- "
                f"use --ordering-scheme latin-square instead (see §6.3.2)."
            )
        return [list(p) for p in itertools.permutations(concept_names)]

    if scheme == "latin-square":
        # Cyclic Latin square: row i is the list rotated left by i places.
        # Every concept appears exactly once in every position across the n rows.
        return [concept_names[i:] + concept_names[:i] for i in range(n)]

    raise ValueError(f"Unknown --ordering-scheme '{scheme}'.")


def _factorial(n: int) -> int:
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # I/O
    p.add_argument("--input", required=True, help="Input file (.csv, .csv.gz, or .parquet)")
    p.add_argument("--concepts", required=True, help="JSON file: concept name -> {definition, cue?}")
    p.add_argument("--output", required=True, help="Output path (.csv), long format across orderings")
    p.add_argument("--text-col", default="text", help="Text column name (default: text)")
    p.add_argument("--id-col", default=None, help="ID column name (default: row index)")
    p.add_argument("--compression", default="infer", help="CSV compression (default: infer from extension)")

    # Model / backend
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="HuggingFace model ID or local path")
    p.add_argument("--backend", default="api", choices=["vllm", "hf", "api"],
                   help="Inference backend (default: api -- see module docstring for why)")
    p.add_argument("--api-base-url", default=None,
                   help="OpenAI-compatible server URL, e.g. http://<pod-service>:8000/v1 (required for backend=api)")
    p.add_argument("--api-key", default="EMPTY", help="Sent to the server (default: EMPTY, vLLM's default)")
    p.add_argument("--api-timeout", type=float, default=600.0, help="Per-request timeout in seconds (backend=api)")
    p.add_argument("--language", default="German", help="Language of the input texts, inserted into the prompt")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="Model dtype -- only matters for backend=vllm/hf, ignored for api")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--task-type", default="chat", choices=["chat", "text2text"])

    # vLLM-only settings (ignored for backend=api/hf)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)

    # Generation
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--scale-min", type=int, default=0, help="Likert floor (default 0)")
    p.add_argument("--scale-max", type=int, default=4,
                   help="Likert ceiling (default 4 -- the package's built-in scale labels "
                        "'None at all' .. 'A great deal' only cover 0-4; going higher is "
                        "supported but points beyond 4 fall back to bare numbers in the "
                        "prompt, with no semantic anchor. See Project_Proposal.md §6.3.")

    # Counterbalancing (Project_Proposal.md §6.3.2)
    p.add_argument("--ordering-scheme", default="single",
                   choices=["single", "full-factorial", "latin-square"],
                   help="How many times (and in what concept order) to run this axis. "
                        "'single' for a normal scoring pass; 'full-factorial' or "
                        "'latin-square' to pilot/run the order-effect check.")

    # Throughput / checkpointing
    p.add_argument("--batch-size", type=int, default=64,
                   help="Documents per generate() call (default 64). vLLM/the server "
                        "batches this internally in ONE request; this is the first lever "
                        "for GPU utilization, independent of --workers below.")
    p.add_argument("--shard-size", type=int, default=2000,
                   help="Documents per checkpoint shard (default 2000, smaller than the "
                        "ddr-library default since this script also multiplies calls by "
                        "the number of orderings).")
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent (ordering, shard) requests against the pod (default 1, "
                        "sequential). Requires --backend api: score_texts() creates a fresh "
                        "backend instance per call, and for vllm/hf that means loading the "
                        "full model into this process again -- fine once, dangerous N times "
                        "concurrently (GPU OOM). The api backend just opens an HTTP client, "
                        "so concurrent calls are cheap here and simply queue on the pod's "
                        "own vLLM server, which already handles concurrent requests. Start "
                        "at 2-4 and watch pod GPU utilization before going higher.")

    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--system-prompt-file", default=None)
    p.add_argument("--smoke-test", nargs="?", type=int, const=20, default=None, metavar="N",
                   help="Run on only the first N documents (default 20 if flag given with "
                        "no value, e.g. --smoke-test alone, or --smoke-test 50 for a "
                        "specific count). Omit entirely for a full run.")
    p.add_argument("--keep-checkpoints", action="store_true")

    args = p.parse_args()

    if args.backend == "api" and not args.api_base_url:
        raise SystemExit("--api-base-url is required when --backend api (e.g. http://<pod-service>:8000/v1)")
    if args.workers > 1 and args.backend != "api":
        raise SystemExit(
            f"--workers {args.workers} requires --backend api (got --backend {args.backend}) -- "
            f"see --workers' help text for why concurrent vllm/hf backends is a GPU-memory hazard, "
            f"not just an inefficiency."
        )
    return args


# ── Data loading ─────────────────────────────────────────────────────────────

def load_corpus(
    input_path: str, text_col: str, id_col: str | None,
    compression: str, smoke_test_n: int | None,
) -> tuple[list[str], list[str]]:
    path = Path(input_path)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        if smoke_test_n is not None:
            df = df.head(smoke_test_n)
    else:
        df = pd.read_csv(path, compression=compression, nrows=smoke_test_n)

    n_before = len(df)
    df = df[df[text_col].notna()].copy()
    if len(df) < n_before:
        logger.warning("Dropped %d rows with missing text.", n_before - len(df))

    texts = df[text_col].astype(str).tolist()

    if id_col and id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
    else:
        if id_col:
            logger.warning("ID column '%s' not found -- using row index.", id_col)
        ids = [str(i) for i in df.index]

    return texts, ids


# ── Work units (one per (ordering, shard) pair) ─────────────────────────────

@dataclass
class WorkUnit:
    ordering_i: int
    ordering_label: str
    ordered_concepts: dict
    shard_i: int
    n_shards: int
    shard_texts: list[str]
    shard_ids: list[str]
    ckpt_path: Path


def build_work_units(
    texts: list[str], text_ids: list[str], orderings: list[list[str]],
    concept_definitions: dict, shard_size: int, ckpt_dir: Path,
) -> list[WorkUnit]:
    n_shards = max(1, (len(texts) + shard_size - 1) // shard_size)
    units = []
    for ordering_i, ordering in enumerate(orderings):
        ordered_concepts = {name: concept_definitions[name] for name in ordering}
        ordering_label = ",".join(ordering)
        for shard_i in range(n_shards):
            start = shard_i * shard_size
            end = min(start + shard_size, len(texts))
            units.append(WorkUnit(
                ordering_i=ordering_i,
                ordering_label=ordering_label,
                ordered_concepts=ordered_concepts,
                shard_i=shard_i,
                n_shards=n_shards,
                shard_texts=texts[start:end],
                shard_ids=text_ids[start:end],
                ckpt_path=ckpt_dir / f"ordering_{ordering_i:02d}_shard_{shard_i:05d}.csv",
            ))
    return units


def run_unit(unit: WorkUnit, config: LLMDDRConfig, concept_names: list[str]) -> tuple[Path, int, float]:
    """Score one (ordering, shard) unit and write its checkpoint. Returns (path, n_docs, elapsed)."""
    t0 = time.time()
    results = score_texts(
        texts=unit.shard_texts, concepts=unit.ordered_concepts, config=config, ids=unit.shard_ids,
    )
    rows = []
    for r in results:
        # No r.text here -- this is long-format (one row per doc PER ORDERING), so it would
        # duplicate the same text once per ordering (6x for a DPM full-factorial run). text_id
        # already matches the source corpus's id column, so join it back there when you want
        # to look (see notebooks/05_llm_scoring_analysis.ipynb's Load Scores section) rather
        # than carrying a redundant copy in every row of the scores file itself.
        row = {"text_id": r.doc_id, "ordering_id": unit.ordering_i, "ordering": unit.ordering_label}
        for concept in concept_names:
            row[concept] = r.ratings.get(concept)
        row["parse_error"] = r.error
        rows.append(row)
    pd.DataFrame(rows).to_csv(unit.ckpt_path, index=False)
    return unit.ckpt_path, len(unit.shard_texts), time.time() - t0


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    configure_logging(level=logging.INFO)

    t_total = time.time()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load concept definitions ──────────────────────────────────────────
    with open(args.concepts, encoding="utf-8") as f:
        concept_definitions: dict = json.load(f)
    concept_names = list(concept_definitions.keys())
    logger.info("Concepts (%d): %s", len(concept_names), concept_names)

    orderings = generate_orderings(concept_names, args.ordering_scheme)
    logger.info("Ordering scheme '%s': %d run(s) -- %s",
                args.ordering_scheme, len(orderings), orderings)

    # ── Load corpus ────────────────────────────────────────────────────────
    logger.info("Loading corpus from %s ...", args.input)
    texts, text_ids = load_corpus(
        args.input, args.text_col, args.id_col, args.compression, args.smoke_test
    )
    logger.info("Corpus: %d documents%s", len(texts),
                f" (SMOKE TEST -- first {args.smoke_test})" if args.smoke_test else "")

    # ── Custom system prompt (optional) ───────────────────────────────────
    system_prompt = None
    if args.system_prompt_file:
        with open(args.system_prompt_file, encoding="utf-8") as f:
            system_prompt = f.read().strip()
        logger.info("Custom system prompt loaded from %s", args.system_prompt_file)

    # ── Build config (shared across orderings -- only the concepts dict's
    #    key order changes between runs, everything else stays fixed) ──────
    config = LLMDDRConfig(
        model_name=args.model,
        backend=args.backend,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        retries=args.retries,
        language=args.language,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        task_type=args.task_type,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        system_prompt=system_prompt,
        enable_thinking=args.enable_thinking,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        api_timeout=args.api_timeout,
    )

    ckpt_dir = out_path.parent / f"_llm_checkpoints_{out_path.stem}"
    ckpt_dir.mkdir(exist_ok=True)

    all_units = build_work_units(texts, text_ids, orderings, concept_definitions, args.shard_size, ckpt_dir)
    pending_units = [u for u in all_units if not u.ckpt_path.exists()]
    n_skipped = len(all_units) - len(pending_units)
    if n_skipped:
        logger.info("%d/%d (ordering, shard) units already checkpointed -- skipping.", n_skipped, len(all_units))

    if args.workers <= 1:
        for unit in tqdm(pending_units, desc="Scoring", unit="unit"):
            _, n_docs, elapsed = run_unit(unit, config, concept_names)
            logger.info(
                "ordering %d shard %d/%d done -- %d docs in %.1fs (%.1f docs/s)",
                unit.ordering_i, unit.shard_i + 1, unit.n_shards, n_docs, elapsed, n_docs / elapsed,
            )
    else:
        logger.info("Running with %d concurrent workers against the pod ...", args.workers)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_unit, unit, config, concept_names): unit for unit in pending_units}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring", unit="unit"):
                unit = futures[future]
                _, n_docs, elapsed = future.result()  # re-raises if the worker thread errored
                logger.info(
                    "ordering %d shard %d/%d done -- %d docs in %.1fs (%.1f docs/s)",
                    unit.ordering_i, unit.shard_i + 1, unit.n_shards, n_docs, elapsed, n_docs / elapsed,
                )

    # ── Merge every checkpoint (this run's + any resumed from before) ──────
    # dtype={"text_id": str} pins text_id as string on read-back -- CSV has no way to
    # distinguish "always a string" from "looks numeric," and without this a numeric-
    # looking id column (e.g. "0","1",...) would silently come back as int64 instead
    # of matching whatever dtype your source id column actually has.
    df_out = pd.concat(
        [pd.read_csv(u.ckpt_path, dtype={"text_id": str}) for u in all_units], ignore_index=True,
    )

    for concept in concept_names:
        n_missing = df_out[concept].isna().sum()
        logger.info(
            "Concept '%s': %d/%d ratings parsed (%d missing, %.1f%%)",
            concept, len(df_out) - n_missing, len(df_out),
            n_missing, 100 * n_missing / len(df_out),
        )

    df_out.to_csv(out_path, index=False)
    logger.info("Saved %d rows (%d docs x %d orderings) to %s",
                len(df_out), len(texts), len(orderings), out_path)

    if not args.keep_checkpoints:
        for u in all_units:
            u.ckpt_path.unlink(missing_ok=True)
        ckpt_dir.rmdir()
        logger.info("Checkpoint files removed.")

    total_elapsed = time.time() - t_total
    logger.info(
        "Done -- %d documents x %d orderings in %.1fs (%.2f docs/s overall, incl. all orderings, %d worker(s)).",
        len(texts), len(orderings), total_elapsed,
        (len(texts) * len(orderings)) / total_elapsed, args.workers,
    )


if __name__ == "__main__":
    main()
