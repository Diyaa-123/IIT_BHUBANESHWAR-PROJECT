"""
src/eval/evaluate_retrieval.py
──────────────────────────────
Evaluation script for the CLIP-based image search engine.

Computes:
  • Precision@K          — fraction of top-K results that are relevant
  • Recall@K             — fraction of all relevant images found in top-K
  • Top-K Accuracy       — ≥1 relevant result appears in the top-K (0/1 per query)
  • Mean Reciprocal Rank — how high the first relevant result ranks
  • Avg cosine similarity of top-K results
  • Average query latency (ms)

Usage (from the project root):
  python -m src.eval.evaluate_retrieval
  python -m src.eval.evaluate_retrieval --k 10
  python -m src.eval.evaluate_retrieval --k 5 --repeats 5
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── project imports ──────────────────────────────────────────────────────────
from src.config import EMBEDDINGS_FILE, IMAGE_PATHS_FILE, CLIP_MODEL_NAME, DEVICE
from src.embeddings import CLIPEmbedder
from src.search import top_k_search

# ── paths ────────────────────────────────────────────────────────────────────
_EVAL_DIR = Path(__file__).parent
TEST_QUERIES_FILE = _EVAL_DIR / "test_queries.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_index() -> Tuple[np.ndarray, List[str]]:
    """Load pre-computed image embeddings and corresponding file paths."""
    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found: {EMBEDDINGS_FILE}\n"
            "Run 'python prepare_embeddings.py' first."
        )
    if not IMAGE_PATHS_FILE.exists():
        raise FileNotFoundError(
            f"Image-paths file not found: {IMAGE_PATHS_FILE}\n"
            "Run 'python prepare_embeddings.py' first."
        )

    embeddings = np.load(EMBEDDINGS_FILE)
    with open(IMAGE_PATHS_FILE, "r", encoding="utf-8") as fh:
        image_paths = json.load(fh)

    return embeddings, image_paths


def load_test_queries() -> List[Dict]:
    """Load relevance-judgment queries from the JSON file."""
    if not TEST_QUERIES_FILE.exists():
        raise FileNotFoundError(
            f"Test queries file not found: {TEST_QUERIES_FILE}"
        )
    with open(TEST_QUERIES_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def normalise_path(p: str) -> str:
    """
    Return the bare filename (stem+suffix) so that paths stored in
    image_paths.json match the short paths in test_queries.json regardless
    of whether they are absolute or relative.
    """
    return Path(p).name


def evaluate_query(
    query: str,
    relevant_paths: List[str],
    embedder: CLIPEmbedder,
    image_embeddings: np.ndarray,
    image_paths: List[str],
    k: int,
    n_repeats: int,
) -> Dict:
    """Run one query and return a dict of all metric values."""

    # Warm-up + latency measurement (average over n_repeats)
    latencies_ms: List[float] = []
    results: List[Tuple[str, float]] = []

    for i in range(n_repeats):
        t0 = time.perf_counter()
        text_emb = embedder.generate_text_embedding(query)
        results = top_k_search(text_emb, image_embeddings, image_paths, top_k=k)
        latencies_ms.append((time.perf_counter() - t0) * 1_000)

    avg_latency_ms = float(np.mean(latencies_ms))

    # Normalise paths to bare filenames for comparison
    retrieved_names = [normalise_path(p) for p, _ in results]
    relevant_names  = {normalise_path(p) for p in relevant_paths}

    # ── metrics ──────────────────────────────────────────────────────────────
    hits = [name in relevant_names for name in retrieved_names]

    true_positives = sum(hits)

    precision_at_k = true_positives / k
    recall_at_k    = (true_positives / len(relevant_names)) if relevant_names else 0.0
    top_k_accuracy = 1.0 if true_positives > 0 else 0.0

    # Mean Reciprocal Rank — rank of first relevant result
    rr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            rr = 1.0 / rank
            break

    avg_cosine = float(np.mean([score for _, score in results]))
    top1_cosine = float(results[0][1]) if results else 0.0

    return {
        "query":          query,
        "precision_at_k": precision_at_k,
        "recall_at_k":    recall_at_k,
        "top_k_accuracy": top_k_accuracy,
        "mrr":            rr,
        "avg_cosine":     avg_cosine,
        "top1_cosine":    top1_cosine,
        "latency_ms":     avg_latency_ms,
        "retrieved":      retrieved_names,
        "relevant":       list(relevant_names),
        "true_positives": true_positives,
        "k":              k,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_per_query(r: Dict, verbose: bool = False) -> None:
    status = "OK" if r["top_k_accuracy"] else "--"
    print(
        f"  [{status}] \"{r['query']}\"\n"
        f"       P@{r['k']}={r['precision_at_k']:.3f}  "
        f"R@{r['k']}={r['recall_at_k']:.3f}  "
        f"MRR={r['mrr']:.3f}  "
        f"cos={r['avg_cosine']:.3f}  "
        f"latency={r['latency_ms']:.1f}ms"
    )
    if verbose:
        print(f"       Retrieved : {r['retrieved']}")
        print(f"       Relevant  : {sorted(r['relevant'])}")


def print_summary(results: List[Dict], k: int) -> None:
    n = len(results)
    mean = lambda key: float(np.mean([r[key] for r in results]))

    print("\n" + "="*60)
    print("  EVALUATION SUMMARY")
    print("="*60)
    print(f"  Queries evaluated         : {n}")
    print(f"  Top-K (K)                 : {k}")
    print(f"  Mean Precision@{k:<2}         : {mean('precision_at_k'):.4f}")
    print(f"  Mean Recall@{k:<2}            : {mean('recall_at_k'):.4f}")
    print(f"  Top-{k} Accuracy            : {mean('top_k_accuracy'):.4f}"
          f"  ({int(sum(r['top_k_accuracy'] for r in results))}/{n} queries)")
    print(f"  Mean Reciprocal Rank (MRR): {mean('mrr'):.4f}")
    print(f"  Mean Avg Cosine (top-{k})  : {mean('avg_cosine'):.4f}")
    print(f"  Mean Top-1 Cosine         : {mean('top1_cosine'):.4f}")
    print(f"  Mean Query Latency        : {mean('latency_ms'):.1f} ms")
    print("="*60)

    # Best / worst queries
    by_prec = sorted(results, key=lambda r: r["precision_at_k"])
    print(f"\n  Best  query : \"{by_prec[-1]['query']}\"  "
          f"P@{k}={by_prec[-1]['precision_at_k']:.3f}")
    print(f"  Worst query : \"{by_prec[0]['query']}\"  "
          f"P@{k}={by_prec[0]['precision_at_k']:.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality of the CLIP image search engine."
    )
    parser.add_argument("--k",       type=int, default=5,
                        help="Top-K images to retrieve per query (default: 5).")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Latency measurement repeats per query (default: 3).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print retrieved / relevant filenames per query.")
    parser.add_argument("--save",    type=str, default="",
                        help="Optional path to save full results as JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\n" + "-"*60)
    print(f"  Image Search Engine -- Retrieval Evaluation")
    print(f"  Model  : {CLIP_MODEL_NAME}")
    print(f"  Device : {DEVICE.upper()}")
    print(f"  K      : {args.k}   Repeats: {args.repeats}")
    print("-"*60 + "\n")

    # load data
    print("Loading index...")
    image_embeddings, image_paths = load_index()
    print(f"  {len(image_paths):,} images indexed  |  "
          f"embedding dim: {image_embeddings.shape[1]}\n")

    print("Loading CLIP model...")
    embedder = CLIPEmbedder(model_name=CLIP_MODEL_NAME, device=DEVICE)
    print()

    tests = load_test_queries()
    print(f"Evaluating {len(tests)} test queries...\n")

    # evaluate
    all_results: List[Dict] = []
    for test in tests:
        r = evaluate_query(
            query            = test["query"],
            relevant_paths   = test["relevant"],
            embedder         = embedder,
            image_embeddings = image_embeddings,
            image_paths      = image_paths,
            k                = args.k,
            n_repeats        = args.repeats,
        )
        all_results.append(r)
        print_per_query(r, verbose=args.verbose)

    # summary
    print_summary(all_results, k=args.k)

    # optional JSON export
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"\n  Full results saved: {out_path}")

    print()


if __name__ == "__main__":
    main()
