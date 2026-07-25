# evaluation/evaluate_embeddings.py

# Imports
import json
import time
import sys
import gc
import numpy as np
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# PATHS
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKED_DIR = PROJECT_ROOT / "data" / "chunked"
GOLDEN_DATASET_PATH = PROJECT_ROOT / "evaluation" / "golden_data.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# MODELS TO EVALUATE
MODELS_TO_EVALUATE = [
    "sentence-transformers/all-MiniLM-L6-v2",         # baseline
    "BAAI/bge-small-en-v1.5",                         
    "thenlper/gte-small",                              
    "sentence-transformers/all-mpnet-base-v2",         
    "nlpaueb/legal-bert-base-uncased",                 

    # --- Larger models (Kaggle/Colab recommended) ---
    # "BAAI/bge-base-en-v1.5",                          # 440MB — strong baseline
    # "BAAI/bge-large-en-v1.5",                         # 1.3GB — best BGE
    # "Euler-Legal-Embedding-V1",                        # check size first
    # "thenlper/gte-base",                               # 220MB
    # "thenlper/gte-large",  
    # Octen-Embedding-0.6B                             # 670MB
]

# METRICS DATA CLASS
@dataclass
class EvaluationResult:
    """
    Stores all evaluation metrics for one embedding model.

    Metrics explained:
    ─────────────────
    Recall@1 (R@1):
        Percentage of questions where the correct chunk is the #1 result.
        The strictest metric. If R@1 = 70%, 7 out of 10 questions return
        the correct chunk as the very top result.
        Target for production: > 60%

    Recall@5 (R@5):
        Percentage of questions where the correct chunk appears anywhere
        in the top 5 results. More forgiving than R@1.
        We retrieve top 5 in production so R@5 is the practical ceiling.
        Target for production: > 85%

    MRR (Mean Reciprocal Rank):
        Average of 1/rank for each correct result.
        If correct chunk is rank 1: score = 1/1 = 1.0
        If correct chunk is rank 2: score = 1/2 = 0.5
        If correct chunk is rank 3: score = 1/3 = 0.33
        Not in top 5: score = 0
        MRR combines precision and ranking quality into one number.
        Target for production: > 0.70

    Why all three?
        R@1 tells you: "Does it get it right first time?"
        R@5 tells you: "Does it find it at all?"
        MRR tells you: "How well does it rank what it finds?"
        A model can have high R@5 but low R@1 — it finds the right chunk
        but buries it behind less relevant results. MRR catches this.
    """

    model_name: str
    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    embedding_time_seconds: float = 0.0
    query_time_seconds: float = 0.0
    embedding_dimensions: int = 0
    total_chunks_embedded: int = 0
    failed: bool = False
    error_message: str = ""
    per_question_results: List[Dict] = field(default_factory=list)

    def summary_line(self) -> str:
        """One-line summary for the results table."""
        if self.failed:
            return (
                f"{self.model_name:<50} "
                f"{'FAILED':<10} "
                f"{self.error_message[:40]}"
            )
        return (
            f"{self.model_name:<50} "
            f"R@1: {self.recall_at_1:>5.1f}%  "
            f"R@5: {self.recall_at_5:>5.1f}%  "
            f"MRR: {self.mrr:>5.3f}  "
            f"Dims: {self.embedding_dimensions:>4}  "
            f"Embed: {self.embedding_time_seconds:>5.1f}s"
        )

    def to_dict(self) -> Dict:
        """Converts to dictionary for JSON serialisation."""
        return {
            "model_name": self.model_name,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "mrr": self.mrr,
            "embedding_time_seconds": self.embedding_time_seconds,
            "query_time_seconds": self.query_time_seconds,
            "embedding_dimensions": self.embedding_dimensions,
            "total_chunks_embedded": self.total_chunks_embedded,
            "failed": self.failed,
            "error_message": self.error_message,
            "per_question_results": self.per_question_results,
        }

# DATA LOADERS
def load_chunks(chunked_dir: Path) -> List[Dict]:
    """
    Loads all chunks from the chunked JSON files.
    Returns a flat list of all chunks across all documents.
    """
    all_chunks = []
    chunked_files = list(chunked_dir.glob("*_chunked.json"))

    if not chunked_files:
        raise FileNotFoundError(
            f"No chunked JSON files found in {chunked_dir}\n"
            f"Run chunker.py first."
        )

    logger.info(f"Loading chunks from {len(chunked_files)} documents...")
    for path in sorted(chunked_files):
        with open(path, "r", encoding="utf-8") as f:
            doc_data = json.load(f)
        chunks = doc_data.get("chunks", [])
        all_chunks.extend(chunks)
        logger.info(f"  {doc_data.get('doc_key', 'unknown'):<35} {len(chunks):>4} chunks")

    logger.info(f"Total chunks loaded: {len(all_chunks):,}")
    return all_chunks


def load_golden_dataset(golden_path: Path) -> List[Dict]:
    """
    Loads the golden evaluation pairs from JSON.
    """
    if not golden_path.exists():
        raise FileNotFoundError(
            f"Golden dataset not found at: {golden_path}\n"
            f"Create evaluation/golden_dataset.json first."
        )

    with open(golden_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    logger.info(f"Golden dataset loaded: {len(pairs)} question-answer pairs")
    return pairs

# SIMILARITY CALCULATION
def cosine_similarity_matrix(query_vectors: np.ndarray,
                               doc_vectors: np.ndarray) -> np.ndarray:
    """
    Calculates cosine similarity between query vectors and all document vectors.

    Cosine similarity = dot product of normalised vectors.
    We normalise first then use matrix multiplication — this is much faster
    than calculating similarity one pair at a time.

    Args:
        query_vectors: Shape (num_queries, embedding_dim)
        doc_vectors:   Shape (num_docs, embedding_dim)

    Returns:
        Similarity matrix of shape (num_queries, num_docs)
        Entry [i, j] = similarity between query i and document j
    """

    # Normalise query vectors to unit length.
    query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
    # np.linalg.norm computes vector magnitude. keepdims=True preserves
    # the shape for broadcasting in the division below.
    query_normalised = query_vectors / (query_norms + 1e-10)
    # + 1e-10 prevents division by zero for zero vectors.

    # Normalise document vectors to unit length.
    doc_norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    doc_normalised = doc_vectors / (doc_norms + 1e-10)

    # Matrix multiplication gives cosine similarity for all pairs at once.
    # Shape: (num_queries, num_docs)
    similarity_matrix = np.dot(query_normalised, doc_normalised.T)
    # .T transposes doc_normalised from (num_docs, dim) to (dim, num_docs)
    # so the matrix multiplication works correctly.

    return similarity_matrix

# CORE EVALUATION FUNCTION
def evaluate_model(
    model_name: str,
    chunks: List[Dict],
    golden_pairs: List[Dict],
) -> EvaluationResult:
    """
    Evaluates one embedding model against the golden dataset.

    Steps:
    1. Load the model
    2. Embed all chunks (corpus embedding)
    3. Embed all golden questions (query embedding)
    4. Calculate similarity matrix
    5. For each question, check if correct chunk ranks in top 1 and top 5
    6. Calculate Recall@1, Recall@5, MRR
    7. Unload model and free memory

    Args:
        model_name: HuggingFace model identifier
        chunks: All document chunks (from load_chunks)
        golden_pairs: Golden question-answer pairs (from load_golden_dataset)

    Returns:
        EvaluationResult with all metrics
    """

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*60}")

    result = EvaluationResult(model_name=model_name)

    # --- Step 1: Load model ---
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        result.embedding_dimensions = model.get_sentence_embedding_dimension()
        logger.info(f"  Model loaded | Dimensions: {result.embedding_dimensions}")
    except Exception as e:
        logger.error(f"  Failed to load model: {e}")
        result.failed = True
        result.error_message = str(e)
        return result

    # --- Step 2: Embed all chunks ---
    chunk_texts = [chunk["text"] for chunk in chunks]
    result.total_chunks_embedded = len(chunk_texts)

    logger.info(f"  Embedding {len(chunk_texts):,} chunks...")
    embed_start = time.time()

    try:
        chunk_embeddings = model.encode(
            chunk_texts,
            batch_size=64,
            # batch_size=64: process 64 chunks at a time.
            # Larger batches are faster but use more RAM.
            # 64 is safe for 6GB RAM with most models.
            show_progress_bar=True,
            convert_to_numpy=True,
            # convert_to_numpy=True: return numpy array for cosine_similarity_matrix.
            normalize_embeddings=True,
            # normalize_embeddings=True: pre-normalise vectors.
            # Makes cosine similarity calculation faster (just dot product).
        )
    except Exception as e:
        logger.error(f"  Chunk embedding failed: {e}")
        result.failed = True
        result.error_message = f"Chunk embedding failed: {e}"
        del model
        gc.collect()
        return result

    result.embedding_time_seconds = round(time.time() - embed_start, 2)
    logger.info(f"  Chunks embedded in {result.embedding_time_seconds}s")

    # --- Step 3: Embed all golden questions ---
    questions = [pair["question"] for pair in golden_pairs]

    logger.info(f"  Embedding {len(questions)} golden questions...")
    query_start = time.time()

    try:
        question_embeddings = model.encode(
            questions,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as e:
        logger.error(f"  Question embedding failed: {e}")
        result.failed = True
        result.error_message = f"Question embedding failed: {e}"
        del model
        gc.collect()
        return result

    result.query_time_seconds = round(time.time() - query_start, 3)

    # Unload model immediately — no longer needed.
    # Freeing RAM here is important when testing large models in sequence.
    del model
    gc.collect()
    logger.info(f"  Model unloaded. RAM freed.")

    # --- Step 4: Calculate similarity matrix ---
    logger.info("  Calculating similarities...")
    similarity_matrix = cosine_similarity_matrix(
        question_embeddings,
        chunk_embeddings
    )
    # Shape: (num_questions, num_chunks)
    # similarity_matrix[i][j] = similarity between question i and chunk j

    # --- Step 5 & 6: Score each question ---
    recall_at_1_scores = []
    recall_at_5_scores = []
    reciprocal_ranks = []

    for q_idx, pair in enumerate(golden_pairs):

        # Get similarity scores for this question against all chunks.
        similarities = similarity_matrix[q_idx]

        # Get indices of top 10 chunks, sorted by similarity descending.
        top_10_indices = np.argsort(similarities)[::-1][:10]
        # np.argsort returns indices that would sort the array ascending.
        # [::-1] reverses to descending. [:10] takes top 10.

        # Find the rank of the correct chunk.
        correct_rank = None
        for rank, chunk_idx in enumerate(top_10_indices, start=1):
            # rank starts at 1 (not 0) for human-readable output.
            chunk = chunks[chunk_idx]

            # Check if this chunk matches th e expected answer.
            doc_match = pair["expected_doc"] in chunk.get("doc_key", "")
            section_match = pair["section"] in chunk.get("section_number", "")
            # "section 9" matches "Section 9. Termination of Contract"

            if doc_match and section_match:
                correct_rank = rank
                break

        # Calculate metrics for this question.
        r_at_1 = 1.0 if correct_rank == 1 else 0.0
        r_at_5 = 1.0 if (correct_rank is not None and correct_rank <= 5) else 0.0
        rr = (1.0 / correct_rank) if correct_rank is not None else 0.0
        # Reciprocal Rank: 1/1=1.0, 1/2=0.5, 1/3=0.33, not found=0.0

        recall_at_1_scores.append(r_at_1)
        recall_at_5_scores.append(r_at_5)
        reciprocal_ranks.append(rr)

        # Store per-question detail for the report.
        result.per_question_results.append({
            "question": pair["question"],
            "correct_rank": correct_rank,
            "recall_at_1": r_at_1,
            "recall_at_5": r_at_5,
            "reciprocal_rank": round(rr, 3),
            "top_result_doc": chunks[top_10_indices[0]].get("doc_key", ""),
            "top_result_heading": chunks[top_10_indices[0]].get("heading", "")[:60],
            "top_similarity": round(float(similarities[top_10_indices[0]]), 4),
        })

        # Log each question result.
        status = f"Rank {correct_rank}" if correct_rank else "NOT FOUND"
        logger.info(
            f"  Q{q_idx+1:02d} [{status:>8}] "
            f"Top result: {chunks[top_10_indices[0]].get('doc_key','')[:20]} | "
            f"{chunks[top_10_indices[0]].get('heading','')[:40]}"
        )

    # --- Aggregate metrics ---
    result.recall_at_1 = round(np.mean(recall_at_1_scores) * 100, 2)
    result.recall_at_5 = round(np.mean(recall_at_5_scores) * 100, 2)
    result.mrr = round(float(np.mean(reciprocal_ranks)), 4)
    # np.mean() computes the average. * 100 converts to percentage for R@1/R@5.

    logger.info(f"\n  RESULTS FOR: {model_name}")
    logger.info(f"  Recall@1:  {result.recall_at_1}%")
    logger.info(f"  Recall@5:  {result.recall_at_5}%")
    logger.info(f"  MRR:       {result.mrr}")
    logger.info(f"  Dims:      {result.embedding_dimensions}")
    logger.info(f"  Embed time:{result.embedding_time_seconds}s")

    return result

# REPORT GENERATION
def generate_report(results: List[EvaluationResult]) -> str:
    """
    Generates a human-readable evaluation report.

    Args:
        results: List of EvaluationResult objects from all model evaluations

    Returns:
        Formatted report string
    """

    lines = []
    lines.append("=" * 70)
    lines.append("9jaLawya — Embedding Model Evaluation Report")
    lines.append("=" * 70)
    lines.append(f"Models tested: {len(results)}")
    lines.append(f"Golden pairs:  {len(results[0].per_question_results) if results else 0}")
    lines.append("")

    # Sort by Recall@1 descending (primary), MRR descending (tiebreaker).
    successful = [r for r in results if not r.failed]
    failed = [r for r in results if r.failed]
    successful.sort(key=lambda r: (r.recall_at_1, r.mrr), reverse=True)

    # --- Results table ---
    lines.append("RESULTS (sorted by Recall@1):")
    lines.append("-" * 70)
    lines.append(
        f"{'Rank':<5} {'Model':<42} {'R@1':>6} {'R@5':>6} "
        f"{'MRR':>6} {'Dims':>5} {'Time':>6}"
    )
    lines.append("-" * 70)

    for rank, result in enumerate(successful, 1):
        winner_marker = " ← WINNER" if rank == 1 else ""
        lines.append(
            f"{rank:<5} {result.model_name:<42} "
            f"{result.recall_at_1:>5.1f}% "
            f"{result.recall_at_5:>5.1f}% "
            f"{result.mrr:>6.3f} "
            f"{result.embedding_dimensions:>5} "
            f"{result.embedding_time_seconds:>5.1f}s"
            f"{winner_marker}"
        )

    if failed:
        lines.append("")
        lines.append("FAILED MODELS:")
        for result in failed:
            lines.append(f"  {result.model_name}: {result.error_message[:60]}")

    # --- Winner recommendation ---
    if successful:
        winner = successful[0]
        lines.append("")
        lines.append("=" * 70)
        lines.append("RECOMMENDATION")
        lines.append("=" * 70)
        lines.append(f"Best model: {winner.model_name}")
        lines.append(f"  Recall@1: {winner.recall_at_1}%")
        lines.append(f"  Recall@5: {winner.recall_at_5}%")
        lines.append(f"  MRR:      {winner.mrr}")
        lines.append("")
        lines.append("To use this model in 9jaLawya:")
        lines.append("")
        lines.append("  1. Update EMBEDDING_MODEL_NAME in src/ingestion/ingestor.py:")
        lines.append(f'     EMBEDDING_MODEL_NAME = "{winner.model_name}"')
        lines.append("")
        lines.append("  2. Update EMBEDDING_MODEL_NAME in src/retrieval/retriever.py:")
        lines.append(f'     EMBEDDING_MODEL_NAME = "{winner.model_name}"')
        lines.append("")
        lines.append("  3. Rebuild the vector store:")
        lines.append("     python src/ingestion/ingestor.py --reset")
        lines.append("")
        lines.append("  4. Test retrieval:")
        lines.append("     python src/retrieval/retriever.py")

    # --- Per-question breakdown for winner ---
    if successful:
        winner = successful[0]
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"PER-QUESTION BREAKDOWN — {winner.model_name}")
        lines.append("=" * 70)
        lines.append(
            f"{'Q#':<4} {'Rank':>5} {'R@1':>4} {'MRR':>5}  Question"
        )
        lines.append("-" * 70)
        for i, q in enumerate(winner.per_question_results, 1):
            rank_str = str(q["correct_rank"]) if q["correct_rank"] else "MISS"
            lines.append(
                f"{i:<4} {rank_str:>5} "
                f"{'✓' if q['recall_at_1'] else '✗':>4} "
                f"{q['reciprocal_rank']:>5.2f}  "
                f"{q['question'][:50]}"
            )

    return "\n".join(lines)

# MAIN ORCHESTRATOR
def run_evaluation():
    """
    Runs the full embedding model evaluation pipeline.
    """

    logger.info("9jaLawya — Embedding Model Evaluation")
    logger.info(f"Chunked data: {CHUNKED_DIR}")
    logger.info(f"Golden dataset: {GOLDEN_DATASET_PATH}")
    logger.info(f"Results will be saved to: {RESULTS_DIR}")

    # --- Load data ---
    try:
        chunks = load_chunks(CHUNKED_DIR)
        golden_pairs = load_golden_dataset(GOLDEN_DATASET_PATH)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if len(golden_pairs) < 10:
        logger.warning(
            f"Only {len(golden_pairs)} golden pairs found. "
            f"Evaluation scores may not be reliable. "
            f"Aim for at least 25-30 pairs."
        )

    # --- Run evaluation for each model ---
    all_results = []

    for model_name in MODELS_TO_EVALUATE:
        result = evaluate_model(model_name, chunks, golden_pairs)
        all_results.append(result)

        # Save intermediate results after each model.
        # If the script crashes on a later model, earlier results are safe.
        intermediate_path = RESULTS_DIR / "embedding_results_partial.json"
        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2)
        logger.info(f"  Intermediate results saved to: {intermediate_path.name}")

    # --- Generate and save final report ---
    report = generate_report(all_results)
    print("\n" + report)

    # Save machine-readable JSON results.
    json_path = RESULTS_DIR / "embedding_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2, ensure_ascii=False)

    # Save human-readable text report.
    txt_path = RESULTS_DIR / "embedding_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"\nResults saved:")
    logger.info(f"  JSON: {json_path}")
    logger.info(f"  Report: {txt_path}")

    return all_results


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
