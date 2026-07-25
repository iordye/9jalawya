# src/retrieval/retriever.py
# IMPORTS
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# CONSTANTS
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# Must match exactly the model used in ingestor.py.
# If these differ, query vectors and document vectors live in different
# mathematical spaces and similarity scores become meaningless.

CHROMA_COLLECTION_NAME = "9jalawya_corpus"
# Must match exactly the collection name used in ingestor.py.

DEFAULT_TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

# RETRIEVAL RESULT DATA CLASS
@dataclass
class RetrievalResult:
    """
    Represents a single retrieved legal chunk with all its metadata.

    Using a dataclass instead of a plain dictionary gives us:
    1. Dot notation access: result.heading (not result["heading"])
    2. Type hints on every field for IDE autocomplete and error catching
    3. Automatic __repr__ for clean printing during debugging
    4. Immutability option (frozen=True) if needed later

    Every field here corresponds to a piece of information the LLM
    or agent layer will need to construct a cited legal answer.
    """

    text: str
    source: str

    doc_key: str

    heading: str

    similarity: float

    chunk_type: str

    chunk_index: int

    document_category: str

    document_year: int

    authority: str

    is_continuation: bool = False

    chunk_id: str = ""

    def to_context_string(self) -> str:
        """
        Formats this result as a context block for LLM prompt injection.

        This is the string that will be inserted into the LLM's system
        prompt as retrieved context. The format is designed to:
        1. Clearly label the source so the LLM can cite it accurately
        2. Include the legal text verbatim so the LLM answers from evidence
        3. Be compact enough to fit multiple results in one prompt

        Returns:
            Formatted string ready for LLM context injection
        """
        return (
            f"[SOURCE: {self.source}]\n"
            f"[SECTION: {self.heading}]\n"
            f"[RELEVANCE: {self.similarity:.2f}]\n"
            f"{self.text}"
        )
        

    def to_citation(self) -> str:
        """
        Produces a short citation string for use in answers.

        e.g. "Section 9 of the Labour Act Cap L1 LFN 2004"

        Returns:
            Short citation string
        """
        return f"{self.heading} of the {self.source}"

    def to_dict(self) -> Dict:
        """
        Converts the result to a plain dictionary.

        Useful when serialising results to JSON for logging,
        API responses, or passing to non-Python systems.

        Returns:
            Dictionary with all fields
        """
        return {
            "text": self.text,
            "source": self.source,
            "doc_key": self.doc_key,
            "heading": self.heading,
            "similarity": self.similarity,
            "chunk_type": self.chunk_type,
            "chunk_index": self.chunk_index,
            "document_category": self.document_category,
            "document_year": self.document_year,
            "authority": self.authority,
            "is_continuation": self.is_continuation,
            "chunk_id": self.chunk_id,
        }

# RETRIEVAL FILTER DATA CLASS
@dataclass
class RetrievalFilter:
    """
    Optional filters to narrow retrieval to specific documents or categories.

    When the agent determines a question is specifically about labour law,
    it can pass RetrievalFilter(doc_keys=["labour_act"]) to restrict
    the search to only Labour Act chunks. This improves precision.

    All fields are optional — an empty RetrievalFilter() means no filtering
    (search the entire corpus).

    Example usage:
        # Search only the constitution
        f = RetrievalFilter(doc_keys=["constitution"])

        # Search all acts but not the constitution
        f = RetrievalFilter(document_categories=["act"])

        # Search only labour and electoral acts
        f = RetrievalFilter(doc_keys=["labour_act", "electoral_act"])
    """

    doc_keys: Optional[List[str]] = None
    # Restrict to specific documents by their doc_key identifiers.
    # e.g. ["labour_act"] or ["constitution", "electoral_act"]

    document_categories: Optional[List[str]] = None
    # Restrict by document category: ["constitution"] or ["act"] or both.

    chunk_types: Optional[List[str]] = None
    # Restrict by chunk type: ["section"] to exclude preambles/schedules.

    def to_chroma_where(self) -> Optional[Dict]:
        """
        Converts this filter to a ChromaDB 'where' clause dictionary.

        ChromaDB supports filtering on metadata using a MongoDB-style
        query syntax. This method translates our clean filter object
        into that syntax.

        ChromaDB 'where' clause syntax:
          Single condition:   {"field": {"$eq": "value"}}
          Multiple values:    {"field": {"$in": ["val1", "val2"]}}
          AND of conditions:  {"$and": [condition1, condition2]}

        Returns:
            ChromaDB where clause dict, or None if no filters are set
        """

        conditions = []
        # We build a list of conditions and combine them with $and.

        if self.doc_keys:
            if len(self.doc_keys) == 1:
                conditions.append({"doc_key": {"$eq": self.doc_keys[0]}})
                # $eq: equal to — exact match on a single value.
            else:
                conditions.append({"doc_key": {"$in": self.doc_keys}})
                # $in: matches any value in the list.

        if self.document_categories:
            if len(self.document_categories) == 1:
                conditions.append(
                    {"document_category": {"$eq": self.document_categories[0]}}
                )
            else:
                conditions.append(
                    {"document_category": {"$in": self.document_categories}}
                )

        if self.chunk_types:
            if len(self.chunk_types) == 1:
                conditions.append({"chunk_type": {"$eq": self.chunk_types[0]}})
            else:
                conditions.append({"chunk_type": {"$in": self.chunk_types}})

        # Return None if no conditions (means no filtering).
        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]
            # A single condition does not need to be wrapped in $and.

        return {"$and": conditions}
        # Multiple conditions are combined with $and —
        # ALL conditions must be satisfied for a chunk to be returned.

# 9jaLAWya RETRIEVER CLASS
class NaijaLawRetriever:
    """
    The core retrieval engine for the NaijaLaw RAG system.

    This class manages the connection to ChromaDB and provides a clean
    .retrieve() method that the agent layer calls to get relevant legal
    chunks for any given query.

    WHY A CLASS instead of standalone functions?
    Because the retriever needs to maintain STATE between calls:
      - The ChromaDB client connection (expensive to re-open each call)
      - The embedding model (80MB loaded into memory once)
    A class lets us initialise these once in __init__ and reuse them
    across many .retrieve() calls — essential for a responsive system.

    Usage:
        retriever = NaijaLawRetriever()
        results = retriever.retrieve("What are employee rights?")
        for result in results:
            print(result.to_context_string())
    """

    def __init__(
        self,
        vectorstore_dir: Path = VECTORSTORE_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL_NAME,
    ):
        """
        Initialises the retriever by loading the model and connecting to ChromaDB.

        This runs ONCE when the retriever is created. All subsequent
        .retrieve() calls reuse the loaded model and open connection.

        Args:
            vectorstore_dir: Path to the ChromaDB database directory
            collection_name: Name of the ChromaDB collection to query
            embedding_model: HuggingFace model name for query embedding
        """

        logger.info("Initialising NaijaLawRetriever...")

        # --- Validate vector store exists ---
        if not vectorstore_dir.exists():
            raise FileNotFoundError(
                f"Vector store not found at: {vectorstore_dir}\n"
                f"Please run src/ingestion/ingestor.py first to build the vector store."
            )
            # We raise FileNotFoundError (not return None) because without
            # the vector store, the retriever is completely non-functional.
            # A loud crash with a clear message is better than silent failure.

        # --- Load embedding model ---
        logger.info(f"  Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        # Load the same model used during ingestion.
        # Stored as self.model so all .retrieve() calls share one instance.

        # --- Connect to ChromaDB ---
        logger.info(f"  Connecting to ChromaDB: {vectorstore_dir}")
        self.client = chromadb.PersistentClient(path=str(vectorstore_dir))

        # Attach the same embedding function used during ingestion.
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )

        # Get the existing collection — do NOT create if missing.
        # If the collection doesn't exist, ingestion hasn't been run yet.
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=embedding_fn,
            )
            # get_collection (not get_or_create_collection) raises an error
            # if the collection doesn't exist. This is intentional —
            # we want a clear error message, not a silent empty collection.

        except Exception as e:
            raise RuntimeError(
                f"Collection '{collection_name}' not found in ChromaDB.\n"
                f"Please run src/ingestion/ingestor.py first.\n"
                f"Original error: {e}"
            )

        doc_count = self.collection.count()
        logger.info(f"  Connected to collection '{collection_name}'")
        logger.info(f"  Documents in store: {doc_count:,}")

        if doc_count == 0:
            logger.warning(
                "  WARNING: Collection is empty. "
                "Run ingestor.py to populate the vector store."
            )

        logger.info("  NaijaLawRetriever ready.")


    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[RetrievalFilter] = None,
        similarity_threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[RetrievalResult]:
        """
        The main retrieval method. Converts a query to a vector and finds
        the most semantically similar legal chunks in the vector store.

        This is the method the agent layer will call on every user question.

        Args:
            query: Natural language question from the user
                   e.g. "What are the grounds for terminating employment?"

            top_k: Number of results to return (default: 5)
                   The agent may request more (e.g. 10) for complex questions
                   or fewer (e.g. 3) for simple factual lookups.

            filters: Optional RetrievalFilter to restrict the search.
                     e.g. RetrievalFilter(doc_keys=["labour_act"])
                     Default None means search the entire corpus.

            similarity_threshold: Minimum similarity score to include.
                                   Results below this are filtered out.

        Returns:
            List of RetrievalResult objects, sorted by similarity (highest first).
            Returns empty list if no relevant results found above threshold.
        """

        if not query or not query.strip():
            logger.warning("Empty query received — returning no results.")
            return []

        query = query.strip()
        start_time = time.time()

        logger.info(f"Query: '{query[:80]}{'...' if len(query) > 80 else ''}'")
        # Log the query, truncated to 80 chars for readable logs.

        # --- Build ChromaDB where clause from filters ---
        where_clause = None
        if filters:
            where_clause = filters.to_chroma_where()
            if where_clause:
                logger.info(f"  Filter applied: {where_clause}")

        # --- Query ChromaDB ---
        try:
            query_params = {
                "query_texts": [query],
                # query_texts: ChromaDB auto-embeds this using the
                # collection's embedding_function. The embedding happens
                # here internally — we don't need to call model.encode()
                # manually for queries (only documents need manual encoding).

                "n_results": min(top_k, self.collection.count()),
                # n_results: how many results to return.
                # min() guard: can't request more results than exist in DB.

                "include": ["documents", "metadatas", "distances"],
                # What to include in the response:
                # "documents" → the chunk text strings
                # "metadatas" → the metadata dicts (heading, doc_key, etc.)
                # "distances" → cosine distances (we convert to similarity)
            }

            if where_clause:
                query_params["where"] = where_clause
                # Only add 'where' key if we have filters.
                # Passing where=None to ChromaDB causes an error.

            raw_results = self.collection.query(**query_params)
            # **query_params unpacks the dictionary as keyword arguments.
            # Equivalent to: self.collection.query(query_texts=[...], ...)

        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        # --- Parse and structure results ---
        results = self._parse_results(raw_results, similarity_threshold)

        elapsed = round(time.time() - start_time, 3)
        logger.info(
            f"  Retrieved {len(results)} results "
            f"(above threshold {similarity_threshold}) in {elapsed}s"
        )

        # Log a brief summary of top results for debugging.
        for i, r in enumerate(results[:3], 1):
            # enumerate(results[:3], 1) gives (1, result), (2, result), (3, result)
            # We only log the top 3 to keep output manageable.
            logger.info(
                f"  [{i}] {r.similarity:.3f} | "
                f"{r.doc_key} | "
                f"{r.heading[:50]}"
            )

        return results


    def _parse_results(
        self,
        raw_results: Dict,
        similarity_threshold: float
    ) -> List[RetrievalResult]:
        """
        Converts raw ChromaDB query output into RetrievalResult objects.

        ChromaDB returns results in this nested structure:
        {
          "documents": [["text1", "text2", ...]],   ← note: nested list
          "metadatas": [[{...}, {...}, ...]],         ← nested list
          "distances": [[0.15, 0.23, ...]]            ← nested list
        }
        The outer list is per-query (we only send one query at a time,
        so we always take index [0] to get the inner list).

        Args:
            raw_results: Raw dictionary returned by collection.query()
            similarity_threshold: Minimum similarity to include

        Returns:
            List of RetrievalResult objects, filtered and sorted
        """

        results = []

        # Extract the inner lists (index [0] because we sent one query).
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        # zip() pairs up corresponding elements from all three lists.
        # zip([doc1, doc2], [meta1, meta2], [dist1, dist2])
        # → [(doc1, meta1, dist1), (doc2, meta2, dist2)]
        for text, metadata, distance in zip(documents, metadatas, distances):

            # Convert cosine distance to cosine similarity.
            # ChromaDB returns DISTANCE (lower = more similar).
            # We convert to SIMILARITY (higher = more similar) because
            # similarity is more intuitive: 0.85 means "85% similar".
            similarity = round(1.0 - distance, 4)
            # round(..., 4) keeps 4 decimal places: 0.84723 → 0.8472

            # Apply similarity threshold filter.
            if similarity < similarity_threshold:
                continue
                # Skip results that are not relevant enough.
                # 'continue' moves to the next iteration without adding
                # this result to our list.

            if not text or not text.strip():
                continue
                # Skip empty text results (should never happen but
                # defensive programming prevents silent bad results).

            # Build a RetrievalResult from this raw result.
            result = RetrievalResult(
                text=text,
                source=metadata.get("full_document_name", metadata.get("doc_key", "Unknown")),
                doc_key=metadata.get("doc_key", ""),
                heading=metadata.get("heading", "Unknown Section"),
                similarity=similarity,
                chunk_type=metadata.get("chunk_type", ""),
                chunk_index=metadata.get("chunk_index", 0),
                document_category=metadata.get("document_category", ""),
                document_year=metadata.get("document_year", 0),
                authority=metadata.get("authority", ""),
                is_continuation=metadata.get("is_continuation", False),
                chunk_id=metadata.get("chunk_id", ""),
            )

            CONTINUATION_PENALTY = 0.05

        def sort_key(r: RetrievalResult) -> float:
            return r.similarity - (CONTINUATION_PENALTY if r.is_continuation else 0.0)

        results.sort(key=sort_key, reverse=True)
        return results


    def retrieve_with_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filters: Optional[RetrievalFilter] = None,
    ) -> str:
        """
        Retrieves chunks and formats them as a single context string
        ready for injection into an LLM prompt.

        This is a convenience method for the agent layer — instead of
        calling .retrieve() and manually formatting each result,
        the agent can call this and get a ready-to-use context block.

        The returned string looks like:
        ---
        [SOURCE: Labour Act Cap L1 LFN 2004]
        [SECTION: Section 9. Termination of Contract]
        [RELEVANCE: 0.85]
        Either party to a contract of employment may terminate...

        [SOURCE: Constitution of Nigeria 1999]
        [SECTION: Section 36. Right to Fair Hearing]
        [RELEVANCE: 0.71]
        Every person shall be entitled to a fair hearing...
        ---

        Args:
            query: User's natural language question
            top_k: Number of chunks to retrieve
            filters: Optional document filters

        Returns:
            Formatted multi-chunk context string for LLM prompt injection
        """

        results = self.retrieve(query, top_k=top_k, filters=filters)

        if not results:
            return "No relevant legal provisions found for this query."

        # Join all results into one context block separated by dividers.
        context_blocks = [r.to_context_string() for r in results]
        return "\n\n---\n\n".join(context_blocks)
        # "\n\n---\n\n" creates a clear visual separator between chunks
        # that the LLM can use to distinguish between different sources.


    def get_collection_stats(self) -> Dict:
        """
        Returns statistics about the current state of the vector store.

        Useful for health checks and debugging — tells you how many
        chunks are stored and how they are distributed across documents.

        Returns:
            Dictionary with collection statistics
        """

        total = self.collection.count()

        # Get a sample of all metadata to count per-document chunks.
        # We use get() with a large limit to fetch metadata without vectors.
        try:
            all_meta = self.collection.get(
                include=["metadatas"],
                limit=total,
                # limit=total fetches all records' metadata.
            )

            doc_counts: Dict[str, int] = {}
            for meta in all_meta["metadatas"]:
                key = meta.get("doc_key", "unknown")
                doc_counts[key] = doc_counts.get(key, 0) + 1
                # .get(key, 0) returns the current count or 0 if not seen yet.
                # We then add 1 to it.

        except Exception:
            doc_counts = {}

        return {
            "total_chunks": total,
            "collection_name": CHROMA_COLLECTION_NAME,
            "vectorstore_dir": str(VECTORSTORE_DIR),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "chunks_per_document": doc_counts,
        }

# INTERACTIVE TEST RUNNER
def run_interactive_test(retriever: NaijaLawRetriever) -> None:
    """
    Runs an interactive command-line session for testing retrieval quality.

    Type any legal question and see the top retrieved chunks with their
    similarity scores. Type 'quit' to exit.

    This is how you manually evaluate whether your RAG retrieval is
    working correctly before connecting it to an LLM.

    Args:
        retriever: An initialised NaijaLawRetriever instance
    """

    print("\n" + "="*60)
    print("NaijaLaw Retrieval Test — Interactive Mode")
    print("="*60)
    print("Type a legal question to retrieve relevant chunks.")
    print("Commands:")
    print("  'stats'        — show vector store statistics")
    print("  'filter <key>' — filter next query to a doc_key")
    print("                   e.g. 'filter labour_act'")
    print("  'quit'         — exit")
    print("="*60 + "\n")

    active_filter = None
    # Holds the current filter if the user set one with 'filter <key>'.

    while True:
        try:
            user_input = input("Query > ").strip()
            # input() displays the prompt and waits for user to press Enter.
            # .strip() removes any accidental leading/trailing whitespace.

        except (KeyboardInterrupt, EOFError):
            # KeyboardInterrupt: user pressed Ctrl+C
            # EOFError: input stream ended (e.g. piped input finished)
            print("\nExiting.")
            break

        if not user_input:
            continue
            # Empty input — just show the prompt again.

        if user_input.lower() == "quit":
            print("Goodbye.")
            break

        if user_input.lower() == "stats":
            stats = retriever.get_collection_stats()
            print(f"\nVector Store Statistics:")
            print(f"  Total chunks:    {stats['total_chunks']:,}")
            print(f"  Embedding model: {stats['embedding_model']}")
            print(f"\n  Chunks per document:")
            for doc, count in sorted(stats["chunks_per_document"].items()):
                print(f"    {doc:<35} {count:>4} chunks")
            print()
            continue

        if user_input.lower().startswith("filter "):
            doc_key = user_input[7:].strip()
            # [7:] slices off "filter " (7 characters) to get just the key.
            active_filter = RetrievalFilter(doc_keys=[doc_key])
            print(f"  Filter set to: {doc_key}")
            print(f"  (Next query will search only '{doc_key}')")
            print(f"  (Type 'filter off' to remove the filter)\n")
            continue

        if user_input.lower() == "filter off":
            active_filter = None
            print("  Filter removed — searching full corpus.\n")
            continue

        # --- Run the retrieval ---
        print()
        results = retriever.retrieve(
            query=user_input,
            top_k=DEFAULT_TOP_K,
            filters=active_filter,
        )

        if not results:
            print("  No relevant results found above similarity threshold.")
            print("  Try rephrasing your question or lowering the threshold.\n")
            continue

        # Display results in a readable format.
        print(f"  Found {len(results)} relevant chunk(s):\n")
        for i, result in enumerate(results, 1):
            print(f"  {'─'*56}")
            print(f"  [{i}] Similarity: {result.similarity:.4f}")
            print(f"      Source:     {result.source}")
            print(f"      Section:    {result.heading}")
            print(f"      Type:       {result.chunk_type}")
            if result.is_continuation:
                print(f"      Note:       This is a continuation chunk")
            print(f"\n      Text preview:")
            # Display first 300 chars of the chunk text, word-wrapped.
            preview = result.text[:300]
            # Indent the preview with 6 spaces for readability.
            indented = "\n      ".join(
                preview[j:j+70] for j in range(0, len(preview), 70)
                # Split into 70-char lines for terminal readability.
            )
            print(f"      {indented}...")
            print()

        # Clear the active filter after each query (one-shot filtering).
        if active_filter:
            print(f"  (Filter '{active_filter.doc_keys}' was active for this query)\n")
            active_filter = None

# ENTRY POINT
if __name__ == "__main__":
    # Initialise the retriever — loads model and connects to ChromaDB.
    try:
        retriever = NaijaLawRetriever()
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        exit(1)
        # exit(1) terminates the script with error code 1.
        # Code 0 means success, any non-zero means failure.
        # This allows shell scripts to detect whether we failed.

    # Print collection statistics on startup.
    stats = retriever.get_collection_stats()
    logger.info(f"\nVector store ready:")
    logger.info(f"  Total chunks: {stats['total_chunks']:,}")
    logger.info(f"  Documents:")
    for doc, count in sorted(stats["chunks_per_document"].items()):
        logger.info(f"    {doc:<35} {count:>4} chunks")

    # Launch the interactive test session.
    run_interactive_test(retriever)
