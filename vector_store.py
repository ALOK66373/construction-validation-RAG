"""
============================================================
  VECTOR STORE  (app/vector_store.py)
============================================================

PURPOSE:
  Stores document embeddings on disk and performs fast semantic
  similarity search to find the most relevant chunks for a query.

REGULAR DB vs VECTOR DB:
  ┌─────────────────────────────────────────────────────────────┐
  │  SQL (keyword):                                             │
  │    SELECT * WHERE text LIKE '%setback%'                     │
  │    Finds "setback" only — misses "boundary distance"        │
  │                                                             │
  │  Vector DB (semantic):                                      │
  │    search(embed("boundary distance"))                       │
  │    → returns chunks about "setback rules"                   │
  │    Even though the words don't match!                       │
  └─────────────────────────────────────────────────────────────┘

WHY ChromaDB?
  • 100% free and open-source
  • Runs locally (no cloud account, no API cost)
  • Persists to disk → survives server restarts
  • Simple Python API
  • Handles up to ~1M chunks comfortably

HOW CHROMA STORES DATA:
  For each chunk, ChromaDB saves three things in ./chroma_db/:
    1. Text string         → shown in source cards + given to Claude
    2. Embedding vector    → used for similarity search
    3. Metadata dict       → source filename, chunk index

COSINE DISTANCE vs SIMILARITY:
  ChromaDB returns "distance" (not "similarity").
  For cosine metric:   distance = 1 − similarity
  So:                  similarity = 1 − distance
  We convert this so the UI shows intuitive 0–100% relevance scores.

HNSW ALGORITHM (inside ChromaDB):
  Finding nearest vectors in a 384-dim space naively = O(N) comparisons.
  HNSW (Hierarchical Navigable Small World) = O(log N) using a layered graph.
  At 100 chunks it doesn't matter, but at 100,000 chunks it's 10,000× faster.
"""

from typing import List, Tuple, Optional
import chromadb

from document_processor import Document
from embedder import TextEmbedder


class VectorStore:
    """
    A clean wrapper around ChromaDB for storing and searching embeddings.
    """

    COLLECTION_NAME = "building_codes"

    def __init__(self, persist_dir: str = "./chroma_db"):
        """
        Connect to ChromaDB with disk persistence.

        Args:
            persist_dir : Where ChromaDB writes its files.
                          Created on first run, loaded on subsequent runs.

        FIRST RUN:  Creates ./chroma_db/ directory and empty database.
        LATER RUNS: Loads existing data — no re-ingestion needed!
        """
        print(f"[VectorStore] Connecting to ChromaDB at: {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)

        # get_or_create_collection:
        #   Already exists → load it (data intact from previous run)
        #   Doesn't exist → create empty collection
        # hnsw:space=cosine → use cosine as the distance metric
        #   (correct for L2-normalized embeddings from SentenceTransformer)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        print(f"[VectorStore] Loaded. Chunk count: {self.collection.count()}")

    # ── WRITE ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document], embedder: TextEmbedder) -> None:
        """
        Embed all documents and store them in ChromaDB.

        Args:
            documents : List of Document objects from document_processor.
            embedder  : TextEmbedder instance (MUST be the same model as used for queries).

        FLOW:
            documents[].text
                ↓  embedder.embed_documents()
            embeddings  shape=(N, 384)
                ↓  collection.add()
            ChromaDB stores (text, embedding, metadata) on disk

        IDs must be unique strings per collection.
        We use "doc_0", "doc_1", etc.
        """
        if not documents:
            print("[VectorStore] No documents to add.")
            return

        texts     = [doc.text          for doc in documents]
        metadatas = [doc.metadata      for doc in documents]
        ids       = [f"doc_{i}"        for i in range(len(documents))]

        # Generate all embeddings in one efficient batch
        embeddings = embedder.embed_documents(texts)   # shape: (N, 384)

        print(f"[VectorStore] Storing {len(documents)} chunks...")
        self.collection.add(
            documents  = texts,
            embeddings = embeddings.tolist(),   # ChromaDB needs Python lists, not numpy
            metadatas  = metadatas,
            ids        = ids,
        )
        print(f"[VectorStore] Done. Total chunks stored: {self.collection.count()}")

    def clear(self) -> None:
        """
        Delete all chunks and reset the collection.
        Called by ingest.py --force and /ingest?force=true.

        We delete + recreate rather than deleting individual documents
        because it's simpler and faster.
        """
        print("[VectorStore] Clearing existing data...")
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("[VectorStore] Collection reset.")

    # ── READ ───────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: "np.ndarray",
        n_results: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Tuple[str, dict, float]]:
        """
        Find the top-k chunks most semantically similar to the query.

        Args:
            query_embedding : Embedded user query — shape (384,).
            n_results       : How many chunks to return (top-k).
            source_filter   : Optional filename to restrict search to one file.
                              e.g., source_filter="bangalore_bylaws.txt"

        Returns:
            List of (chunk_text, metadata_dict, similarity_score).
            Sorted descending by similarity (most relevant first).
            similarity_score ∈ [0, 1] — closer to 1 = more relevant.

        CHROMADB INTERNAL FLOW:
            1. Receive query_embedding as a 384-dim vector
            2. Run HNSW graph traversal to find k nearest neighbors
            3. Return distances (cosine metric) for those k neighbors
            4. We convert distance → similarity = 1 - distance
        """
        available = self.collection.count()
        if available == 0:
            return []

        k = min(n_results, available)   # Can't request more than we have

        where_filter = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_embeddings = [query_embedding.tolist()],   # List of lists (batch API)
            n_results        = k,
            where            = where_filter,
            include          = ["documents", "metadatas", "distances"],
        )

        # ChromaDB returns batch results — index [0] = our single query
        texts     = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # distance ∈ [0, 2] for unnormalized cosine, but our embeddings are
        # L2-normalized so distance ∈ [0, 1] → similarity = 1 - distance
        return [
            (text, meta, round(1.0 - dist, 4))
            for text, meta, dist in zip(texts, metadatas, distances)
        ]

    # ── UTILITY ────────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        """True if no documents have been ingested yet."""
        return self.collection.count() == 0

    def count(self) -> int:
        """Total number of chunks currently stored."""
        return self.collection.count()
