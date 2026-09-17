"""
============================================================
  TEXT EMBEDDER  (app/embedder.py)
============================================================

PURPOSE:
  Converts text (sentences, paragraphs, questions) into numeric
  vectors called "embeddings". These vectors capture the MEANING
  of the text, not just the words.

WHAT IS AN EMBEDDING?
  Think of it like GPS coordinates for meaning:

  "What setbacks are needed in Bangalore?"
                          ↓ embed()
  [-0.042, 0.831, -0.210, ..., 0.073]   ← 384 numbers

  "How much space must I leave from the boundary in Bengaluru?"
                          ↓ embed()
  [-0.039, 0.829, -0.198, ..., 0.071]   ← nearly identical numbers!

  Different words, same meaning → nearly identical vectors.
  This is the magic that makes RAG smarter than keyword search.

WHY all-MiniLM-L6-v2?
  ╔══════════════════════════╤══════════╤══════════╤═══════════╗
  ║ Model                    │ Size     │ Speed    │ Quality   ║
  ╠══════════════════════════╪══════════╪══════════╪═══════════╣
  ║ all-MiniLM-L6-v2         │  80 MB   │  fast    │  good     ║  ← We use this
  ║ all-mpnet-base-v2        │ 420 MB   │  medium  │  better   ║
  ║ text-embedding-3-small   │  API     │  fast    │  great    ║  (OpenAI, paid)
  ╚══════════════════════════╧══════════╧══════════╧═══════════╝

  For building code questions, all-MiniLM-L6-v2 is more than good enough,
  runs on CPU (no GPU required), and is completely free.

HOW IT WORKS INTERNALLY:
  text → tokenize → 6-layer transformer → pool → normalize → 384-dim vector
                                                   ↑
                                    L2-normalize so cosine similarity works:
                                    cos(a,b) = dot(a,b) when |a|=|b|=1
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    Wraps SentenceTransformer to embed both documents (at ingestion time)
    and queries (at retrieval time).

    CRITICAL: Always use the SAME model for both.
    If you embedded docs with Model A but query with Model B,
    the vectors live in different mathematical "spaces" and
    similarity scores will be meaningless garbage.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DIMENSION = 384   # Number of dimensions in each output vector

    def __init__(self):
        """
        Load the model. On first run, downloads ~80 MB from HuggingFace Hub.
        After that, loads from local cache (~/.cache/huggingface/hub/) in ~2s.
        """
        print(f"[Embedder] Loading model: {self.MODEL_NAME} ...")
        self.model = SentenceTransformer(self.MODEL_NAME)
        print(f"[Embedder] Ready. Output dimension: {self.DIMENSION}")

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of document chunks — called once during ingestion.

        Args:
            texts : List of chunk strings.

        Returns:
            np.ndarray of shape (N, 384) — one row per chunk.

        WHY BATCH?
            Passing all texts at once is 5–10× faster than a loop because
            the model can parallelize across the batch on CPU/GPU.
            show_progress_bar gives a visual during long ingestions.
        """
        print(f"[Embedder] Embedding {len(texts)} chunks (batch mode)...")
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,   # L2-norm for correct cosine similarity
        )

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user question — called on every /ask request.

        Args:
            query : The user's question string.

        Returns:
            np.ndarray of shape (384,) — a single vector.

        SPEED: ~5 ms on CPU. Fast enough for real-time use.
        """
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding[0]   # Return (384,) not (1, 384)

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Cosine similarity between two normalized vectors.
        Range: -1.0 (opposite) to +1.0 (identical meaning).
        For L2-normalized vectors: similarity = dot(a, b)

        Mostly used for debugging/testing; ChromaDB computes this internally.
        """
        return float(np.dot(a, b))
