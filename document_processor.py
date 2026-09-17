"""
============================================================
  DOCUMENT PROCESSOR  (app/document_processor.py)
============================================================

PURPOSE:
  Reads your raw text files and cuts them into small, overlapping
  pieces called "chunks". These chunks are what gets embedded and
  stored in the vector database.

WHY DO WE CHUNK?
  ┌─────────────────────────────────────────────────────────────┐
  │  LLMs have token limits — they can't process an entire      │
  │  book at once. Chunking lets us:                            │
  │                                                             │
  │  1. Store precise sub-topics separately (better retrieval)  │
  │  2. Fit within embedding model token limits (~256 tokens)   │
  │  3. Return only the RELEVANT section, not the whole doc     │
  └─────────────────────────────────────────────────────────────┘

OVERLAP EXPLAINED:
  Imagine chunk_size=500 chars, overlap=100 chars:

  [Chunk 1]  chars   0 → 500
  [Chunk 2]  chars 400 → 900   ← starts 100 chars BEFORE Chunk 1 ends
  [Chunk 3]  chars 800 → 1300
                    ↑
              These 100 chars appear in BOTH Chunk 1 and Chunk 2,
              so a rule that straddles a boundary is never lost.

CHUNK SIZE TUNING:
  • chunk_size=600  → ~120 words → good for regulation paragraphs
  • overlap=100     → ~20 words overlap → enough for context continuity
  • Too large:  retrieval returns one broad blob, not the specific rule
  • Too small:  each chunk loses context, answers feel fragmented
"""

import os
from typing import List
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────
@dataclass
class Document:
    """
    A single chunk of text with its origin metadata.

    Attributes:
        text     : The actual chunk text (what gets embedded + shown to Claude).
        metadata : A dict with WHERE this chunk came from.
                   Shown to users as "source cards" in the UI.
    """
    text: str
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────
def load_text_file(filepath: str) -> str:
    """
    Read an entire .txt file as a Python string.
    utf-8 handles Indian language characters and special symbols.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def chunk_text(
    text: str,
    source_name: str,
    chunk_size: int = 600,
    overlap: int = 100,
) -> List[Document]:
    """
    Split one large string into overlapping Document chunks.

    Args:
        text        : Full text of one document/file.
        source_name : Filename used in metadata for attribution.
        chunk_size  : Characters per chunk (~120 words per chunk).
        overlap     : Characters to re-use from the previous chunk.

    Returns:
        List of Document objects, one per chunk.

    STEP-BY-STEP ALGORITHM:
        1.  start = 0
        2.  end   = start + chunk_size
        3.  Trim end backward to the last "." in the second half
            → chunks end at sentence boundaries, not mid-sentence
        4.  Save chunk as Document(text, metadata)
        5.  start = end - overlap   (creates the overlap)
        6.  Repeat until start >= len(text)
    """
    chunks: List[Document] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size

        # ── Sentence-boundary trimming ────────────────────────
        # Don't cut in the middle of a sentence. Look for the
        # last "." in the second half of the chunk window.
        if end < len(text):
            search_window = text[start + chunk_size // 2 : end]
            last_period = search_window.rfind(".")
            if last_period != -1:
                end = start + chunk_size // 2 + last_period + 1

        chunk_text_raw = text[start:end].strip()

        # Skip micro-chunks (just whitespace or a few stray chars)
        if len(chunk_text_raw) > 80:
            chunks.append(
                Document(
                    text=chunk_text_raw,
                    metadata={
                        "source": source_name,      # e.g. "bangalore_bylaws.txt"
                        "chunk_index": chunk_index,
                        "char_start": start,
                        "char_end": end,
                    },
                )
            )
            chunk_index += 1

        # Advance pointer — step back `overlap` chars to create continuity
        start = end - overlap

    return chunks


def load_documents(data_dir: str) -> List[Document]:
    """
    Load ALL .txt files from a directory → return a flat list of chunks.

    This is the single entry point used by both ingest.py and the RAG engine.

    Args:
        data_dir : Folder containing your .txt knowledge files.

    Returns:
        All chunks from all files, in one combined list.

    EXAMPLE OUTPUT (3 files → 115 total chunks):
        data/national_building_code.txt  → 42 chunks
        data/bangalore_bylaws.txt        → 38 chunks
        data/permit_procedures.txt       → 35 chunks
        ──────────────────────────────────────────────
        returns  [chunk_0, chunk_1, ..., chunk_114]
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_documents: List[Document] = []
    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    if not txt_files:
        raise ValueError(f"No .txt files found in {data_dir}")

    for filename in sorted(txt_files):
        filepath = os.path.join(data_dir, filename)
        print(f"  Loading: {filename}")
        raw_text = load_text_file(filepath)
        doc_chunks = chunk_text(raw_text, source_name=filename)
        all_documents.extend(doc_chunks)
        print(f"    → {len(doc_chunks)} chunks created")

    print(f"\nTotal chunks ready for embedding: {len(all_documents)}")
    return all_documents
