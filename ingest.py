"""
============================================================
  DOCUMENT INGESTION SCRIPT  (ingest.py)
============================================================

PURPOSE:
  Run this ONCE before starting the server to load your building
  code documents into ChromaDB. The server also auto-ingests on
  first start, but running this manually gives you more control
  and progress visibility.

WHEN TO RUN:
  • First-time setup (no chroma_db/ folder exists)
  • When you add or edit .txt files in data/
  • When you want to fully reset the knowledge base

USAGE:
  python ingest.py              # Ingest (skips if already done)
  python ingest.py --force      # Force re-ingest (clears existing data)
  python ingest.py --preview    # Show chunk statistics, don't store
  python ingest.py --data-dir ./my_docs --chroma-dir ./my_db

WHAT THIS SCRIPT DOES:
  1. Reads all .txt files from data/
  2. Splits them into ~600-character overlapping chunks
  3. Embeds each chunk with all-MiniLM-L6-v2 (384-dim vectors)
  4. Stores (text, embedding, metadata) in ChromaDB at chroma_db/
  5. ChromaDB auto-persists to disk — no re-run needed after restart

AFTER RUNNING:
  Start server:  uvicorn app.main:app --reload --port 8000
  Open UI:       frontend/index.html
  View API docs: http://localhost:8000/docs
"""

import sys
import os
import argparse
import time
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from document_processor import load_documents
from embedder import TextEmbedder
from vector_store import VectorStore


def print_banner():
    print("\n" + "=" * 60)
    print("  Building Code Advisor — Document Ingestion")
    print("=" * 60)


def check_data_directory(data_dir: str) -> list:
    """Verify the data/ directory exists and contains .txt files."""
    if not os.path.exists(data_dir):
        print(f"\n❌  Data directory '{data_dir}' not found!")
        print("    Create a data/ folder and add your .txt knowledge files.")
        sys.exit(1)

    txt_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if not txt_files:
        print(f"\n❌  No .txt files found in '{data_dir}'")
        print("    Add your building code .txt files to the data/ folder.")
        sys.exit(1)

    print(f"\n📁  Found {len(txt_files)} document(s) in '{data_dir}':")
    for f in sorted(txt_files):
        size_kb = os.path.getsize(os.path.join(data_dir, f)) / 1024
        print(f"    • {f}  ({size_kb:.1f} KB)")

    return txt_files


def preview_chunks(documents: list) -> None:
    """Print statistics without writing to ChromaDB."""
    print(f"\n📊  Chunk Statistics:")
    print(f"    Total chunks: {len(documents)}")

    by_source = defaultdict(list)
    for doc in documents:
        by_source[doc.metadata["source"]].append(doc)

    for source, docs in sorted(by_source.items()):
        avg_len = sum(len(d.text) for d in docs) / len(docs)
        print(f"    {source}: {len(docs)} chunks  (avg {avg_len:.0f} chars each)")

    print("\n📝  Sample chunks (first 3):")
    for i, doc in enumerate(documents[:3]):
        print(f"\n    [Chunk {i} from {doc.metadata['source']}]")
        print(f"    {doc.text[:200]}...")

    print("\n✅  Preview done. Run without --preview to store to ChromaDB.")


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Ingest building code documents into ChromaDB"
    )
    parser.add_argument("--force",     action="store_true", help="Clear and re-ingest")
    parser.add_argument("--preview",   action="store_true", help="Show stats, don't store")
    parser.add_argument("--data-dir",  default="./data",     help="Folder with .txt files")
    parser.add_argument("--chroma-dir",default="./chroma_db",help="ChromaDB storage path")
    args = parser.parse_args()

    # ── Step 1: Verify data directory ────────────────────────────────────────
    check_data_directory(args.data_dir)

    # ── Step 2: Load and chunk documents ─────────────────────────────────────
    print(f"\n📄  Step 1/3: Loading and chunking documents...")
    t0 = time.time()
    documents = load_documents(args.data_dir)
    print(f"    ✅ {len(documents)} chunks created in {time.time() - t0:.1f}s")

    if args.preview:
        preview_chunks(documents)
        return

    # ── Step 3: Load embedding model ─────────────────────────────────────────
    print(f"\n🤖  Step 2/3: Loading embedding model (all-MiniLM-L6-v2)...")
    print("    (First run downloads ~80 MB — cached after that)")
    t0 = time.time()
    embedder = TextEmbedder()
    print(f"    ✅ Model loaded in {time.time() - t0:.1f}s")

    # ── Step 4: Connect to ChromaDB ───────────────────────────────────────────
    print(f"\n💾  Step 3/3: Connecting to ChromaDB at '{args.chroma_dir}'...")
    vector_store = VectorStore(persist_dir=args.chroma_dir)

    if not vector_store.is_empty() and not args.force:
        count = vector_store.count()
        print(f"\n⚠️   ChromaDB already contains {count} chunks.")
        print("    To re-ingest:  python ingest.py --force")
        print("    To preview:    python ingest.py --preview")
        return

    if args.force and not vector_store.is_empty():
        print("    🗑️  Clearing existing data (--force)...")
        vector_store.clear()

    # ── Step 5: Embed + store ─────────────────────────────────────────────────
    print(f"\n    Generating embeddings for {len(documents)} chunks...")
    print("    (May take 30–120 seconds on first run)")
    t0 = time.time()
    vector_store.add_documents(documents, embedder)
    elapsed = time.time() - t0

    # ── Done ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  ✅  Ingestion Complete!")
    print(f"  • Chunks stored : {vector_store.count()}")
    print(f"  • Time taken    : {elapsed:.1f}s")
    print(f"  • Storage path  : {args.chroma_dir}/")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Start the server:  uvicorn app.main:app --reload --port 8000")
    print("  2. Open the UI:       frontend/index.html")
    print("  3. API Explorer:      http://localhost:8000/docs")
    print()


if __name__ == "__main__":
    main()
