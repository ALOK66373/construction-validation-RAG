"""
============================================================
  RAG ENGINE  (app/rag_engine.py)
============================================================

THIS IS THE HEART OF THE APPLICATION.

It orchestrates the complete RAG (Retrieval-Augmented Generation) pipeline:

    User Question
         │
         ▼
    [RETRIEVE]  Embed query → search ChromaDB → top-5 chunks
         │
         ▼
    [AUGMENT]   Inject those chunks into a prompt as CONTEXT
         │
         ▼
    [GENERATE]  Claude reads context + question → writes answer
         │
         ▼
    Grounded, citable answer

WHY RAG BEATS BOTH APPROACHES:

  ┌────────────────────────────────────────────────────────────┐
  │  Pure LLM (no RAG):                                        │
  │    Claude uses only training data → might "hallucinate"    │
  │    local rules it never actually saw.                      │
  ├────────────────────────────────────────────────────────────┤
  │  Fine-tuning:                                              │
  │    Re-train the model on your data → $$$, slow, needs ML  │
  │    expertise, outdated the moment rules change.            │
  ├────────────────────────────────────────────────────────────┤
  │  RAG (what we use):                                        │
  │    Fetch real text, prepend to prompt → cheap, fast,       │
  │    always up-to-date (just edit .txt files and re-ingest). │
  │    Answers are CITABLE — users see exactly which document. │
  └────────────────────────────────────────────────────────────┘
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

from document_processor import load_documents
from embedder import TextEmbedder
from vector_store import VectorStore


class RAGEngine:
    """
    Orchestrates the full Retrieve → Augment → Generate pipeline.

    Usage:
        engine = RAGEngine()
        result = engine.answer("What approvals do I need for a second floor?")
        print(result["answer"])    # Groq answer
        print(result["sources"])   # List of source chunks used
    """

    def __init__(self, data_dir: str = "./data", chroma_dir: str = "./chroma_db"):
        """
        Initialize all three components. Auto-ingests if vector store is empty.

        Args:
            data_dir   : Folder containing .txt knowledge files.
            chroma_dir : Where ChromaDB persists its data to disk.

        STARTUP SEQUENCE:
            1. Load embedding model (all-MiniLM-L6-v2 from HuggingFace)
            2. Connect to ChromaDB (or create it if first run)
            3. Create Groq client (reads GROQ_API_KEY from env)
            4. If vector store is empty → automatically ingest data/
        """
        print("[RAGEngine] Initializing components...")

        # Component 1: Embedding model
        # CRITICAL: Same instance used for BOTH ingestion and retrieval.
        # Never create two different TextEmbedder instances — the model must be
        # identical for query vectors to be comparable to document vectors.
        self.embedder = TextEmbedder()

        # Component 2: Vector store (ChromaDB on disk)
        self.vector_store = VectorStore(persist_dir=chroma_dir)

        # Component 3: Groq chat client
        # Reads GROQ_API_KEY from environment automatically.
        # NEVER hardcode an API key in source code — use .env!
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            self.groq = Groq(api_key=api_key)
        else:
            self.groq = None
            print("[RAGEngine] GROQ_API_KEY is not set. The app will start, but generation requests will be disabled until you add the key.")

        # Auto-ingest if first run (no data in vector store yet)
        if self.vector_store.is_empty():
            print("[RAGEngine] Vector store is empty. Running initial ingestion...")
            self.ingest(data_dir=data_dir, force=False)
        else:
            print(f"[RAGEngine] Vector store ready — {self.vector_store.count()} chunks loaded.")

        print("[RAGEngine] ✅ Ready to answer questions!\n")

    # ── INGESTION ──────────────────────────────────────────────────────────

    def ingest(self, data_dir: str = "./data", force: bool = False) -> dict:
        """
        Load .txt files → chunk → embed → store in ChromaDB.

        This is the "offline" phase. Run once before serving, or again
        whenever you update the .txt files.

        Args:
            data_dir : Path to folder with .txt knowledge files.
            force    : True  → clear existing data and re-ingest everything.
                       False → skip if vector store already has data.

        Returns:
            dict with status and count of ingested chunks.
        """
        if not self.vector_store.is_empty() and not force:
            count = self.vector_store.count()
            print(f"[RAGEngine] Skipping ingest — {count} chunks already stored. Use force=True to re-ingest.")
            return {"status": "skipped", "chunks": count}

        if force:
            self.vector_store.clear()

        print(f"[RAGEngine] Loading documents from: {data_dir}")
        documents = load_documents(data_dir)
        self.vector_store.add_documents(documents, self.embedder)

        return {"status": "success", "chunks_ingested": len(documents)}

    # ── RETRIEVAL ──────────────────────────────────────────────────────────

    def retrieve(self, query: str, n_results: int = 5) -> List[Tuple[str, dict, float]]:
        """
        Step 1 of RAG: find the most relevant chunks for this query.

        Process:
          1. embed(query)          → 384-dim vector
          2. ChromaDB HNSW search  → top-k nearest chunks
          3. Return chunks with similarity scores

        Args:
            query     : The user's raw question string.
            n_results : How many chunks to fetch. More = richer context
                        but longer prompt (more tokens = higher API cost).
                        5 is a good default for building-code questions.

        Returns:
            [(chunk_text, metadata_dict, similarity_0_to_1), ...]
            Sorted by similarity, highest first.
        """
        print(f"[RAGEngine] Retrieving top-{n_results} chunks for: '{query[:60]}...'")
        query_vector = self.embedder.embed_query(query)
        results      = self.vector_store.search(query_vector, n_results=n_results)

        if results:
            print(f"[RAGEngine] Top chunk score: {results[0][2]:.3f} (from {results[0][1]['source']})")
        return results

    # ── AUGMENTATION ───────────────────────────────────────────────────────

    def build_prompt(self, query: str, context_chunks: List[Tuple[str, dict, float]]) -> str:
        """
        Step 2 of RAG: inject retrieved chunks into the model prompt.

        This is the "augmentation" step — we give the model real document text
        as CONTEXT so it answers from facts, not from training guesses.

        PROMPT ENGINEERING DECISIONS:
          • Role statement   → tells the model it's a building-code expert
          • CRITICAL INSTRUCTION → instructs it to use ONLY the context
          • Source labels    → [filename, chunk #X] so answers are attributable
          • Structure ask    → approvals / rules / documents / warnings
          • Honesty fallback → say "not in knowledge base" rather than guess

        Args:
            query          : The user's question.
            context_chunks : Output of retrieve() — (text, meta, score) tuples.

        Returns:
            The full prompt string to send to Claude.
        """
        context_parts = []
        for text, meta, score in context_chunks:
            label = f"[{meta['source']}  chunk #{meta['chunk_index']}  relevance={score:.2f}]"
            context_parts.append(f"{label}\n{text}")

        context_block = "\n\n---\n\n".join(context_parts)

        return f"""You are a knowledgeable Construction and Building Code Advisor specializing in Indian building regulations, especially Bangalore (BBMP) and the National Building Code (NBC 2016).

CRITICAL INSTRUCTIONS:
- Answer ONLY using the information in the CONTEXT below.
- If the context is incomplete, say: "This information is not fully covered in my current knowledge base" and answer only what is supported by the context.
- Do NOT copy large chunks of text from the retrieved documents or repeat raw excerpts.
- Do NOT invent rules, approvals, measurements, or procedures that are not present in the context.
- Synthesize a detailed, logically organized answer that reads like a professional advisor note.

WRITE THE RESPONSE IN THIS FORMAT:
1. Short conclusion or direct answer
2. Required approvals / permissions
3. Applicable rules, limits, or calculations
4. Documents or checks usually required
5. Risks, cautions, or local considerations

Use clear paragraphs and short bullet points where helpful. Explain the logic in plain English, but keep the answer grounded in the provided code text.

═══════════════════════════════════════════════════
CONTEXT FROM BUILDING CODE DOCUMENTS:
═══════════════════════════════════════════════════
{context_block}
═══════════════════════════════════════════════════

USER QUESTION: {query}

Provide a detailed, coherent, professional answer based on the context above."""

    # ── GENERATION ─────────────────────────────────────────────────────────

    def answer(self, query: str, n_results: int = 5) -> dict:
        """
        The complete RAG pipeline: Retrieve → Augment → Generate.

        This is called by the FastAPI /ask endpoint on every user question.

        Args:
            query     : The user's question.
            n_results : Number of chunks to retrieve as context.

        Returns:
            {
              "answer"      : Groq text response (string)
              "sources"     : List of source-card dicts for the UI
              "query"       : The original question (echoed back)
              "chunks_used" : How many chunks were given to the model
            }

        FULL TRACE for "What setbacks for G+2 in Bangalore?":
          ① embed("What setbacks for G+2 in Bangalore?")
              → [0.12, -0.45, ..., 0.32]   (384 numbers)

          ② ChromaDB.search(that vector, k=5)
              → [("For G+2 to G+3 buildings, setback = 4m...", score=0.89),
                 ("Setbacks depend on road width...",           score=0.82),
                 ... 3 more ...]

          ③ build_prompt(query, those 5 chunks)
              → "You are a building code advisor... CONTEXT: [...] QUESTION: ..."

          ④ claude.messages.create(prompt)
              → "For a G+2 residential building in Bangalore, the required setback..."

          ⑤ Return {answer, sources, query, chunks_used}
        """
        # Guard: empty vector store
        if self.vector_store.is_empty():
            return {
                "answer": (
                    "The knowledge base is empty. "
                    "Please run: `python ingest.py` to load your documents first."
                ),
                "sources":      [],
                "query":        query,
                "chunks_used":  0,
            }

        # ── Step 1: Retrieve ────────────────────────────────────────────────
        context_chunks = self.retrieve(query, n_results=n_results)

        if not context_chunks:
            return {
                "answer":       "No relevant information found. Try rephrasing your question.",
                "sources":      [],
                "query":        query,
                "chunks_used":  0,
            }

        # ── Step 2: Augment ─────────────────────────────────────────────────
        prompt = self.build_prompt(query, context_chunks)

        # ── Step 3: Generate ────────────────────────────────────────────────
        if self.groq is None:
            return {
                "answer": (
                    "Groq is not configured yet. Please add GROQ_API_KEY to your .env file "
                    "and restart the app before asking questions."
                ),
                "sources": [],
                "query": query,
                "chunks_used": 0,
            }

        # Prefer the user's configured model, then try models known to be available on Groq.
        model_candidates = []
        env_model = os.environ.get("GROQ_MODEL")
        if env_model:
            model_candidates.append(env_model)

        default_candidates = [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",
            "groq/compound",
            "allam-2-7b",
        ]

        try:
            account_models = self.groq.models.list()
            for model in account_models.data:
                model_id = getattr(model, "id", None)
                if not model_id or "prompt-guard" in model_id or "safeguard" in model_id:
                    continue
                if model_id not in model_candidates:
                    model_candidates.append(model_id)
        except Exception:
            pass

        for model_name in default_candidates:
            if model_name not in model_candidates:
                model_candidates.append(model_name)

        last_error = None
        response = None
        used_model = None
        for model_name in model_candidates:
            print(f"[RAGEngine] Calling Groq API with model: {model_name}")
            try:
                response = self.groq.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.2,
                )
                used_model = model_name
                break
            except Exception as exc:
                last_error = exc
                print(f"[RAGEngine] Model '{model_name}' failed: {exc}")

        if response is None:
            raise RuntimeError(
                "Groq model request failed for all configured candidates. "
                "Set GROQ_MODEL to a model that is available in your Groq account, "
                "for example: llama-3.1-8b-instant or llama-3.3-70b-specdec. "
                f"Last error: {last_error}"
            ) from last_error

        answer_text = response.choices[0].message.content or ""
        print(f"[RAGEngine] Answer received ({len(answer_text)} chars) via {used_model}")

        # ── Format source attribution for the UI ────────────────────────────
        # The frontend renders these as "source cards" with relevance bars.
        sources = [
            {
                "source":            meta["source"],
                "chunk_index":       meta["chunk_index"],
                "relevance_percent": round(score * 100, 1),
                "preview":           text[:200] + ("..." if len(text) > 200 else ""),
            }
            for text, meta, score in context_chunks
        ]

        return {
            "answer":      answer_text,
            "sources":     sources,
            "query":       query,
            "chunks_used": len(context_chunks),
        }
