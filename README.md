# 🏗️ Construction Building Code Advisor

A Retrieval-Augmented Generation (RAG) app for answering building-code and permit questions using local knowledge files, semantic search, and Groq-hosted LLMs.

The project currently uses:
- Streamlit for the web UI
- ChromaDB for local vector search
- Sentence Transformers for embeddings
- Groq for chat generation
- Plain text building-code documents in the `Data/` folder as the official knowledge base

---

## What this app does

The app takes a user question, searches the most relevant code chunks from the knowledge base, and sends them to a Groq model with a grounded prompt. The model then answers in a structured, professional way using only the retrieved context.

This helps reduce hallucination and keeps the answer connected to the actual Bangalore/BBMP and national-building-code material in the project.

---

## Project structure

```text
Construction-Building-Code-Advisor-main/
├── main.py                  # Streamlit app entry point
├── rag_engine.py            # Retrieval + augmentation + Groq generation
├── document_processor.py    # Reads .txt docs and splits them into chunks
├── embedder.py              # Embedding model wrapper
├── vector_store.py          # ChromaDB wrapper for search and storage
├── ingest.py                # Manual ingestion helper
├── requirements.txt         # Python dependencies
├── .env                     # Local runtime secrets (not committed)
├── .env.example             # Environment template
├── Data/
│   ├── bangalore_bylaws.txt
│   ├── national_building_code.txt
│   └── permit_procedures.txt
├── chroma_db/               # Auto-created vector store
├── tests/
│   └── test_streamlit_app.py
├── README.md
└── .venv/                   # Local virtual environment
```

---

## Tech stack

- Python 3.10+
- Streamlit
- sentence-transformers
- chromadb
- groq
- python-dotenv

---

## Quick start

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure your Groq API key

Create a `.env` file in the project root using the template below:

```env
GROQ_API_KEY="your_groq_api_key_here"
GROQ_MODEL="openai/gpt-oss-20b"
```

You can copy from `.env.example` if present:

```bash
copy .env.example .env
```

> Keep your `.env` file local and do not commit it to source control.

### 4) Load the knowledge base

```bash
python ingest.py
```

This reads the `.txt` files from `Data/`, chunks them, embeds them, and stores them in `chroma_db/`.

### 5) Start the app

```bash
streamlit run main.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

---

## How the RAG flow works

```text
User question
   ↓
Embed query
   ↓
Search ChromaDB for similar chunks
   ↓
Build context prompt with retrieved chunks
   ↓
Call Groq model
   ↓
Return structured answer + source excerpts
```

The app retrieves the most relevant chunks and asks the LLM to answer using only that evidence, reducing unsupported answers.

---

## Model configuration

The application reads the model name from the environment variable `GROQ_MODEL`.

Example:

```env
GROQ_MODEL="openai/gpt-oss-20b"
```

If a model is unavailable on your Groq account, the app will try other known candidates automatically, but the recommended setup is to keep your active model in `.env`.

---

## Troubleshooting

### Groq model not found

If you see a Groq model error like `The model does not exist or you do not have access to it`, check:

- your key is valid and active
- your Groq account has access to that model
- `GROQ_MODEL` matches an available model in your account

A working example is currently:

```env
GROQ_MODEL="openai/gpt-oss-20b"
```

### App starts but answers feel weak

Possible reasons:
- the retrieved chunks are too generic
- the prompt is asking for unsupported facts
- the documents in `Data/` do not contain enough detail on the question

Try rephrasing the question and increasing the number of retrieved chunks using the sidebar slider.

### Knowledge base is empty

Run:

```bash
python ingest.py
```

If needed, delete the `chroma_db/` folder and re-run ingestion.

---

## Example questions

- What approvals are required for a second floor in Bangalore?
- What are the setback requirements for G+2 construction?
- What documents are normally required for a building permit?
- Are there airport-related restrictions for building height?

---

## Notes

- This app is designed for grounded legal/building-code guidance, not legal advice.
- Always verify exact local requirements with BBMP, BDA, or the relevant authority before construction decisions.
- The content is based on the local `.txt` knowledge files and should be kept updated as regulations change.

---
