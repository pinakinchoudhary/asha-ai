# Use Case: Protocol-Based Q&A (RAG)

## Problem Statement

ASHA workers are expected to follow complex clinical protocols from the National Health Mission (NHM) and SUMAN guidelines, but these protocols are spread across lengthy PDF documents that are difficult to search and reference in the field. ASHAs frequently encounter situations where they need to recall specific procedures — danger signs during pregnancy, vaccine schedules, postnatal care steps — but have no quick way to look them up.

---

## Solution

Asha AI implements a **Retrieval-Augmented Generation (RAG) pipeline** that ingests NHM/SUMAN clinical protocol PDFs, builds a FAISS vector index, and enables ASHAs to ask questions in Hindi or English. The system retrieves the most relevant protocol passages and uses an LLM to generate a grounded, fact-checked answer — eliminating hallucination by anchoring responses to official source documents.

---

## Data Flow

```
ASHA asks: "Pregnancy mein kya danger signs hain?"
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Translation (if Hindi)                           │
│  Sarvam Mayura: Hindi → English                   │
│  "What are the danger signs during pregnancy?"    │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Query Embedding                                  │
│  all-MiniLM-L6-v2 → 384-dim vector              │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  FAISS Vector Search (CPU)                        │
│  Top-k nearest neighbors (k=5)                   │
│  From pre-built index of NHM protocol chunks      │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Context Assembly                                 │
│  Retrieved chunks concatenated as context         │
│  System prompt enforces grounding:                │
│  "Answer ONLY from the provided context.          │
│   If the answer is not in the context, say so."   │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  LLM Grounded Generation                          │
│  Sarvam-m / Groq Llama 3.3 70B                   │
│  Input: question + retrieved context              │
│  Output: factual answer with source attribution   │
└──────────┬───────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────┐
│  Translation + TTS                                │
│  English answer → Hindi (Mayura)                  │
│  Hindi text → Hindi audio (Bulbul TTS)            │
└──────────────────────────────────────────────────┘
```

---

## Architecture

| Layer | Component | Technology |
|-------|-----------|-----------|
| **PDF Ingestion** | Extract text from NHM protocol PDFs | PyPDF2 3.0.1 |
| **Chunking** | Split into retrieval units | 500-token chunks, 50-token overlap |
| **Embedding** | Convert chunks to vectors | `all-MiniLM-L6-v2` (384-dim, sentence-transformers) |
| **Vector Index** | Store and search embeddings | FAISS CPU (in-memory, persisted to DBFS) |
| **Retrieval** | Find relevant chunks for a query | FAISS nearest-neighbor search (top-5) |
| **Generation** | Answer grounded in retrieved context | Sarvam-m / Groq Llama 3.3 70B |
| **Anti-Hallucination** | Enforce factual grounding | System prompt constraint + source attribution |

---

## Index Details

| Metric | Value |
|--------|-------|
| Embedding dimensions | 384 |
| Chunk size | 500 tokens |
| Chunk overlap | 50 tokens |
| Index type | FAISS Flat L2 (CPU) |
| Index size on disk | ~5 MB |
| RAM footprint | ~50 MB |
| Source documents | NHM HBNC/HBYC Handbook, ASHA Module 6, Routine Immunization Handbook |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/rag_pipeline.py` | PDF ingestion, chunking, FAISS index build, retrieval, LLM-grounded generation |
| `data/nhm_protocols/` | Directory for uploading NHM/SUMAN protocol PDFs |
| `data/faiss_index/` | Persisted FAISS index and chunk metadata |
| `notebooks/03_build_rag_index.py` | Build/rebuild the FAISS index from uploaded PDFs |

---

## Demo

```
Input:  "What are the danger signs during pregnancy?"
Output: "According to the NHM HBNC/HBYC Handbook, the following are danger signs during pregnancy:
         1. Severe vaginal bleeding
         2. Convulsions / fits
         3. Severe headache with blurred vision
         4. Fever and too weak to get out of bed
         5. Severe abdominal pain
         6. Fast or difficult breathing
         If any of these signs are present, refer immediately to the nearest health facility."
         
         Source: NHM Handbook for ASHA Facilitators, Chapter 3, Pages 42-45
```
