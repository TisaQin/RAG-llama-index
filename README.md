# Project Overview — Retrieval-Augmented Generation (RAG) Mini-Pipeline

This repo implements a compact, end-to-end **RAG** pipeline on a single public text corpus (Paul Graham’s essay) using **LlamaIndex 0.9.48** and an OpenAI chat model. The notebook builds a vector index, answers natural-language questions over the corpus, and **evaluates retrieval quality** with **Hit Rate** and **MRR**. It’s designed as a clear, reproducible baseline that you can swap to your own documents and scale up with rerankers, hybrid search, and persistence.

> **Why RAG?** Retrieval grounds model answers in your source documents and reduces hallucination. This project shows the backbone you’ll need: **ingest → chunk → index → retrieve → generate → evaluate**.

---

## The Problem We’re Solving

LLMs are great pattern matchers, not databases. If you ask models about niche or private material, they can improvise. **RAG** addresses this by **retrieving** the most relevant chunks from your corpus and using them as **context** for generation—so answers are constrained by what’s actually in your files.

---

## What This Project Does

- **Loads** a small text dataset (Paul Graham essay) with `SimpleDirectoryReader`.
- **Chunks** it with `SimpleNodeParser(chunk_size=512)`.
- **Indexes** the chunks via `VectorStoreIndex`.
- **Answers** questions with a `QueryEngine` (top-k = 2 by default).
- **Evaluates** retrieval using **Hit Rate** and **MRR** via `RetrieverEvaluator`.

---

## System Diagram
<p align="center"><img width="2085" height="800" alt="RAG Pipeline Flow (Top-Down)" src="https://github.com/user-attachments/assets/9d4c2280-fc63-4206-b805-e8165d4f93fd" />


**Flow**: Ingestion → Chunking → Vector Index → Retriever (top-k) → LLM Answer → Metrics (MRR, Hit Rate)

---

# 1) Data, Ingestion, and Chunking

The demo corpus is downloaded at runtime into `data/paul_graham/paul_graham_essay.txt` and loaded with `SimpleDirectoryReader`. The text is **chunked to 512 tokens** using `SimpleNodeParser`, producing nodes for embedding and search.

```python
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser

# 1) Ingest local files (replace with your own directory path)
documents = SimpleDirectoryReader("data/paul_graham").load_data()

# 2) Chunk into nodes (tune chunk_size for your corpus)
node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)
```

**Notes**
- Start with 400–800 token chunks; adjust for your domain.
- Use **overlap** in later iterations if your queries depend on cross-paragraph context.

---

# 2) Indexing & Querying

We build an in-memory **`VectorStoreIndex`** and expose a **`QueryEngine`**. By default the engine returns top-2 chunks (you can set `similarity_top_k=k`). The notebook shows a simple example QA like _“What did the author do growing up?”_ answered from the corpus.

```python
from llama_index import VectorStoreIndex

# 3) Build the vector index
vector_index = VectorStoreIndex(nodes)

# 4) Create a query engine
query_engine = vector_index.as_query_engine(similarity_top_k=2)

# 5) Ask a question grounded in the corpus
response = query_engine.query("What did the author do growing up?")
print(response.response)
```

> The notebook demonstrates top-2 retrieval behavior and shows the matched source nodes for transparency.

---

# 3) Models & Configuration

The notebook uses an **OpenAI chat model** with `temperature=0` for deterministic answers. If you change models, keep the evaluation section so you can **measure** impact.

```python
from llama_index.llms import OpenAI

# LLM configuration (swap to your preferred model)
llm = OpenAI(model="gpt-3.5-turbo", temperature=0)
```

**Security: DO NOT hardcode API keys. Use environment variables:**

```bash
# macOS/Linux
export OPENAI_API_KEY="sk-..."
# Windows (Powershell)
setx OPENAI_API_KEY "sk-..."
```

The notebook pulls credentials from the environment—keep secrets out of version control.

---

# 4) Retrieval Evaluation (Hit Rate & MRR)

We programmatically **generate QA pairs** from the corpus with `generate_question_context_pairs`, then evaluate a **Retriever** at `top_k=2` using **Mean Reciprocal Rank (MRR)** and **Hit Rate**. The async evaluator is made notebook-friendly via `nest_asyncio.apply()`.

### Metrics (intuitions)

- **Hit Rate@k**: how often at least one relevant chunk is returned in the top-k.
- **MRR@k**: average of `1/rank` of the first relevant chunk (higher = better).

```python
import nest_asyncio; nest_asyncio.apply()

from llama_index.evaluation import (
    generate_question_context_pairs,
    RetrieverEvaluator
)

# Build evaluation dataset from nodes
qa_dataset = generate_question_context_pairs(
    nodes,
    llm=llm,
    num_questions_per_chunk=2
)

# Evaluate the retriever at top_k=2
retriever = vector_index.as_retriever(similarity_top_k=2)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["mrr", "hit_rate"], retriever=retriever
)

eval_results = await retriever_evaluator.aevaluate_dataset(qa_dataset)
```

Aggregate to a summary table:

```python
import pandas as pd

def display_results(name, eval_results):
    metric_dicts = [r.metric_vals_dict for r in eval_results]
    full_df = pd.DataFrame(metric_dicts)
    metric_df = pd.DataFrame({
        "Retriever Name": [name],
        "Hit Rate": [full_df["hit_rate"].mean()],
        "MRR": [full_df["mrr"].mean()],
    })
    return metric_df

display_results("OpenAI Embedding Retriever", eval_results)
```

| Retriever Name               | Hit Rate |   MRR |
|-----------------------------|---------:|------:|
| OpenAI Embedding Retriever  | **0.7966** | **0.6737** |

> Numbers vary slightly by run/model; treat these as a baseline for testing upgrades (rerankers, hybrid search, better embeddings).

---

# 5) Results & Example Behavior

- **Grounded answers**: responses cite the most similar text chunks used as evidence.
- **Adjustable recall**: tweak `similarity_top_k` to balance precision and recall.
- **Transparent eval**: MRR and Hit Rate quantify if the right evidence is being surfaced.

---

# 6) How to Run

### 6.1. Environment

```bash
# Python 3.10+ recommended
pip install "llama-index==0.9.48" pandas
# plus your chosen OpenAI / embedding providers as needed
```

### 6.2. Credentials

Export `OPENAI_API_KEY` in your shell (see Security note above).

### 6.3. Execute

Run the notebook `RAG.ipynb` (or port the cells into a Python script). The included cells:
1) **Download** demo text into `data/paul_graham/`
2) **Parse & chunk** with `SimpleNodeParser`
3) **Index & query** with `VectorStoreIndex`
4) **Evaluate** with `RetrieverEvaluator` (MRR, Hit Rate)

---

# 7) Design Choices & Rationale

- **Chunk size = 512**: balances semantic cohesion and index granularity for the essay-length corpus.
- **Top-k = 2** retrieval: encourages concise grounding for a short corpus; raise when moving to multi-file datasets.
- **Async evaluation** in notebooks via `nest_asyncio`: simplifies running `aevaluate_dataset` without event-loop conflicts.

---

# 8) Extending the Baseline

1. **Persistent vector store** (Chroma/FAISS/Weaviate) so you don’t rebuild on every run.
2. **Add a reranker** (cross-encoder) for improved precision at low k.
3. **Hybrid search** (BM25 + dense) to catch lexical exact-match queries.
4. **Metadata & filters**: tag nodes with source/file/section and allow structured filtering.
5. **Prompt-aware chunking** and **overlap** for QA spanning paragraphs.
6. **Response citation rendering**: show the exact passages and source IDs in the answer UI.
7. **Evaluation at scale**: create a fixed QA set and regression-test MRR/Hit Rate on changes.

---

# 9) Limitations & Future Work

- **Single-document demo**: Great for pedagogy, not a stress test. Expand to multi-file, multi-domain sources.
- **No reranking**: Base vector similarity can surface near-misses; add rerankers.
- **No latency budgeting**: For apps, measure retrieval time, token usage, and end-to-end P95.
- **Model variability**: Results change with different LLMs/embeddings—keep the evaluation harness and track metrics over time.

---


# 10) Acknowledgments

- Built with **LlamaIndex 0.9.48**.
- Demo corpus from the LlamaIndex examples (Paul Graham essay).
- Notebook cells mirrored and adapted from the included `RAG.ipynb` for this repo.

---
