# RAG

*Hybrid retrieval, parent-child chunking, cost-aware model routing, and LangSmith observability — the fixes a naive RAG pipeline needs before it holds up past a toy corpus. Each one is implemented as its own standalone, runnable module.*

## Layout

```
RAG/
├── RAG_pipeline.py                          baseline: semantic chunking + similarity retrieval
├── Document_loaders.py                      loader patterns (text / web / directory / PDF)
├── Langsmith_setup.py                       tracing, named runs, run metadata
├── Hybrid_Search/
│   └── hybrid_search_prod.py                dense (Chroma) + sparse (BM25) ensemble retrieval
├── RAG_Optimizations/
│   ├── ParentChild_document_retrival.py     parent-child chunking
│   ├── Contextual_compression.py            LLM-based filtering of retrieved context
│   ├── Multi_query.py                       query expansion / rewriting
│   └── cost_budgeting_and_model_routing.py  model routing + token budgets
└── main.py                                  environment sanity check
```
No shared entry point — each file runs on its own.

## Retrieval

- **Baseline chunking** (`RAG_pipeline.py`) — splits at embedding-similarity dips (95th percentile) instead of a fixed character count, so chunks don't cut off mid-thought.

- **Hybrid search** (`hybrid_search_prod.py`) — pure vector similarity misses exact terms and rare vocabulary. Combines Chroma (dense) with BM25 (sparse) via `EnsembleRetriever`, weighted 0.7/0.3:
  ```python
  EnsembleRetriever(retrievers=[semantic_retriever, bm25_retriever], weights=[0.7, 0.3])
  ```

- **Parent-child chunking** (`ParentChild_document_retrival.py`) — small chunks retrieve precisely but don't give the LLM enough context. Indexes 200-char children, returns their 800-char parents:
  ```python
  parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
  child_splitter  = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
  ```

- **Contextual compression** (`Contextual_compression.py`) — an `LLMChainExtractor` strips each retrieved chunk down to the spans actually relevant to the query before it hits the generation prompt.

- **Multi-query expansion** (`Multi_query.py`) — `MultiQueryRetriever` has an LLM rewrite the question into several variants, retrieves for each, and unions the results.

## Cost optimization

- **Model routing** (`cost_budgeting_and_model_routing.py`) — a classifier call tags each query `simple`/`complex` and routes accordingly, returning the model used and an estimated cost with every response:
  ```python
  # simple  -> gemini-2.5-flash, $0.00015 / 1k tokens
  # complex -> gemini-3.5-flash, $0.0025 / 1k tokens
  ```

- **Token budgeting** — `TokenBudget` estimates request size and rejects anything over a configured ceiling (4,000 tokens by default) before it's sent, and tracks running input/output usage.

## Observability

`Langsmith_setup.py` wires LangSmith tracing through `@traceable`, so every chain call becomes a searchable trace instead of a print statement — including named runs with tags and per-request metadata (user ID, request type) for filtering in the dashboard.

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain (`core`, `classic`, `community`, `experimental`) |
| Vector store | Chroma |
| Sparse retrieval | BM25 (`rank-bm25`) |
| LLM / embeddings | Gemini — `gemini-2.5-flash`, `gemini-3.5-flash`, `gemini-embedding-001` |
| Observability | LangSmith |
| Deps | Python 3.14, `uv` |

`langchain-openai` and `langchain-anthropic` are already imported in `main.py` — swapping providers is a one-line change since everything here is built on LangChain's model-agnostic interfaces.

## Running it

```bash
uv sync
```

`.env`:
```
GOOGLE_API_KEY=...
LANGSMITH_API_KEY=...      # only needed for Langsmith_setup.py
LANGSMITH_TRACING=true
```

Run any module directly, e.g.:
```bash
uv run RAG_Optimizations/cost_budgeting_and_model_routing.py
```
`Document_loaders.py` and `hybrid_search_prod.py` point at a local PDF path — swap in your own before running those two.

## Notes

- No single pipeline wires hybrid search → parent-child → compression → routing together yet — each is standalone by design, for now.
- Cost estimates use word-count-based token approximation, not `tiktoken` — good enough to show the routing logic, not a billing meter.
- `langgraph` is a listed dependency for future agent orchestration, not used yet.
