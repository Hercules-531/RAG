# RAG

*A build-through of the techniques that separate a working RAG demo from one that holds up past a toy corpus — hybrid retrieval, parent-child chunking, cost-aware model routing, and LangSmith observability, each implemented as a standalone, runnable module.*

Most RAG walkthroughs stop at `vector_store.similarity_search(query, k=3)` and call it done. This repo is where I worked past that baseline, one file at a time: hybrid search for when pure vector similarity misses exact terms, parent-child chunking for when small chunks retrieve well but don't give the LLM enough to work with, contextual compression for when a "relevant" chunk is still mostly noise, multi-query expansion for when a single phrasing of the question misses half the relevant docs, and a cost layer so every query doesn't default to hitting the most expensive model available.

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
│   └── cost_budgeting_and_model_routing.py  complexity-based model routing + token budgets
└── main.py                                  environment / provider sanity check
```

Each file has its own `if __name__ == "__main__"` demo and can be run on its own — there's no shared pipeline gluing them together yet (more on that in [Notes](#notes)).

## Retrieval

**Baseline — semantic chunking.** `RAG_pipeline.py` is the simplest thing in the repo: instead of splitting text every N characters, `SemanticChunker` breaks at points where consecutive sentences are least similar in embedding space:
```python
splitter = SemanticChunker(embeddings=embeddings_model, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95)
```
That avoids the classic fixed-size problem of cutting a chunk off mid-thought. Everything below addresses a limitation this baseline still has.

**Hybrid search.** Pure vector similarity is bad at exact terms — acronyms, IDs, rare vocabulary that doesn't cluster well in embedding space. `Hybrid_Search/hybrid_search_prod.py` combines a Chroma dense retriever with a BM25 keyword retriever through LangChain's `EnsembleRetriever`:
```python
semantic_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
bm25_retriever = BM25Retriever.from_documents(chunks, k=3)
hybrid_retriever = EnsembleRetriever(retrievers=[semantic_retriever, bm25_retriever], weights=[0.7, 0.3])
```
Weighted 70/30 toward the dense side, so semantic similarity still leads the ranking, but BM25 pulls in exact-match candidates the embedding model would otherwise rank too low to surface.

**Parent-child chunking.** The core tension in chunking: small chunks embed and match precisely, but they don't hand the LLM enough surrounding context to actually answer from. `ParentChild_document_retrival.py` splits every document twice — once at 200 characters for indexing, once at 800 for what actually gets returned:
```python
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore, docstore=store,
    child_splitter=child_splitter, parent_splitter=parent_splitter,
)
```
Similarity search runs against the 200-char children in Chroma; `InMemoryStore` maps each hit back to its 800-char parent, so generation gets the full surrounding paragraph instead of an isolated fragment.

**Contextual compression.** Even a well-matched chunk is often mostly irrelevant to the actual question — one sentence that answers it, buried in a paragraph that doesn't. `Contextual_compression.py` wraps the base retriever in an `LLMChainExtractor` that strips each retrieved document down to just the spans relevant to the query before it reaches the generation prompt, which keeps prompt size (and cost) down and cuts the noise the final LLM call has to reason around.

**Multi-query expansion.** A single embedding of the user's exact wording can miss documents that answer the same question phrased differently. `Multi_query.py` uses `MultiQueryRetriever` to have an LLM generate several reworded variants of the incoming query, retrieves for each, and unions the deduplicated results — a few extra LLM calls traded for retrieval recall.

## Cost optimization

**Complexity-based model routing.** Sending every query through your best model is the fastest way to burn a RAG budget. `cost_budgeting_and_model_routing.py`'s `ModelRouter` classifies a query as `simple` or `complex` with one cheap classifier call, then routes accordingly:
```python
if complexity == "simple":
    model, model_name, cost_per_1k = self.cheap_model, "gemini-2.5-flash", 0.00015
else:
    model, model_name, cost_per_1k = self.expensive_model, "gemini-3.5-flash", 0.0025
```
Every call returns the response alongside which model handled it and an estimated cost, so routing decisions are auditable instead of silent.

**Token budgeting.** `TokenBudget` / `BudgetedLLM` check a request's estimated size against a configured ceiling (4,000 tokens by default) *before* sending it, rejecting oversized requests instead of paying for them first and finding out after:
```python
within_budget, tokens = self.budget.check_budget(query)
if not within_budget:
    raise ValueError(f"Query exceeds token budget: {tokens} > {self.budget.max_per_request}")
```
It also keeps a running tally of input/output tokens and request count — a minimal but real version of the usage accounting any team running LLM calls at volume ends up needing.

## Observability

`Langsmith_setup.py` wires LangSmith tracing through the pipeline via the `@traceable` decorator, so every chain invocation shows up as a trace with inputs, outputs, and latency instead of a `print` statement. It also covers named runs with tags —
```python
@traceable(name="named_runs_demo", tags=["production", "summarization"])
```
— and attaching request-level metadata (user ID, request type) so runs can be filtered in the dashboard. The difference this makes in practice: "something broke somewhere" versus being able to pull up the exact failing run.

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain (`core`, `classic`, `community`, `experimental`) |
| Vector store | Chroma |
| Sparse retrieval | BM25 (`rank-bm25`) |
| LLM / embeddings | Google Gemini — `gemini-2.5-flash`, `gemini-3.5-flash`, `gemini-embedding-001` |
| Observability | LangSmith |
| Document loading | `pypdf`, `beautifulsoup4` |
| Environment / deps | Python 3.14, managed with `uv` |

`langchain-openai` and `langchain-anthropic` are already imported in `main.py` — every retriever and chain here is built on LangChain's model-agnostic interfaces, so swapping the active provider is a one-line change. The demos themselves currently run on Gemini.

## Running it

```bash
# install dependencies (uv reads pyproject.toml / uv.lock)
uv sync
```

Set up `.env`:
```
GOOGLE_API_KEY=...
LANGSMITH_API_KEY=...      # only needed for Langsmith_setup.py
LANGSMITH_TRACING=true
```

Then run any module directly:
```bash
uv run RAG_Optimizations/cost_budgeting_and_model_routing.py
uv run RAG_Optimizations/ParentChild_document_retrival.py
uv run Hybrid_Search/hybrid_search_prod.py
```

`Document_loaders.py` and `hybrid_search_prod.py` point at a local PDF path for testing — swap in a path to your own PDF before running those two.

## Notes

- Every technique above is demonstrated in isolation, on purpose. There's no single `app.py` wiring hybrid search → parent-child retrieval → compression → routing into one end-to-end pipeline yet — that's the natural next step.
- The cost estimates in the routing module use a word-count-based token approximation, not `tiktoken` — accurate enough to demonstrate the routing logic, not a production billing meter.
- `langgraph` is in the dependency list for agent-based orchestration on top of this retrieval layer, which isn't built out in this repo yet.
