# IP AgentLab

IP AgentLab is a portfolio project for intellectual-property AI agent engineering.
It focuses on patent novelty search, technical FTO workflows, and design FTO
visual search demos.

The project demonstrates three separated IP workflows.

Novelty search:

1. Parse a technical disclosure.
2. Extract invention structure and innovation points.
3. Expand search keywords and classification hints.
4. Retrieve chunk-level prior-art evidence with BM25 + vector hybrid retrieval.
5. Judge evidence coverage for each innovation point.
6. Generate grounded RAG analysis with chunk citations.
7. Generate a Markdown novelty report.

Technical FTO:

1. Parse implementation or product technical features.
2. Expand claim-focused search queries.
3. Retrieve claim-level patent evidence.
4. Build an initial FTO claim chart from retrieved claim chunks.
5. Generate grounded RAG analysis with claim citations.
6. Generate a Markdown FTO report.

Design FTO:

1. Upload a product image or use the included demo image.
2. Retrieve visually similar sample design-patent images.
3. Score visual similarity and assign risk.
4. Generate a Markdown design FTO report.

The current version is intentionally small and local-first: it runs with a sample
patent corpus and deterministic scoring so the complete demo works without cloud
services. The architecture is designed to be upgraded into a stronger interview
demo with LlamaIndex-based document indexing, production embeddings, vector
storage, reranking, LangGraph-based agent orchestration, and stronger LLM-based
evidence judging.

## Product Positioning

IP AgentLab is positioned as an AI copilot for early IP review:

- For inventors and R&D teams: convert a raw technical disclosure into a reviewable
  invention summary and search strategy.
- For patent analysts: retrieve prior-art evidence at chunk level and understand
  why each result was selected.
- For IP workflow demos: show how LLM agents, RAG retrieval, evidence judging,
  and report generation can work together in a practical vertical application.

This project is not a legal-opinion system. It is an AI-assisted research tool
that prepares evidence and structured analysis for human review.

## Target Interview Capabilities

The roadmap is aligned with AI Agent engineering roles that expect:

- IP-domain workflows such as patent novelty search, FTO analysis, and evidence
  review.
- RAG chains for complex retrieval and analysis tasks.
- Multi-agent orchestration for parsing, searching, judging, comparing, and
  reporting.
- LLM prompt engineering with structured JSON outputs.
- Python engineering practices that make the demo reproducible and extensible.

## Run

Recommended conda setup:

```powershell
conda create -y -n ip-agent-lab python=3.11
conda activate ip-agent-lab
pip install -r requirements.txt
streamlit run frontend/app.py
```

If the environment already exists:

```powershell
conda activate ip-agent-lab
streamlit run frontend/app.py
```

Run commands from the `ip-agent-lab` directory.

## Current Scope

- Patent novelty search MVP
- Local sample patent corpus
- Patent document chunking
- BM25 + local vector hybrid retrieval
- Optional LlamaIndex vector retrieval backend
- RRF result fusion
- Innovation-point evidence matrix
- Initial FTO claim chart generation
- Grounded RAG generation with patent chunk citations
- Separate novelty search and technical FTO workflows
- Design FTO image upload demo
- Local sample design-patent image corpus
- Lightweight perceptual visual similarity search
- Streamlit product demo
- Structured agent workflow
- Markdown report generation

## Current Limitations

- The vector path uses a deterministic hashing embedding fallback, not a
  production semantic embedding model.
- The sample corpus is intentionally small and should be expanded or connected
  to patent APIs for realistic evaluation.
- The FTO claim chart is an MVP mapping layer based on retrieved claim chunks;
  it still needs claim-element parsing and attorney-grade review workflow.
- RAG generation has a deterministic fallback when no LLM API key is configured;
  stronger LLM judging should be used for serious review.
- Design FTO currently uses lightweight local image descriptors. It is a demo
  retrieval layer and should be upgraded to CLIP/SigLIP embeddings for broader
  product images.
- The workflow is currently orchestrated by Python classes rather than a graph
  runtime such as LangGraph.

## Planned Extensions

- LlamaIndex document ingestion, indexing, hybrid retrieval, and reranking.
- LangGraph workflow orchestration for multi-agent state management.
- Patent API integration
- Hybrid BM25 + vector search with Qdrant
- LLM structured extraction and relevance judge
- Stronger claim-element parsing and FTO claim chart review
- Design FTO image retrieval with CLIP/SigLIP and real design-patent image APIs
- Evaluation set for recall@k and precision@k

See [docs/project_positioning.md](docs/project_positioning.md) for the project
positioning and staged upgrade plan.
See [docs/canonical_schema.md](docs/canonical_schema.md) for the canonical
disclosure schema used between parsing, retrieval, comparison, and reporting.

## Retrieval Backend

The project defaults to the LlamaIndex retriever with BGE-M3 embeddings. It also
keeps the local retriever and local hashing embedding as offline fallbacks.

The project supports two retriever backends:

```text
IP_AGENT_RETRIEVER=llamaindex  # default LlamaIndex + BM25 hybrid retrieval
IP_AGENT_RETRIEVER=local       # offline local BM25 + hashing-vector fallback
```

PowerShell example:

```powershell
$env:IP_AGENT_RETRIEVER="llamaindex"
$env:IP_AGENT_EMBEDDING_PROVIDER="bge_m3"
streamlit run frontend/app.py
```

The LlamaIndex path supports pluggable embedding providers:

```text
IP_AGENT_EMBEDDING_PROVIDER=bge_m3      # default local HuggingFace BGE-M3
IP_AGENT_EMBEDDING_PROVIDER=local       # deterministic fallback
IP_AGENT_EMBEDDING_PROVIDER=openai      # OpenAI-compatible embedding API
IP_AGENT_EMBEDDING_PROVIDER=huggingface # custom HuggingFace embedding model
```

OpenAI embedding example:

```powershell
pip install llama-index-embeddings-openai
$env:IP_AGENT_RETRIEVER="llamaindex"
$env:IP_AGENT_EMBEDDING_PROVIDER="openai"
$env:IP_AGENT_EMBEDDING_MODEL="text-embedding-3-small"
$env:IP_AGENT_EMBEDDING_API_KEY="your_api_key"
streamlit run frontend/app.py
```

BGE-M3 example:

```powershell
pip install llama-index-embeddings-huggingface
$env:IP_AGENT_RETRIEVER="llamaindex"
$env:IP_AGENT_EMBEDDING_PROVIDER="bge_m3"
$env:IP_AGENT_EMBEDDING_MODEL="BAAI/bge-m3"
streamlit run frontend/app.py
```

For interview demos, `openai` is usually the quickest stable semantic embedding
path. For private IP workflows, `bge_m3` is a stronger long-term option because
technical disclosures can stay local.

If the machine has not downloaded BGE-M3 yet, the first run may download the
model and take longer. Use `IP_AGENT_EMBEDDING_PROVIDER=local` when you need a
fully offline smoke test without model downloads.

## Optional LLM API

The parser and keyword expansion agents can use an OpenAI-compatible LLM API.
See [docs/llm_configuration.md](docs/llm_configuration.md).
