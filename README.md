# IP AgentLab

IP AgentLab is a portfolio project for intellectual-property AI agent engineering.
It focuses on patent novelty search and future FTO workflows for communication
and technology disclosures.

The project demonstrates an end-to-end AI-assisted prior-art analysis workflow:

1. Parse a technical disclosure.
2. Extract invention structure and innovation points.
3. Expand search keywords and classification hints.
4. Retrieve chunk-level prior-art evidence with BM25 + vector hybrid retrieval.
5. Compare invention points with retrieved evidence chunks.
6. Generate a Markdown novelty search report.

The current version is intentionally small and local-first: it runs with a sample
patent corpus and deterministic scoring so the complete demo works without cloud
services. The architecture is designed to be upgraded into a stronger interview
demo with LlamaIndex-based document indexing, production embeddings, vector
storage, reranking, LangGraph-based agent orchestration, and FTO claim-chart
generation.

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
- RRF result fusion
- Streamlit product demo
- Structured agent workflow
- Markdown report generation

## Current Limitations

- The vector path uses a deterministic hashing embedding fallback, not a
  production semantic embedding model.
- The sample corpus is intentionally small and should be expanded or connected
  to patent APIs for realistic evaluation.
- Prior-art comparison currently uses rule-based overlap scoring; it should be
  upgraded with an evidence-grounded LLM judge.
- The workflow is currently orchestrated by Python classes rather than a graph
  runtime such as LangGraph.
- FTO claim-chart generation is planned but not implemented yet.

## Planned Extensions

- LlamaIndex document ingestion, indexing, hybrid retrieval, and reranking.
- LangGraph workflow orchestration for multi-agent state management.
- Patent API integration
- Hybrid BM25 + vector search with Qdrant
- LLM structured extraction and relevance judge
- FTO claim chart generation
- Design FTO image retrieval with CLIP/SigLIP
- Evaluation set for recall@k and precision@k

See [docs/project_positioning.md](docs/project_positioning.md) for the project
positioning and staged upgrade plan.

## Optional LLM API

The parser and keyword expansion agents can use an OpenAI-compatible LLM API.
See [docs/llm_configuration.md](docs/llm_configuration.md).
