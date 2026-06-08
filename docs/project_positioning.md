# Project Positioning

## One-Sentence Pitch

IP AgentLab is an AI Agent and RAG demo for patent novelty search and early FTO
analysis, turning technical disclosures into prior-art evidence, risk analysis,
and reviewable reports.

## Target Scenario

The target user is an IP analyst, patent engineer, or R&D team member who needs
to quickly understand whether a new invention disclosure may overlap with known
prior art.

The workflow is designed for early review:

```text
technical disclosure
  -> invention structure
  -> keyword strategy
  -> patent evidence retrieval
  -> novelty/FTO comparison
  -> report for human review
```

The system should help users answer:

- What is the invention trying to protect?
- Which technical features are likely to matter for search?
- Which prior-art patents contain similar evidence?
- Which evidence chunks support the overlap?
- Which innovation points look high-risk, partial, or not covered?
- What should a human reviewer inspect next?

## Current Version

The current implementation is a local MVP. It uses:

- Streamlit for the demo interface.
- Pydantic models for structured intermediate results.
- Local sample patents for reproducible runs.
- BM25 and hashing-vector retrieval for hybrid recall.
- RRF fusion to merge lexical and vector search results.
- Rule-based comparison to assign novelty risk.
- Markdown report generation.

This makes the demo easy to run, but it is not yet a production-grade RAG or
patent analysis system.

## Intended Technical Story

The project should demonstrate three layers:

```text
Product layer
  patent novelty search, FTO analysis, evidence matrix, report export

Agent workflow layer
  disclosure parser, keyword strategist, retriever, evidence judge,
  novelty comparator, FTO analyzer, report writer

RAG infrastructure layer
  document loaders, chunking, metadata, embeddings, vector index,
  BM25 recall, hybrid retrieval, reranking, evaluation
```

## Framework Direction

The recommended upgrade path is:

1. Use LlamaIndex for the RAG layer.
2. Use LangGraph for workflow orchestration after retrieval quality is stable.

Clear ownership keeps the project understandable:

```text
LlamaIndex
  document ingestion, chunking, metadata, embedding, vector index,
  retriever, reranker

LangGraph
  stateful workflow, agent nodes, conditional retry paths,
  human-review checkpoints
```

The project should avoid mixing multiple agent abstractions at the same layer.
For example, LlamaIndex should not also own the full agent workflow if LangGraph
is used for orchestration.

## Staged Upgrade Plan

### Stage 1: Positioning and Scope

- Clarify product positioning and target user.
- Document the current scope and limitations.
- Explain the intended LlamaIndex + LangGraph direction.
- Make the README suitable for interview review.

### Stage 2: LlamaIndex RAG Layer

- Add patent/disclosure document loaders.
- Replace hashing embeddings with real embeddings.
- Add vector storage through FAISS, Chroma, or Qdrant.
- Preserve BM25 recall and combine it with vector retrieval.
- Return evidence chunks with stable source IDs and metadata.

### Stage 3: Evidence-Grounded Analysis

- Add an evidence judge that scores whether each chunk supports each invention
  point.
- Produce an innovation-point matrix.
- Require every high-risk conclusion to cite supporting evidence.
- Generate structured JSON before rendering human-readable text.

### Stage 4: LangGraph Workflow

- Replace the sequential orchestrator with a stateful graph.
- Add retry branches when evidence is weak.
- Add optional human review checkpoints.
- Support separate novelty-search and FTO-analysis paths.

### Stage 5: Evaluation and Demo Packaging

- Add a small labeled evaluation set.
- Report recall@k, precision@k, evidence hit rate, and rerank lift.
- Add tests, linting, and a reproducible startup path.
- Prepare a demo video, sample report, and architecture diagram.

## Interview Demo Narrative

Suggested explanation:

> I built IP AgentLab as a vertical AI Agent project for patent novelty search.
> The current MVP runs locally and demonstrates the full workflow from disclosure
> parsing to prior-art evidence retrieval and report generation. The next
> iteration upgrades the retrieval layer with LlamaIndex and production
> embeddings, then uses LangGraph to manage a multi-agent workflow with evidence
> judging, novelty comparison, and FTO claim-chart generation.

