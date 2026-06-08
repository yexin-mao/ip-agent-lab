# Architecture

IP AgentLab is organized around three layers:

```text
Streamlit product demo
  -> agent workflow
      -> RAG retrieval infrastructure
```

The current implementation is local-first so it can be demonstrated without
external services. The intended production-style direction is to keep the same
business workflow while replacing the retrieval and orchestration internals with
LlamaIndex and LangGraph.

```text
Streamlit Frontend
  -> NoveltyOrchestrator
      -> DisclosureParserAgent
      -> KeywordExpansionAgent
      -> PatentSearchAgent
          -> HybridRAGRetriever
              -> PatentChunker
              -> BM25Index
              -> InMemoryVectorIndex
              -> RRF fusion
      -> PriorArtCompareAgent
      -> ReportAgent
  -> Markdown Report
```

## Business Workflow

```text
Technical disclosure
  -> invention summary and innovation points
  -> search terms and classification hints
  -> prior-art evidence retrieval
  -> novelty risk comparison
  -> report for human review
```

The system is intended to support early patent novelty search first. The planned
extension is FTO analysis, where claim elements are compared against product or
technical features and rendered as a claim chart.

## Agent Responsibilities

- `DisclosureParserAgent`: converts raw disclosure text into structured invention information.
- `KeywordExpansionAgent`: generates core terms, synonyms, query groups, and classification hints.
- `PatentSearchAgent`: retrieves chunk-level prior-art evidence.
- `PriorArtCompareAgent`: compares retrieved evidence chunks against innovation points and assigns risk.
- `ReportAgent`: produces a reviewable novelty report.

## Current RAG Retrieval Layer

The current retrieval layer is chunk-based:

```text
PatentDocument[]
  -> PatentChunker
  -> PatentChunk[]
      -> BM25Index
      -> HashingEmbeddingModel + InMemoryVectorIndex
  -> HybridRAGRetriever
      -> BM25 recall
      -> vector recall
      -> Reciprocal Rank Fusion
      -> group evidence chunks by patent_id
```

The system now returns `GroupedPatentEvidence`, where each patent contains
specific `EvidenceChunkResult` objects with section, score, matched terms, and
evidence text.

The local vector path uses deterministic hashing embeddings so the demo runs
without downloading models or starting external services.

## Planned LlamaIndex Retrieval Layer

LlamaIndex should own the document and retrieval layer:

```text
patent PDFs / text / API records
  -> document loaders
  -> node parser and metadata extraction
  -> embedding model
  -> vector store
  -> BM25 retriever
  -> hybrid retrieval
  -> reranker
  -> evidence chunks
```

This keeps indexing, chunking, metadata filtering, retriever configuration, and
reranking in one RAG-focused layer.

## Planned LangGraph Workflow Layer

LangGraph should own multi-agent orchestration:

```text
parse_disclosure
  -> expand_keywords
  -> retrieve_evidence
  -> judge_evidence
  -> compare_novelty
  -> generate_report
```

Future branches:

```text
weak evidence -> expand query -> retrieve again
high novelty risk -> build claim chart
FTO mode -> compare product features against claim elements
human review -> apply risk override -> regenerate report
```

The key boundary is simple: LlamaIndex retrieves evidence; LangGraph decides
which agent step runs next.

## Production Extension Roadmap

Replace the local vector path with a production retrieval stack:

```text
query expansion
  -> BM25 keyword recall
  -> BGE/OpenAI embedding
  -> vector recall with Qdrant
  -> metadata filters, classification filters, jurisdiction filters
  -> reranker
  -> LLM judge with evidence extraction
```

Add future modules:

- FTO claim parsing and claim chart generation.
- Design FTO image search with CLIP/SigLIP.
- Communication-domain document parsing for 3GPP/RFC/technical standards.
