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
  -> NoveltyWorkflow
      -> DisclosureParserAgent
      -> KeywordExpansionAgent
      -> PatentSearchAgent
      -> EvidenceJudgeAgent
      -> NoveltyRAGGenerationAgent
      -> NoveltyReportAgent
  -> TechnicalFTOWorkflow
      -> DisclosureParserAgent
      -> KeywordExpansionAgent
      -> PatentSearchAgent
      -> FTOClaimChartAgent
      -> FTORAGGenerationAgent
      -> FTOReportAgent
```

Design FTO runs as a parallel workflow:

```text
Streamlit Design FTO
  -> DesignFTOAgent
      -> load sample design-patent image corpus
      -> PerceptualImageFeatureExtractor
      -> visual similarity ranking
      -> visual risk analysis
  -> Streamlit Visual Results + Markdown Design FTO Report
```

Both technical workflows share the retrieval layer:

```text
PatentDocument[]
  -> PatentChunker
  -> LlamaIndexPatentRetriever / HybridRAGRetriever
      -> BM25Index
      -> VectorStoreIndex / InMemoryVectorIndex
      -> RRF fusion
  -> GroupedPatentEvidence
  -> EvidenceContextBuilder
  -> grounded RAG generation with citations
```

## Business Workflows

```text
Novelty Search
  -> technical disclosure
  -> invention summary and innovation points
  -> search terms and classification hints
  -> prior-art evidence retrieval
  -> innovation-point evidence matrix
  -> RAG-generated novelty analysis
  -> novelty report for human review
```

```text
Technical FTO
  -> implementation / product technical features
  -> claim-focused search terms
  -> claim evidence retrieval
  -> FTO claim chart
  -> RAG-generated FTO analysis
  -> FTO report for human review
```

The design FTO path is separate from technical FTO because the legal and
technical evidence are different: technical FTO maps text features to claims,
while design FTO compares overall visual impression against design drawings.

## Agent Responsibilities

- `DisclosureParserAgent`: converts raw disclosure text into structured invention information.
- `KeywordExpansionAgent`: generates core terms, synonyms, query groups, and classification hints.
- `PatentSearchAgent`: retrieves chunk-level prior-art evidence.
- `EvidenceJudgeAgent`: maps each innovation point to the strongest retrieved evidence chunk.
- `FTOClaimChartAgent`: maps disclosure elements to retrieved claim chunks for an initial FTO chart.
- `NoveltyRAGGenerationAgent`: generates grounded novelty analysis from retrieved evidence.
- `FTORAGGenerationAgent`: generates grounded FTO analysis from retrieved claim evidence.
- `NoveltyReportAgent`: produces a reviewable novelty report.
- `FTOReportAgent`: produces a reviewable technical FTO report.
- `DesignFTOAgent`: retrieves visually similar sample design-patent images and generates a design FTO report.

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

The first implementation is available behind a backend switch:

```text
IP_AGENT_RETRIEVER=local
  -> HybridRAGRetriever
  -> BM25Index + InMemoryVectorIndex

IP_AGENT_RETRIEVER=llamaindex
  -> LlamaIndexPatentRetriever
  -> TextNode[] with patent metadata
  -> VectorStoreIndex
  -> NodeWithScore[]
  -> GroupedPatentEvidence
```

The LlamaIndex retriever preserves the existing application contract by mapping
retrieved nodes back into `PatentChunk`, `EvidenceChunkResult`, and
`GroupedPatentEvidence`. That keeps the Streamlit UI, comparison agent, and
report agent unchanged while the retrieval internals evolve.

## Planned LangGraph Workflow Layer

LangGraph should own multi-agent orchestration:

```text
parse_disclosure
  -> expand_keywords
  -> retrieve_evidence
  -> judge_evidence
  -> build_fto_claim_chart
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
- Design FTO image search with CLIP/SigLIP and real design-patent image ingestion.
- Communication-domain document parsing for 3GPP/RFC/technical standards.
