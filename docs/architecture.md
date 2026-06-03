# Architecture

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
