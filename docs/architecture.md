# Architecture

```text
Streamlit Frontend
  -> NoveltyOrchestrator
      -> DisclosureParserAgent
      -> KeywordExpansionAgent
      -> PatentSearchAgent
          -> LocalHybridRetriever
      -> PriorArtCompareAgent
      -> ReportAgent
  -> Markdown Report
```

## Agent Responsibilities

- `DisclosureParserAgent`: converts raw disclosure text into structured invention information.
- `KeywordExpansionAgent`: generates core terms, synonyms, query groups, and classification hints.
- `PatentSearchAgent`: retrieves candidate prior art.
- `PriorArtCompareAgent`: compares retrieved patents against innovation points and assigns risk.
- `ReportAgent`: produces a reviewable novelty report.

## Extension Roadmap

Replace `LocalHybridRetriever` with a production retrieval stack:

```text
query expansion
  -> BM25 keyword recall
  -> vector recall with Qdrant
  -> metadata filters, classification filters, jurisdiction filters
  -> reranker
  -> LLM judge with evidence extraction
```

Add future modules:

- FTO claim parsing and claim chart generation.
- Design FTO image search with CLIP/SigLIP.
- Communication-domain document parsing for 3GPP/RFC/technical standards.
