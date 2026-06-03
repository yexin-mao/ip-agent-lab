# IP AgentLab

IP AgentLab is a first-version portfolio MVP for intellectual-property AI agent work.
It demonstrates a patent novelty search workflow:

1. Parse a technical disclosure.
2. Extract invention structure and innovation points.
3. Expand search keywords and classification hints.
4. Retrieve chunk-level prior-art evidence with BM25 + vector hybrid retrieval.
5. Compare invention points with retrieved evidence chunks.
6. Generate a Markdown novelty search report.

This MVP intentionally uses a local corpus and rule-based scoring so the complete
workflow runs without cloud services. The architecture leaves clear extension
points for external patent APIs, Qdrant, production embeddings, rerankers, and
LLM-based agents.

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

## Planned Extensions

- Patent API integration
- Hybrid BM25 + vector search with Qdrant
- LLM structured extraction and relevance judge
- FTO claim chart generation
- Design FTO image retrieval with CLIP/SigLIP
- Evaluation set for recall@k and precision@k

## Optional LLM API

The parser and keyword expansion agents can use an OpenAI-compatible LLM API.
See [docs/llm_configuration.md](docs/llm_configuration.md).
