# Evaluation Plan

The MVP is designed so retrieval and agent quality can be measured later.

## Metrics

- `recall@k`: whether known relevant prior art appears in the top k results.
- `precision@k`: proportion of top k results judged relevant by humans.
- rerank lift: improvement after reranking compared with baseline retrieval.
- evidence quality: whether each risk conclusion includes supporting text.
- human override rate: how often reviewers change AI risk labels.

## Feedback Loop

Each result should eventually support these labels:

- relevant / irrelevant
- add to report
- risk override
- false positive reason
- missing prior art note

These labels can be used to tune prompts, query expansion, reranking, and retrieval filters.
