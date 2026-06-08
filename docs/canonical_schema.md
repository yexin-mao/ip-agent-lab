# Canonical Disclosure Schema

`DisclosureAnalysis` is the current canonical disclosure schema for IP AgentLab.
It is the stable internal representation used after parsing a raw technical
disclosure.

The input can be structured, unstructured, Chinese, English, or LLM-generated.
The rest of the pipeline should not depend on the original format. It should
depend on this canonical shape:

```text
title
technical_field
problem
solution
innovation_points
effects
applications
key_terms
```

## Why This Schema Exists

Patent novelty search needs more than a single query string. The system needs to
separate:

- what the invention is
- which field it belongs to
- what technical problem it addresses
- what solution it proposes
- which innovation points should be compared against prior art
- what technical effects and application scenarios can guide search
- which terms should feed query expansion

That is why raw disclosure text is normalized into `DisclosureAnalysis` before
keyword expansion, retrieval, comparison, and report generation.

## Parser Metadata

The current model also carries lightweight parser metadata:

```text
parser_mode
parse_quality
warnings
```

These fields describe how the canonical schema was produced and how reliable the
result appears to be.

```text
parser_mode:
  llm
  rule
  plain_text_fallback

parse_quality:
  high
  medium
  low
```

Future versions can split this into a dedicated `DisclosureParseResult` object
that wraps the canonical disclosure and parser metadata separately.

## Future Extensions

Likely next additions:

- `parse_confidence`: numeric confidence for workflow control.
- field-level warnings for UI display.
- object-shaped innovation points with IDs, importance, and per-point keywords.
- technical entities such as components, mechanisms, actions, and effects.
- raw and normalized input tracking for auditability.
