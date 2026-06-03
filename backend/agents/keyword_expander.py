from __future__ import annotations

from backend.llm.client import LLMClient
from backend.schemas.models import DisclosureAnalysis, KeywordSet


DOMAIN_SYNONYMS = {
    "handover": ["mobility management", "cell switching", "handoff"],
    "beam": ["beam management", "beamforming", "beam measurement"],
    "latency": ["low latency", "delay reduction", "interruption time"],
    "prediction": ["predictive", "mobility prediction", "link degradation prediction"],
    "5g": ["nr", "new radio", "cellular network"],
    "resource": ["radio resource", "preconfiguration", "reservation"],
}


class KeywordExpansionAgent:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LLMClient()

    def run(self, disclosure: DisclosureAnalysis) -> KeywordSet:
        llm_result = self._run_llm(disclosure)
        if llm_result:
            return llm_result

        base_terms = []
        for term in disclosure.key_terms:
            if term not in base_terms:
                base_terms.append(term)

        for phrase in disclosure.innovation_points + disclosure.applications:
            for token in phrase.replace(",", " ").split():
                normalized = token.strip(" .;:()").lower()
                if len(normalized) > 3 and normalized not in base_terms:
                    base_terms.append(normalized)

        core_terms = base_terms[:14]
        synonyms = []
        for term in core_terms:
            synonyms.extend(DOMAIN_SYNONYMS.get(term.lower(), []))
        synonyms = list(dict.fromkeys(synonyms))[:14]

        english_terms = list(dict.fromkeys(core_terms + synonyms))[:20]
        query_groups = self._build_query_groups(disclosure, core_terms, synonyms)
        classification_hints = self._classification_hints(english_terms)

        return KeywordSet(
            core_terms=core_terms,
            synonyms=synonyms,
            english_terms=english_terms,
            query_groups=query_groups,
            classification_hints=classification_hints,
        )

    def _run_llm(self, disclosure: DisclosureAnalysis) -> KeywordSet | None:
        system_prompt = (
            "You are a patent search strategist. Generate professional prior-art "
            "search keywords and query groups from structured invention data. "
            "Return only valid JSON."
        )
        user_prompt = f"""
Generate patent-search keywords for this invention.

Invention:
- Title: {disclosure.title}
- Technical field: {disclosure.technical_field}
- Problem: {disclosure.problem}
- Solution: {disclosure.solution}
- Innovation points: {disclosure.innovation_points}
- Effects: {disclosure.effects}
- Applications: {disclosure.applications}
- Existing key terms: {disclosure.key_terms}

Return this exact JSON shape:
{{
  "core_terms": ["8-15 high impact technical terms"],
  "synonyms": ["synonyms, abbreviations, upper/lower concepts"],
  "english_terms": ["English terms for global patent and paper search"],
  "query_groups": ["4-8 practical search queries, not sentences"],
  "classification_hints": ["IPC/CPC hints with short labels"]
}}

Rules:
- Prefer terms likely to appear in patent abstracts or claims.
- Include mechanism terms, application-domain terms, and functional-effect terms.
- Include both broad and specific queries.
- Do not invent unsupported product names.
- Keep all arrays concise.
"""
        raw = self.llm_client.chat_json(system_prompt, user_prompt)
        if not raw:
            return None

        try:
            return KeywordSet(
                core_terms=normalize_list(raw.get("core_terms"))[:15],
                synonyms=normalize_list(raw.get("synonyms"))[:20],
                english_terms=normalize_list(raw.get("english_terms"))[:24],
                query_groups=normalize_list(raw.get("query_groups"))[:8],
                classification_hints=normalize_list(raw.get("classification_hints"))[:8],
            )
        except Exception:
            return None

    def _build_query_groups(self, disclosure: DisclosureAnalysis, core_terms, synonyms):
        groups = [
            disclosure.title,
            " ".join(core_terms[:6]),
            " ".join((core_terms[:4] + synonyms[:4])[:8]),
        ]
        if disclosure.applications:
            groups.append(" ".join(disclosure.applications[:2]))
        if disclosure.innovation_points:
            groups.append(" ".join(disclosure.innovation_points[:2]))
        return [group for group in groups if group.strip()]

    def _classification_hints(self, terms):
        joined = " ".join(terms).lower()
        hints = []
        if any(word in joined for word in ["handover", "mobility", "cell", "beam", "5g", "radio"]):
            hints.extend(["H04W36/00 mobility management", "H04W72/00 radio resource management"])
        if any(word in joined for word in ["latency", "packet", "reliability"]):
            hints.append("H04W28/02 traffic and QoS control")
        return hints or ["Classification requires manual review"]


def normalize_list(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.replace("；", ";").split(";") if part.strip()]
    return []
