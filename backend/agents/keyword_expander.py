from __future__ import annotations

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
    def run(self, disclosure: DisclosureAnalysis) -> KeywordSet:
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
