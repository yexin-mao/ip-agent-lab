from __future__ import annotations

import re
from typing import Dict, List

from backend.llm.client import LLMClient
from backend.retrieval.hybrid import tokenize
from backend.schemas.models import DisclosureAnalysis


SECTION_ALIASES: Dict[str, List[str]] = {
    "title": ["title", "发明名称", "名称"],
    "technical_field": ["technical field", "技术领域"],
    "problem": ["background", "背景", "现有技术", "problem"],
    "solution": ["technical solution", "技术方案", "solution"],
    "innovation_points": ["innovation points", "创新点", "核心创新"],
    "effects": ["technical effects", "技术效果", "效果"],
    "applications": ["applications", "应用场景", "应用"],
}


class DisclosureParserAgent:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LLMClient()

    def run(self, text: str) -> DisclosureAnalysis:
        llm_result = self._run_llm(text)
        if llm_result:
            return llm_result

        sections = self._split_sections(text)
        title = sections.get("title") or self._guess_title(text)
        solution = sections.get("solution") or self._first_sentences(text, 4)
        key_terms = self._extract_key_terms(text)

        return DisclosureAnalysis(
            title=title.strip() or "Untitled invention",
            technical_field=(sections.get("technical_field") or "Not specified").strip(),
            problem=(sections.get("problem") or self._first_sentences(text, 3)).strip(),
            solution=solution.strip(),
            innovation_points=self._extract_list(sections.get("innovation_points") or solution),
            effects=self._extract_list(sections.get("effects") or ""),
            applications=self._extract_list(sections.get("applications") or ""),
            key_terms=key_terms,
        )

    def _run_llm(self, text: str) -> DisclosureAnalysis | None:
        system_prompt = (
            "You are a patent search professional. Extract a technical disclosure "
            "into strict JSON for patent novelty search. Do not provide legal advice. "
            "Return only valid JSON."
        )
        user_prompt = f"""
Extract the disclosure below into this exact JSON shape:
{{
  "title": "concise invention title",
  "technical_field": "technical field",
  "problem": "problem or unmet technical need",
  "solution": "technical solution summary",
  "innovation_points": ["point 1", "point 2"],
  "effects": ["effect 1", "effect 2"],
  "applications": ["application 1", "application 2"],
  "key_terms": ["term1", "term2"]
}}

Rules:
- Keep arrays focused and technical.
- Extract 3-8 innovation points when possible.
- Extract 8-18 key technical terms.
- Use English terms if the disclosure is English; preserve Chinese terms if the disclosure is Chinese.
- If a field is missing, infer conservatively from the text.

Disclosure:
\"\"\"
{text[:14000]}
\"\"\"
"""
        raw = self.llm_client.chat_json(system_prompt, user_prompt)
        if not raw:
            return None

        try:
            return DisclosureAnalysis(
                title=str(raw.get("title") or "Untitled invention"),
                technical_field=str(raw.get("technical_field") or "Not specified"),
                problem=str(raw.get("problem") or "Not specified"),
                solution=str(raw.get("solution") or "Not specified"),
                innovation_points=normalize_str_list(raw.get("innovation_points")),
                effects=normalize_str_list(raw.get("effects")),
                applications=normalize_str_list(raw.get("applications")),
                key_terms=normalize_str_list(raw.get("key_terms"))[:18],
            )
        except Exception:
            return None

    def _split_sections(self, text: str) -> Dict[str, str]:
        lines = text.replace("\r\n", "\n").split("\n")
        current_key = None
        sections: Dict[str, List[str]] = {}

        for line in lines:
            stripped = line.strip()
            normalized = stripped.rstrip(":：").lower()
            matched_key = None
            for key, aliases in SECTION_ALIASES.items():
                if any(normalized == alias.lower() for alias in aliases):
                    matched_key = key
                    break
                for alias in aliases:
                    prefix = alias.lower() + ":"
                    if stripped.lower().startswith(prefix):
                        matched_key = key
                        remainder = stripped[len(prefix):].strip()
                        sections.setdefault(matched_key, [])
                        if remainder:
                            sections[matched_key].append(remainder)
                        break
                if matched_key:
                    break

            if matched_key:
                current_key = matched_key
                sections.setdefault(current_key, [])
            elif current_key and stripped:
                sections[current_key].append(stripped)

        return {key: "\n".join(value).strip() for key, value in sections.items()}

    def _guess_title(self, text: str) -> str:
        for line in text.splitlines():
            line = line.strip()
            if 8 <= len(line) <= 120:
                return line.rstrip(":：")
        return "Untitled invention"

    def _first_sentences(self, text: str, count: int) -> str:
        sentences = re.split(r"(?<=[.!?。！？])\s+", text.strip())
        return " ".join(sentences[:count])

    def _extract_list(self, text: str) -> List[str]:
        items = []
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*(\d+[\).、]|[-*])\s*", "", line).strip()
            if cleaned:
                items.append(cleaned)
        if len(items) <= 1 and text:
            items = [part.strip() for part in re.split(r";|；|\n", text) if part.strip()]
        return items[:8]

    def _extract_key_terms(self, text: str) -> List[str]:
        stopwords = {
            "the", "and", "for", "with", "method", "system", "this", "that",
            "from", "before", "after", "based", "using", "invention",
        }
        terms = [t for t in tokenize(text) if len(t) > 2 and t not in stopwords]
        ranked = [term for term, _ in sorted(CounterLike(terms).items(), key=lambda kv: kv[1], reverse=True)]
        return ranked[:18]


def CounterLike(items: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def normalize_str_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"\n|;|；", value) if part.strip()]
    return []
