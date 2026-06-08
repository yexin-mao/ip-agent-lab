from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from backend.llm.client import LLMClient
from backend.retrieval.hybrid import tokenize
from backend.schemas.models import DisclosureAnalysis


SECTION_ALIASES: Dict[str, List[str]] = {
    "title": ["title", "invention title", "发明名称", "名称", "标题"],
    "technical_field": ["technical field", "field", "技术领域", "所属领域"],
    "problem": ["background", "prior art", "problem", "technical problem", "背景技术", "背景", "现有技术", "技术问题"],
    "solution": ["technical solution", "solution", "summary", "发明内容", "技术方案", "解决方案", "方案"],
    "innovation_points": ["innovation points", "novel points", "key innovations", "创新点", "核心创新", "主要创新", "发明点"],
    "effects": ["technical effects", "effects", "benefits", "技术效果", "有益效果", "效果"],
    "applications": ["applications", "use cases", "scenarios", "应用场景", "应用", "使用场景"],
}


@dataclass(frozen=True)
class NormalizedDisclosureText:
    original_text: str
    normalized_text: str
    language_hint: str
    detected_sections: List[str]


class DisclosureParserAgent:
    def __init__(self, llm_client: LLMClient | None = None):
        self.llm_client = llm_client or LLMClient()

    def run(self, text: str) -> DisclosureAnalysis:
        normalized = self._normalize_text(text)
        llm_result = self._run_llm(normalized.normalized_text)
        if llm_result:
            return self._with_quality(llm_result, parser_mode="llm")

        sections = self._split_sections(normalized.normalized_text)
        title = sections.get("title") or self._guess_title(normalized.normalized_text)
        problem = sections.get("problem") or self._infer_problem(normalized.normalized_text)
        solution = sections.get("solution") or self._infer_solution(normalized.normalized_text)
        innovation_points = self._extract_innovation_points(
            sections.get("innovation_points") or "",
            normalized.normalized_text,
        )
        parser_mode = "rule" if sections else "plain_text_fallback"

        disclosure = DisclosureAnalysis(
            title=title.strip() or "Untitled invention",
            technical_field=(sections.get("technical_field") or "Not specified").strip(),
            problem=problem.strip(),
            solution=solution.strip(),
            innovation_points=innovation_points,
            effects=self._extract_list(sections.get("effects") or ""),
            applications=self._extract_list(sections.get("applications") or ""),
            key_terms=self._extract_key_terms(normalized.normalized_text),
        )
        return self._with_quality(disclosure, parser_mode=parser_mode)

    def _normalize_text(self, text: str) -> NormalizedDisclosureText:
        original = text or ""
        normalized = original.replace("\r\n", "\n").replace("\r", "\n")
        normalized = normalized.replace("：", ":")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        normalized = re.sub(r"^\s*#{1,6}\s+", "", normalized, flags=re.MULTILINE)
        normalized = normalized.strip()
        detected_sections = list(self._split_sections(normalized).keys()) if normalized else []
        language_hint = "zh" if re.search(r"[\u4e00-\u9fff]", normalized) else "en"
        return NormalizedDisclosureText(
            original_text=original,
            normalized_text=normalized,
            language_hint=language_hint,
            detected_sections=detected_sections,
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
            stripped = re.sub(r"^\s*(\d+[\).、]|[-*])\s*", "", stripped)
            normalized = stripped.rstrip(":").lower()
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
                return line.rstrip(":")
        return "Untitled invention"

    def _sentences(self, text: str) -> List[str]:
        normalized = (text or "").replace("\n", " ")
        parts = re.split(r"(?<=[.!?。！？])\s+", normalized.strip())
        return [part.strip() for part in parts if part.strip()]

    def _infer_problem(self, text: str) -> str:
        sentences = self._sentences(text)
        markers = ["problem", "challenge", "drawback", "conventional", "existing", "问题", "缺陷", "不足", "现有", "传统", "难以"]
        selected = [sentence for sentence in sentences if any(marker in sentence.lower() for marker in markers)]
        return " ".join((selected or sentences)[:3]) or "Not specified"

    def _infer_solution(self, text: str) -> str:
        sentences = self._sentences(text)
        markers = ["proposed", "method", "system", "solution", "comprising", "configured", "提出", "本发明", "方法", "系统", "方案", "包括", "配置"]
        selected = [sentence for sentence in sentences if any(marker in sentence.lower() for marker in markers)]
        if selected:
            return " ".join(selected[:4])
        if len(sentences) >= 4:
            return " ".join(sentences[2:6])
        return " ".join(sentences[:4]) or "Not specified"

    def _extract_innovation_points(self, section_text: str, full_text: str) -> List[str]:
        explicit = self._extract_list(section_text)
        if explicit:
            return explicit

        sentences = self._sentences(full_text)
        markers = [
            "novel", "innovative", "improve", "reduce", "optimize", "predict",
            "preconfigure", "adaptive", "创新", "改进", "降低", "提高",
            "优化", "预测", "预配置", "自适应",
        ]
        selected = [sentence for sentence in sentences if any(marker in sentence.lower() for marker in markers)]
        return selected[:5]

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
        terms = [term for term in tokenize(text) if len(term) > 2 and term not in stopwords]
        ranked = [term for term, _ in sorted(CounterLike(terms).items(), key=lambda kv: kv[1], reverse=True)]
        return ranked[:18]

    def _with_quality(self, disclosure: DisclosureAnalysis, parser_mode: str) -> DisclosureAnalysis:
        warnings = []
        score = 0

        if disclosure.title and disclosure.title != "Untitled invention":
            score += 1
        else:
            warnings.append("No clear invention title was detected.")

        if disclosure.technical_field and disclosure.technical_field != "Not specified":
            score += 1
        else:
            warnings.append("Technical field is missing or inferred weakly.")

        problem_min_length = self._min_meaningful_length(disclosure.problem, english_min=40, cjk_min=18)
        if disclosure.problem and disclosure.problem != "Not specified" and len(disclosure.problem) >= problem_min_length:
            score += 1
        else:
            warnings.append("Technical problem is short or missing.")

        solution_min_length = self._min_meaningful_length(disclosure.solution, english_min=60, cjk_min=24)
        if disclosure.solution and disclosure.solution != "Not specified" and len(disclosure.solution) >= solution_min_length:
            score += 1
        else:
            warnings.append("Technical solution is short or missing.")

        if len(disclosure.innovation_points) >= 2:
            score += 1
        elif disclosure.innovation_points:
            warnings.append("Only one innovation point was detected.")
        else:
            warnings.append("No explicit innovation points were detected.")

        if len(disclosure.key_terms) >= 6:
            score += 1
        else:
            warnings.append("Few technical key terms were extracted.")

        if parser_mode == "llm":
            score += 1
        elif parser_mode == "plain_text_fallback":
            warnings.append("Input did not match known disclosure section headings; plain-text fallback was used.")

        disclosure.parser_mode = parser_mode
        disclosure.parse_quality = "high" if score >= 6 else "medium" if score >= 4 else "low"
        disclosure.warnings = warnings[:6]
        return disclosure

    def _min_meaningful_length(self, text: str, english_min: int, cjk_min: int) -> int:
        return cjk_min if re.search(r"[\u4e00-\u9fff]", text or "") else english_min


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
