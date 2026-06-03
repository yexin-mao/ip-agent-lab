from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional


JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 60

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url and self.model and self.provider != "none")


def load_llm_config() -> LLMConfig:
    return LLMConfig(
        provider=os.getenv("IP_AGENT_LLM_PROVIDER", "none").strip().lower(),
        api_key=os.getenv("IP_AGENT_LLM_API_KEY", "").strip(),
        base_url=os.getenv("IP_AGENT_LLM_BASE_URL", "").strip().rstrip("/"),
        model=os.getenv("IP_AGENT_LLM_MODEL", "").strip(),
        timeout_seconds=int(os.getenv("IP_AGENT_LLM_TIMEOUT", "60")),
    )


class LLMClient:
    """Minimal OpenAI-compatible chat client.

    It intentionally avoids framework dependencies. Providers such as OpenAI,
    DeepSeek, Qwen-compatible gateways, and many local servers can be configured
    by setting base URL, model, and API key.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or load_llm_config()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def chat_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None

        try:
            text = self._chat(system_prompt, user_prompt, temperature)
            return extract_json_object(text)
        except Exception:
            return None

    def _chat(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP error {exc.code}: {body}") from exc

        data = json.loads(raw)
        return data["choices"][0]["message"]["content"]


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = JSON_OBJECT_PATTERN.search(cleaned)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
