from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import httpx
import yaml


class LLMClient:
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        self.config = self._load_config(config_path)
        self.provider: str = self.config.get("llm", {}).get("provider", "ollama")
        self.model: str = self.config.get("llm", {}).get("model", "llama3")
        self.endpoint: str = self.config.get("llm", {}).get("endpoint", "http://localhost:11434")

    @staticmethod
    def _load_config(config_path: str | Path) -> dict:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 768,
    ) -> str:
        if self.provider.lower() == "ollama":
            return self._generate_with_ollama(prompt, system, temperature, max_tokens)
        raise ValueError(f"LLM provider '{self.provider}'는 지원되지 않습니다.")

    def _generate_with_ollama(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        url = self._join_url(self.endpoint, "/api/generate")
        payload = {
            "model": self.model,
            "prompt": prompt,
            # Ollama는 system 프롬프트를 messages 기반 대화가 아닌 generate API에서도 전달 가능
            "system": system or "",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        # Ollama non-stream 응답은 단일 JSON에 `response`를 포함
        text = data.get("response", "").strip()
        return text

    @staticmethod
    def _join_url(base: str, path: str) -> str:
        return base.rstrip("/") + path


__all__ = ["LLMClient"]

