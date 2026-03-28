from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    timeout_s: float = 60.0


def generate_explanation_ollama(
    prompt: str,
    model: str,
    cfg: OllamaConfig | None = None,
) -> str | None:
    """
    Uses Ollama's local HTTP API:
      POST /api/generate { model, prompt, stream:false }
    """
    if cfg is None:
        cfg = OllamaConfig()

    url = f"{cfg.base_url}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(url, json=payload, timeout=cfg.timeout_s)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response")

