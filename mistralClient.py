"""
Remote client for the Mistral LLM server.
- Discovers base URL from:
    1) env MISTRAL_API_URL (e.g., http://zabih-compute-01.orie.cornell.edu:8000)
    2) ~/.mistral_url
    3) ~/.mistral_node + ~/.mistral_port
    4) fallback http://127.0.0.1:8000 (works only on the GPU node)
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, Any, List
import os, time
from pathlib import Path
import httpx

_CLIENT = None

def _discover_base_url() -> str:
    # 1) explicit env
    url = os.environ.get("MISTRAL_API_URL")
    if url:
        return url.rstrip("/")

    # 2) ~/.mistral_url
    url_file = Path.home() / ".mistral_url"
    if url_file.exists():
        txt = url_file.read_text().strip()
        if txt:
            return txt.rstrip("/")

    # 3) ~/.mistral_node + ~/.mistral_port
    node_file = Path.home() / ".mistral_node"
    port_file = Path.home() / ".mistral_port"
    if node_file.exists() and port_file.exists():
        node = node_file.read_text().strip()
        port = port_file.read_text().strip()
        if node and port:
            return f"http://{node}:{port}"

    # 4) fallback
    return "http://127.0.0.1:8000"


class MistralRemoteClient:
    def __init__(self, base_url: str, timeout_s: float = 600.0, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout_s)

    def query(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.15,
        max_tokens: int = 512,
    ) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": "mistralai/Mistral-Small-24B-Instruct-2501",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_err = e
                time.sleep(0.5 * attempt)
        raise RuntimeError(
            f"Mistral API request failed after {self.max_retries} attempts: {last_err}"
        )


def get_mistral_client() -> MistralRemoteClient:
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    base_url = _discover_base_url()
    print(f"[remote_mistral_client] Using LLM API at: {base_url}")
    _CLIENT = MistralRemoteClient(base_url)
    return _CLIENT


if __name__ == "__main__":
    c = get_mistral_client()
    msg = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "Say hi in exactly 5 words."},
    ]
    out = c.query(msg, temperature=0.0, max_tokens=32)
    print("Output:", out)
