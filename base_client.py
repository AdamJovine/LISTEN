"""Abstract base class for LLM preference clients."""
from __future__ import annotations

import random
import re
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

RATE_LIMIT_RETRIES = 6
RATE_LIMIT_WAIT_SECONDS = 5
RATE_LIMIT_MAX_WAIT_SECONDS = 90


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "rate" in msg or "429" in msg or ("resource" in msg and "exhausted" in msg)


_RETRY_HINT_RE = re.compile(
    r"(?:try again in|retry in|retry[_ ]?delay[^0-9]*?)\s*(\d+(?:\.\d+)?)\s*(ms|s|seconds?)",
    re.IGNORECASE,
)


def _parse_retry_after(exc: Exception) -> Optional[float]:
    """Extract the suggested retry-after (seconds) from a rate-limit error message."""
    m = _RETRY_HINT_RE.search(str(exc))
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    return value / 1000.0 if unit == "ms" else value


class BaseLLMClient(ABC):
    """Shared interface and utilities for LLM preference clients."""

    def __init__(
        self,
        model_name: str,
        simple: bool = False,
        rate_limit_delay: float = 0.1,
        max_tokens: int = 8192,
        max_retries: int = 20,
        default_seed: Optional[int] = 12345,
    ):
        self.model_name = model_name
        self.model_id = model_name
        self.simple = simple
        self.rate_limit_delay = rate_limit_delay
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.default_seed = default_seed

        self.default_temperature = 0.0
        self.default_top_p = 1.0
        self.default_max_new_tokens = max_tokens

    def _sleep_backoff(self, attempt: int):
        base_delay = min(2 ** (attempt - 1), 8)
        jitter = random.random() * 0.25
        time.sleep(base_delay + jitter + self.rate_limit_delay)

    def _with_rate_limit_retry(self, fn):
        """Call *fn*; on rate-limit errors wait the API-suggested delay (or
        exponential backoff if no hint) and retry up to RATE_LIMIT_RETRIES times."""
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                return fn()
            except Exception as e:
                if attempt >= RATE_LIMIT_RETRIES or not _is_rate_limit_error(e):
                    raise
                hint = _parse_retry_after(e)
                if hint is not None:
                    delay = min(max(hint + 0.5, 1.0), RATE_LIMIT_MAX_WAIT_SECONDS)
                    reason = f"API hint {hint:.2f}s"
                else:
                    delay = min(RATE_LIMIT_WAIT_SECONDS * (2 ** attempt), RATE_LIMIT_MAX_WAIT_SECONDS)
                    reason = "exp backoff"
                delay += random.random() * 0.5
                print(
                    f"[rate-limit] attempt {attempt + 1}/{RATE_LIMIT_RETRIES + 1} "
                    f"– waiting {delay:.1f}s ({reason})"
                )
                time.sleep(delay)

    @staticmethod
    def _apply_stop(text: str, stop: Optional[List[str]]) -> str:
        if not stop:
            return text
        out = text
        for s in stop:
            if not s:
                continue
            i = out.find(s)
            if i != -1:
                out = out[:i]
        return out

    def _parse_choice(self, text: str) -> str:
        """Extract A/B choice from oracle response. Override for custom parsing."""
        return text[0].upper() if text else "A"

    def _resolve_params(
        self,
        temperature: Optional[float],
        top_p: Optional[float],
        max_new_tokens: Optional[int],
        seed: Optional[int],
    ):
        return (
            self.default_temperature if temperature is None else temperature,
            self.default_top_p if top_p is None else top_p,
            self.default_max_new_tokens if max_new_tokens is None else max_new_tokens,
            self.default_seed if seed is None else seed,
        )

    @abstractmethod
    def _post_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: Optional[int],
        stop: Optional[List[str]],
    ) -> str: ...

    def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> str:
        temperature, top_p, max_new_tokens, seed = self._resolve_params(
            temperature, top_p, max_new_tokens, seed
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._post_chat(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
            stop=stop,
        )
        return self._apply_stop(text.strip(), stop)

    def call_oracle(
        self,
        prompt: str,
        sched_a,
        sched_b,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> Tuple[str, str]:
        temperature, top_p, max_new_tokens, seed = self._resolve_params(
            temperature, top_p, max_new_tokens, seed
        )
        messages = [{"role": "user", "content": prompt}]
        text = self._post_chat(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
            stop=stop,
        )
        text = self._apply_stop(text.strip(), stop)
        choice = self._parse_choice(text)
        return choice, text
