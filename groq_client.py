from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from groq import Groq

from base_client import BaseLLMClient


class FreeLLMPreferenceClient(BaseLLMClient):
    """Groq-backed LLM preference client."""

    def __init__(
        self,
        provider: str = "groq",
        api_key: Optional[str] = None,
        model_name: str = "llama-3.3-70b-versatile",
        simple: bool = False,
        rate_limit_delay: float = 0.1,
        max_tokens: int = 8192,
        max_retries: int = 20,
        default_seed: Optional[int] = 12345,
        enable_logprobs: bool = False,
        logprobs_k: int = 10,
    ):
        super().__init__(
            model_name=model_name,
            simple=simple,
            rate_limit_delay=rate_limit_delay,
            max_tokens=max_tokens,
            max_retries=max_retries,
            default_seed=default_seed,
        )
        self.provider = provider
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable or api_key parameter required")
        self.client = Groq(api_key=self.api_key)
        self.enable_logprobs = enable_logprobs
        self.logprobs_k = min(int(logprobs_k), 20)  # Groq caps at 20
        self._last_logprobs: Optional[Dict[str, Any]] = None

    def _extract_logprobs(self, response) -> Optional[Dict[str, Any]]:
        """
        Extract token logprobs from a Groq response into the same dict format
        used by GeminiPreferenceClient, so _extract_pl_logprobs() works unchanged.

        Groq logprobs structure:
          response.choices[0].logprobs.content[i].token       — chosen token at position i
          response.choices[0].logprobs.content[i].logprob     — its log-probability
          response.choices[0].logprobs.content[i].top_logprobs[j].token   — j-th best token
          response.choices[0].logprobs.content[i].top_logprobs[j].logprob — its log-probability

        Output mirrors GeminiPreferenceClient._extract_logprobs():
          {
            "chosen_tokens": [{"token": str, "log_probability": float}, ...],
            "top_candidates": [[{"token": str, "log_probability": float}, ...], ...]
          }
        """
        try:
            choice = response.choices[0]
            lp = getattr(choice, "logprobs", None)
            if not lp or not lp.content:
                return None

            chosen_tokens = []
            top_candidates_list = []
            for pos in lp.content:
                chosen_tokens.append({
                    "token": pos.token,
                    "log_probability": pos.logprob,
                })
                alts = [
                    {"token": t.token, "log_probability": t.logprob}
                    for t in (pos.top_logprobs or [])
                ]
                top_candidates_list.append(alts)

            return {"chosen_tokens": chosen_tokens, "top_candidates": top_candidates_list}
        except Exception:
            return None

    def _post_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: Optional[int],
        stop: Optional[List[str]],
    ) -> str:
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if stop:
            params["stop"] = stop
        if seed is not None:
            params["seed"] = int(seed)
        if self.enable_logprobs:
            params["logprobs"] = True
            params["top_logprobs"] = self.logprobs_k

        # Reset before each call
        self._last_logprobs = None

        def _call():
            last_err: Optional[Exception] = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = self.client.chat.completions.create(**params)
                    if self.enable_logprobs:
                        self._last_logprobs = self._extract_logprobs(response)
                    return response.choices[0].message.content
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    # Rate-limit errors propagate to the outer retry handler
                    if "rate" in msg or "429" in msg:
                        raise
                    # Server errors get exponential backoff
                    if any(code in msg for code in ["500", "502", "503", "504"]):
                        self._sleep_backoff(attempt)
                        continue
                    if attempt < self.max_retries:
                        self._sleep_backoff(attempt)
                        continue
            raise last_err if last_err else RuntimeError("Unknown error in _post_chat")

        return self._with_rate_limit_retry(_call)


_CLIENT = None


def get_local_client(
    model_id: str = "llama-3.3-70b-versatile",
    *,
    force_full_precision: bool = None,  # compatibility, unused
):
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is required")

    model_name = os.getenv("GROQ_MODEL_NAME", model_id)
    print(f"[groq_client] Using Groq API with model: {model_name}")

    _CLIENT = FreeLLMPreferenceClient(
        provider="groq",
        api_key=api_key,
        model_name=model_name,
        simple=False,
        rate_limit_delay=0.1,
        max_tokens=8192,
    )
    return _CLIENT


__all__ = ["FreeLLMPreferenceClient", "get_local_client"]
