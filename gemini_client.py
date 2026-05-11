"""
Gemini preference client with the same lightweight interface as the Groq/local clients.

Uses the unified `google.genai` SDK throughout. The legacy
`google.generativeai` package is end-of-life and is no longer imported.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai.types import GenerateContentConfig

from base_client import BaseLLMClient


class GeminiPreferenceClient(BaseLLMClient):
    """Gemini-backed LLM preference client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-pro",
        simple: bool = False,
        rate_limit_delay: float = 0.1,
        max_tokens: int = 8192,
        max_retries: int = 4,
        default_seed: Optional[int] = 12345,
        enable_logprobs: bool = False,
        logprobs_k: int = 10,
        gcp_project: Optional[str] = None,
        gcp_location: str = "global",
    ):
        super().__init__(
            model_name=model_name,
            simple=simple,
            rate_limit_delay=rate_limit_delay,
            max_tokens=max_tokens,
            max_retries=max_retries,
            default_seed=default_seed,
        )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY (or api_key) is required")

        self.enable_logprobs = enable_logprobs
        self.logprobs_k = logprobs_k
        self._last_logprobs: Optional[Dict[str, Any]] = None

        # Logprobs are only available via Vertex AI. The public Gemini API
        # (api_key auth) does not return them.
        if self.enable_logprobs:
            self.gcp_project = gcp_project or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
            if not self.gcp_project:
                raise ValueError(
                    "Logprobs require Vertex AI. Set --gcp-project, GCP_PROJECT, or GOOGLE_CLOUD_PROJECT env var."
                )
            self.gcp_location = gcp_location or "us-central1"
            self._client = genai.Client(
                vertexai=True,
                project=self.gcp_project,
                location=self.gcp_location,
            )
            print(
                f"[GeminiPreferenceClient] Logprobs enabled via Vertex AI "
                f"(project={self.gcp_project}, location={self.gcp_location})"
            )
        else:
            self._client = genai.Client(api_key=self.api_key)

        self._FINAL_TAG_RE = re.compile(r"^\s*FINAL\s*[:=\-]?\s*([AB])\b", re.IGNORECASE | re.MULTILINE)
        self._seed_supported = True

    # ------------------------------------------------------------------ utils
    def _extract_logprobs(self, response) -> Optional[Dict[str, Any]]:
        """Extract logprobs data from a Gemini response object."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        cand0 = candidates[0]
        logprobs_result = getattr(cand0, "logprobs_result", None)
        if not logprobs_result:
            return None

        chosen_tokens = []
        top_candidates_list = []

        chosen_candidates = getattr(logprobs_result, "chosen_candidates", None) or []
        top_candidates_raw = getattr(logprobs_result, "top_candidates", None) or []

        for i, chosen in enumerate(chosen_candidates):
            chosen_tokens.append({
                "token": getattr(chosen, "token", ""),
                "log_probability": getattr(chosen, "log_probability", None),
            })
            if i < len(top_candidates_raw):
                alts = []
                for alt in (getattr(top_candidates_raw[i], "candidates", None) or []):
                    alts.append({
                        "token": getattr(alt, "token", ""),
                        "log_probability": getattr(alt, "log_probability", None),
                    })
                top_candidates_list.append(alts)

        return {
            "chosen_tokens": chosen_tokens,
            "top_candidates": top_candidates_list,
        }

    def _finish_reason_str(self, cand) -> str:
        fr = getattr(cand, "finish_reason", None)
        name = getattr(fr, "name", None)
        if name:
            return name
        try:
            return str(fr)
        except Exception:
            return repr(fr)

    def _extract_text_from_response(self, response) -> Tuple[str, Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
            pf = response.prompt_feedback
            meta["prompt_feedback"] = getattr(pf, "block_reason", None) or str(pf)

        text_chunks: List[str] = []
        candidates = getattr(response, "candidates", None) or []

        if candidates:
            cand0 = candidates[0]
            meta["finish_reason"] = self._finish_reason_str(cand0)
            if hasattr(cand0, "safety_ratings") and cand0.safety_ratings:
                try:
                    meta["safety_ratings"] = [getattr(r, "category", None) or str(r) for r in cand0.safety_ratings]
                except Exception:
                    meta["safety_ratings"] = [str(cand0.safety_ratings)]

        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    text_chunks.append(t)
                    continue
                inline = getattr(p, "inline_data", None)
                if inline:
                    data = getattr(inline, "data", None)
                    if isinstance(data, (bytes, bytearray)):
                        try:
                            text_chunks.append(data.decode("utf-8", errors="ignore"))
                        except Exception:
                            pass
                    elif isinstance(data, str):
                        text_chunks.append(data)

        return "".join(text_chunks).strip(), meta

    def _parse_choice(self, text: str) -> str:
        """Override: regex-based parsing for Gemini oracle responses."""
        matches = self._FINAL_TAG_RE.findall(text)
        if matches:
            return matches[-1].upper()
        kv = re.findall(r'"?final"?\s*[:=]\s*"?([AB])"?', text, flags=re.IGNORECASE)
        if kv:
            return kv[-1].upper()
        lone = re.findall(r"^\s*([AB])\s*$", text, flags=re.IGNORECASE | re.MULTILINE)
        if lone:
            return lone[-1].upper()
        return text[0].upper() if text and text[0] in ("A", "B", "a", "b") else "A"

    # ------------------------------------------------------------- config build
    def _build_config(
        self,
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: Optional[int],
        include_logprobs: bool,
    ) -> GenerateContentConfig:
        kwargs: Dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_output_tokens": int(max_tokens),
        }
        if seed is not None and self._seed_supported:
            kwargs["seed"] = int(seed)
        if include_logprobs:
            kwargs["response_logprobs"] = True
            kwargs["logprobs"] = self.logprobs_k
        return GenerateContentConfig(**kwargs)

    # ---------------------------------------------------------------- core call
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
        # Build prompt from messages
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role == "system":
                parts.append(f"[SYSTEM]\n{content}\n")
            elif role == "assistant":
                parts.append(f"[ASSISTANT]\n{content}\n")
            else:
                parts.append(f"{content}\n")
        base_prompt = "\n".join(parts).strip()

        wants_json = ("<END_JSON>" in base_prompt) or ("STRICT OUTPUT RULES" in base_prompt)
        guard = (
            "Reply with ONLY the JSON object. Keep it under 250 tokens. No code fences. End with <END_JSON>.\n\n"
            if wants_json
            else "Be concise. Keep output under 300 tokens.\n\n"
        )
        prompt = guard + base_prompt

        def _build_cfg(cap: int) -> GenerateContentConfig:
            return self._build_config(
                temperature=temperature,
                top_p=top_p,
                max_tokens=cap,
                seed=seed,
                include_logprobs=self.enable_logprobs,
            )

        def _call(stream: bool, prompt_text: str, cap: int) -> Tuple[str, Dict[str, Any]]:
            def _make_request():
                try:
                    cfg = _build_cfg(cap)
                except Exception:
                    raise

                try:
                    if stream:
                        chunks: List[str] = []
                        last_response = None
                        for ev in self._client.models.generate_content_stream(
                            model=self.model_name, contents=prompt_text, config=cfg
                        ):
                            last_response = ev
                            t = getattr(ev, "text", None)
                            if t:
                                chunks.append(t)
                        if self.enable_logprobs and last_response is not None:
                            self._last_logprobs = self._extract_logprobs(last_response)
                        return "".join(chunks).strip(), {}
                    resp = self._client.models.generate_content(
                        model=self.model_name, contents=prompt_text, config=cfg
                    )
                except Exception as e:
                    # Fall back if the server rejects the seed parameter.
                    if "seed" in str(e).lower() and self._seed_supported:
                        self._seed_supported = False
                        return _make_request()
                    raise

                if self.enable_logprobs:
                    self._last_logprobs = self._extract_logprobs(resp)
                text, meta = self._extract_text_from_response(resp)
                return text, meta

            return self._with_rate_limit_retry(_make_request)

        # Reset logprobs before each call
        self._last_logprobs = None

        # When logprobs are enabled, streaming yields incomplete logprob data on
        # the server side; use non-streaming only.
        if self.enable_logprobs:
            try:
                text, _ = _call(stream=False, prompt_text=prompt, cap=max_tokens)
                if text:
                    return self._apply_stop(text, stop)
            except Exception:
                pass

            # Short fallback
            tight_prompt = (
                "Reply with ONLY the JSON object. Keep it under 180 tokens. No code fences. End with <END_JSON>.\n\n"
                if wants_json
                else "Be concise. Keep output under 180 tokens.\n\n"
            ) + base_prompt
            small_cap = min(int(max_tokens), 512)
            text2, _ = _call(stream=False, prompt_text=tight_prompt, cap=small_cap)
            if text2:
                return self._apply_stop(text2, stop)

            raise RuntimeError("Gemini API request failed: no text returned with logprobs enabled.")

        # Standard path: try streaming first
        try:
            text, _ = _call(stream=True, prompt_text=prompt, cap=max_tokens)
            if text:
                return self._apply_stop(text, stop)
        except Exception:
            pass

        # Non-streaming
        last_meta: Dict[str, Any] = {}
        try:
            text, meta = _call(stream=False, prompt_text=prompt, cap=max_tokens)
            if text:
                return self._apply_stop(text, stop)
            last_meta = meta
        except Exception as e:
            last_meta = {"error": str(e)}

        # Short fallback
        tight_prompt = (
            "Reply with ONLY the JSON object. Keep it under 180 tokens. No code fences. End with <END_JSON>.\n\n"
            if wants_json
            else "Be concise. Keep output under 180 tokens.\n\n"
        ) + base_prompt
        small_cap = min(int(max_tokens), 512)
        try:
            text, _ = _call(stream=True, prompt_text=tight_prompt, cap=small_cap)
            if text:
                return self._apply_stop(text, stop)
        except Exception:
            pass
        text2, meta2 = _call(stream=False, prompt_text=tight_prompt, cap=small_cap)
        if text2:
            return self._apply_stop(text2, stop)

        raise RuntimeError(
            f"Gemini API request failed: no text returned. meta1={json.dumps(last_meta, default=str)} meta2={json.dumps(meta2, default=str)}"
        )
