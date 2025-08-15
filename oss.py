
# local_groq_like.py
from __future__ import annotations
import threading, time, itertools, re
from types import SimpleNamespace
from typing import List, Optional, Dict, Iterable, Tuple
from oracle import Oracle
from abc import ABC, abstractmethod  # <-- NEW
from oracle import Oracle
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

class LocalGroqLikeClient(Oracle):
    """
    Minimal, Groq-shaped client for local HF models.
    Usage matches Groq enough for your code:
       client = LocalGroqLikeClient(model_id="unsloth/gpt-oss-20b-BF16", ...)
       out = client.chat.completions.create(model=..., messages=[...], stream=False)
    """

    def __init__(
        self,
        model_id: str = "unsloth/gpt-oss-20b-BF16",
        torch_dtype: Optional[torch.dtype] = None,   # default: bf16 on CUDA else f32
        device_map: str = "auto",
        max_memory: Optional[Dict] = None,           # e.g., {0:"22GiB","cpu":"50GiB"} for 3090
        low_cpu_mem_usage: bool = True,
        trust_remote_code: bool = True,
        pad_to_eos: bool = True,
    ):
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # sensible default for a 24GB 3090 with ~64GB RAM
        if max_memory is None and torch.cuda.is_available():
            max_memory = {0: "22GiB", "cpu": "50GiB"}

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=trust_remote_code,
        ).eval()

        # emulate a tiny sub-API like groq.chat.completions.create(...)
        self.chat = SimpleNamespace(completions=_Completions(self))
        self.pad_to_eos = pad_to_eos
        # convenience defaults (can be overridden per-call)
        self.default_temperature = 0.0
        self.default_top_p = 1.0
        self.default_max_new_tokens = 256

    def call_oracle(
        self,
        prompt: str,
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
        stop: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """
        Call the model with a prompt and parse the response for A/B decision.
        Returns (decision, reasoning). Decision is 'A' or 'B'.
        """
        # Use defaults if not specified
        if temperature is None:
            temperature = self.default_temperature
        if top_p is None:
            top_p = self.default_top_p
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens

        # Generate response
        response = self._generate_text(
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            stop=stop
        )

        # Parse the response for decision and reasoning
        return self._parse_pairwise(response)

    def _parse_pairwise(self, response: str) -> Tuple[str, str]:
        """
        Parse a response to extract A/B decision and reasoning.
        Returns (choice, response) where choice is 'A' or 'B'.
        """
        match = re.search(r"\{([AB])\}\s*$", response.strip())
        if not match:
            choice = (
                response.strip()[0]
                if response.strip() and response.strip()[0] in ("A", "B")
                else "A"
            )
            return choice, response.strip()
        return match.group(1), response

    # ---- internal helpers ----
    def _render_chat(self, messages: List[Dict]) -> str:
        # Same template you used in test.py
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.inference_mode()
    def _generate_text(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> str:
        do_sample = temperature and temperature > 0.0
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=bool(do_sample),
            temperature=float(temperature) if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=self.tok.eos_token_id if self.pad_to_eos else None,
            eos_token_id=self.tok.eos_token_id,
        )
        text = self.tok.decode(out_ids[0], skip_special_tokens=False)

        # crude stop-string handling
        if stop:
            cut = min((text.find(s) for s in stop if s in text), default=-1)
            if cut != -1:
                text = text[:cut]
        return text

    @torch.inference_mode()
    def _stream_text(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> Iterable[str]:
        do_sample = temperature and temperature > 0.0
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=False)

        kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=bool(do_sample),
            temperature=float(temperature) if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=self.tok.eos_token_id if self.pad_to_eos else None,
            eos_token_id=self.tok.eos_token_id,
            streamer=streamer,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=kwargs, daemon=True)
        thread.start()

        so_far = ""
        for piece in streamer:
            so_far += piece
            yield piece
            # optional stop: if a stop string appears, end early
            if stop and any(s in so_far for s in stop):
                break

class _Completions:
    """Mimics groq.chat.completions.create(...) minimally."""

    def __init__(self, parent: LocalGroqLikeClient):
        self.parent = parent

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_completion_tokens: int = 256,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs,
    ):
        # Render chat -> prompt
        prompt = self.parent._render_chat(messages)
        # Defaults if caller omitted
        if temperature is None:
            temperature = self.parent.default_temperature
        if top_p is None:
            top_p = self.parent.default_top_p
        if max_completion_tokens is None:
            max_completion_tokens = self.parent.default_max_new_tokens

        if stream:
            # Yield Groq-shaped streamed deltas
            for piece in self.parent._stream_text(
                prompt, temperature, top_p, max_completion_tokens, stop=stop
            ):
                yield SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))]
                )
        else:
            text = self.parent._generate_text(
                prompt, temperature, top_p, max_completion_tokens, stop=stop
            )
            # Non-streaming Groq-shaped object
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
            )
