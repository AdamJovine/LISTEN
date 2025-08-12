from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

import time
import re
from typing import Dict, Tuple, List, Optional
import pandas as pd

# pip install groq
from groq import Groq


class FreeLLMPreferenceClient:
    """
    Client to get schedule preferences from an LLM via Groq Chat Completions.
    Supports:
      (a) pairwise A vs B
      (b) batch choose-one (up to 100)
      (c) top-K selection from a candidate list (final call)
    """

    def __init__(
        self,
        provider: str = "groq",
        api_key: Optional[str] = None,  # defaults to env GROQ_API_KEY
        model_name: Optional[str] = None,  # default below
        rate_limit_delay: float = 1.0,
        max_tokens: int = 512,
        temperature: float = 0.7,
        max_retries: int = 10,
        simple: bool = True,
        top_p: float = 1.0,
        reasoning_effort: Optional[str] = "medium",
        stream: bool = False,  # you can enable streaming per-call if you want
    ):
        if provider.lower() != "groq":
            raise RuntimeError("This implementation only supports provider='groq'.")

        self.provider = "groq"
        self.api_key = api_key  # or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set. Export it or pass api_key=...")

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0.0
        self.simple = simple
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.stream_default = stream

        # Default to the model you specified
        self.model_name = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

        # Groq client
        self._groq = Groq(api_key=self.api_key)

        # ----- Base prompt (includes two-in-three priority) -----
        base = (
            "You are an experienced University Registrar. Your absolute top priority is ensuring no student has a simultaneous conflict. "
            "After that, your next most critical goal is to minimize the number of students facing three exams in a 24-hour period, "
            "as this causes the most stress. Finally, use the number of back-to-back exams as a tie-breaker to choose between "
            "otherwise equal schedules. Your goal is to find the schedule that best reflects these priorities. "
            "Definitions: conflicts is the number of students with exams scheduled at the same time; "
            "quints is 5 exams in a row; quads is 4 in a row; triples is 3 in a row; "
            "b2b is back-to-back exams; two in three is 2 exams within 24 hours; "
            "three in 24 is 3 exams within 24 hours. "
            "University policy: conflicts and 3 exams in 24 hours require one of the exams to be rescheduled. "
            "Strongly minimize in this priority order: conflicts, three in 24 hours, then minimize b2b."
        )

        # Pairwise templates
        self.prompt_pairwise_verbose = base + (
            " When comparing Schedule A and Schedule B, provide a few sentence analysis that highlights key trade-offs "
            "between their metrics. Conclude with your final choice formatted exactly in curly braces, e.g. {A} or {B}. "
            "Do not output just 'A' or 'B'; include the reasoning and marker. Make sure to end with your choice either {A} or {B}."
        )
        self.prompt_pairwise_simple = base + (
            " Conclude with your final choice formatted exactly in curly braces, e.g. {A} or {B}. Only respond with your decision."
        )

        # Batch choose-one templates
        self.prompt_batch_verbose_header = base + (
            " You will be given up to 100 schedules, each labeled by an ID. "
            "Evaluate them and pick the single best schedule overall.\n"
            "Briefly explain your top choice and key trade-offs among the closest alternatives. "
            "End your response with only the winning ID in curly braces on the last line, e.g. {S42}."
        )
        self.prompt_batch_simple_header = base + (
            " You will be given up to 100 schedules, each labeled by an ID. Pick the single best schedule.\n"
            "Return only the winning ID in curly braces, e.g. {S42}."
        )

        # Final top-K templates (compare batch winners)
        self.prompt_topk_verbose_header = base + (
            " You will be given several candidate schedules (winners from earlier batches), each labeled by an ID. "
            "Pick the best K schedules. "
            "End your response with exactly K IDs in curly braces, comma-separated, e.g. {S3,S7,S9,S12,S20}."
        )
        self.prompt_topk_simple_header = base + (
            " You will be given candidate schedules (winners from earlier batches), each labeled by an ID. "
            "Pick the best K schedules. "
            "Return only K IDs in curly braces, comma-separated, e.g. {S3,S7,S9,S12,S20}."
        )

    # ------------------ Core plumbing ------------------
    def _rate_limit(self):
        now = time.time()
        wait = self.rate_limit_delay - (now - self._last_request_time)
        if wait > 0:
            time.sleep(wait)
        self._last_request_time = time.time()

    def _groq_chat(self, prompt: str, stream: Optional[bool] = None) -> str:
        """Send a single-message prompt to Groq Chat Completions and return the text."""
        use_stream = self.stream_default if stream is None else stream
        last_error = None

        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                completion = self._groq.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    top_p=self.top_p,
                    reasoning_effort=self.reasoning_effort,
                    stream=use_stream,
                    stop=None,
                )

                if use_stream:
                    # Assemble streamed chunks into a single string
                    chunks = []
                    for chunk in completion:
                        piece = chunk.choices[0].delta.content or ""
                        if piece:
                            chunks.append(piece)
                    return "".join(chunks).strip()
                else:
                    # Non-streaming
                    return (completion.choices[0].message.content or "").strip()

            except Exception as e:
                last_error = e
                # Exponential backoff up to 60s
                time.sleep(min(60, 2**attempt))

        raise RuntimeError(
            f"Groq API failed after {self.max_retries} attempts: {last_error}"
        )

    def _call_api(self, prompt: str, stream: Optional[bool] = None) -> str:
        return self._groq_chat(prompt, stream=stream)

    # ------------------ Pairwise API ------------------
    def _format_prompt_pairwise(self, sched_a: Dict, sched_b: Dict) -> str:
        def fmt(name, d):
            return f"{name}: " + ", ".join(f"{k}={v}" for k, v in d.items())

        header = (
            self.prompt_pairwise_simple if self.simple else self.prompt_pairwise_verbose
        )
        return "\n".join(
            [fmt("Schedule A", sched_a), fmt("Schedule B", sched_b), "", header]
        )

    def _parse_pairwise(self, response: str) -> Tuple[str, str]:
        match = re.search(r"\{([AB])\}\s*$", response.strip())
        if not match:
            choice = (
                response.strip()[0]
                if response.strip() and response.strip()[0] in ("A", "B")
                else "A"
            )
            return choice, response.strip()
        return match.group(1), response

    def get_preference(
        self, sched_a: Dict, sched_b: Dict, stream: Optional[bool] = None
    ) -> Tuple[str, str]:
        prompt = self._format_prompt_pairwise(sched_a, sched_b)
        raw = self._call_api(prompt, stream=stream)
        print(f"[LLM raw response] {raw!r}")
        return self._parse_pairwise(raw)

    # ------------------ Batch choose-one (up to 100) ------------------
    def _format_prompt_batch(self, ids: List[str], dicts: List[Dict]) -> str:
        assert len(ids) == len(dicts) and 1 <= len(ids) <= 100
        header = (
            self.prompt_batch_simple_header
            if self.simple
            else self.prompt_batch_verbose_header
        )

        priority_keys = [
            "conflicts",
            "three_in_24",
            "two_in_three",
            "b2b",
            "triples",
            "quads",
            "quints",
        ]
        lines = []
        for sid, d in zip(ids, dicts):
            other_keys = [k for k in d.keys() if k not in priority_keys]
            ordered_keys = [k for k in priority_keys if k in d] + sorted(other_keys)
            metrics_str = ", ".join(f"{k}={d[k]}" for k in ordered_keys)
            lines.append(f"{sid}: {metrics_str}")
        return header + "\n\n" + "\n".join(lines) + "\n"

    def _parse_batch_choice(self, response: str) -> Tuple[str, str]:
        m = re.search(r"\{([A-Za-z0-9_\-]+)\}\s*$", response.strip())
        if m:
            return m.group(1), response
        m2 = re.search(r"\{([A-Za-z0-9_\-]+)\}", response)
        if m2:
            return m2.group(1), response
        m3 = re.search(r"\b([A-Za-z]\w{0,10}\d{0,5})\b", response)
        if m3:
            return m3.group(1), response
        raise ValueError(f"Could not parse winning ID from response: {response!r}")

    def choose_best_in_batch(
        self, ids: List[str], dicts: List[Dict], stream: Optional[bool] = None
    ) -> Tuple[str, str]:
        prompt = self._format_prompt_batch(ids, dicts)
        raw = self._call_api(prompt, stream=stream)
        print(f"[LLM raw response] {raw!r}")
        return self._parse_batch_choice(raw)

    # ------------------ Final top-K (compare favorites from all batches) ------------------
    def _format_prompt_topk(self, ids: List[str], dicts: List[Dict], k: int) -> str:
        assert 1 <= k <= len(ids)
        header = (
            self.prompt_topk_simple_header
            if self.simple
            else self.prompt_topk_verbose_header
        )
        header = header.replace("K", str(k))
        priority_keys = [
            "conflicts",
            "three_in_24",
            "two_in_three",
            "b2b",
            "triples",
            "quads",
            "quints",
        ]
        lines = []
        for sid, d in zip(ids, dicts):
            other_keys = [kk for kk in d.keys() if kk not in priority_keys]
            ordered_keys = [kk for kk in priority_keys if kk in d] + sorted(other_keys)
            metrics_str = ", ".join(f"{kk}={d[kk]}" for kk in ordered_keys)
            lines.append(f"{sid}: {metrics_str}")
        return header + "\n\n" + "\n".join(lines) + "\n"

    def _parse_topk_choice(
        self, response: str, candidate_ids: List[str], k: int
    ) -> Tuple[List[str], str]:
        # Find last {...}
        m = re.findall(r"\{([^}]*)\}", response)
        chosen = []
        if m:
            inside = m[-1]
            parts = re.split(r"[,\s]+", inside.strip())
            candset = set(candidate_ids)
            for p in parts:
                p = p.strip()
                if p and p in candset and p not in chosen:
                    chosen.append(p)
                    if len(chosen) == k:
                        break
        if len(chosen) < k:
            raise ValueError(
                f"Expected {k} IDs but parsed {len(chosen)} from response: {response!r}"
            )
        return chosen, response

    def choose_top_k(
        self, ids: List[str], dicts: List[Dict], k: int, stream: Optional[bool] = None
    ) -> Tuple[List[str], str]:
        prompt = self._format_prompt_topk(ids, dicts, k)
        raw = self._call_api(prompt, stream=stream)
        print(f"[LLM raw response] {raw!r}")
        return self._parse_topk_choice(raw, ids, k)
