# remoteOss.py
"""
Auto-discover and talk to your DeepSeek vLLM API (OpenAI-compatible).
Discovery order:
  1) env DEEPSEEK_API_URL (e.g., http://badfellow:8002 or http://badfellow:8002/v1)
  2) ~/.deepseek_url            (single line URL)
  3) ~/.deepseek_node + ~/.deepseek_port  -> http://<node>:<port>
  4) Slurm: find running job named $DEEPSEEK_JOB_NAME (default deepseek_r1d_8b_api)
     and build http://<node>:$DEEPSEEK_API_PORT (default 8002); cache to ~/.deepseek_url
  5) Back-compat GPT20B_* files/env
  6) Fallback http://127.0.0.1:8002

Parallel clients are fine—vLLM batches across requests.
"""

from __future__ import annotations
from typing import Optional, Dict, Tuple, Any, List
import os, time, random, json, subprocess, shlex, socket
from pathlib import Path

import httpx
import re 
DEFAULT_MODEL_ID = os.getenv(
    "DEEPSEEK_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
)
DEFAULT_JOB_NAME = os.getenv("DEEPSEEK_JOB_NAME", "deepseek_r1d_8b_api")
DEFAULT_PORT = int(os.getenv("DEEPSEEK_API_PORT", "8002"))

_CLIENT = None


def _normalize_base(url: str) -> str:
    url = url.strip().rstrip("/")
    return url if url.endswith("/v1") else (url + "/v1")


def _probe_ok(base_url_v1: str, timeout_s: float = 3.0) -> bool:
    try:
        r = httpx.get(f"{base_url_v1}/models", timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def _write_cache(url: str):
    try:
        (Path.home() / ".deepseek_url").write_text(url.strip() + "\n")
    except Exception:
        pass


def _read_file(p: Path) -> Optional[str]:
    try:
        if p.exists():
            t = p.read_text().strip()
            return t if t else None
    except Exception:
        pass
    return None


def _slurm_find_server(job_name: str, port: int) -> Optional[str]:
    """Return http://<node>:<port> for the RUNNING server job, if found."""
    try:
        # Columns: JOBID NAME STATE NODELIST(REASON) TIME
        cmd = 'squeue -u $USER -h -o "%i|%j|%T|%R|%M"'
        out = subprocess.check_output(cmd, shell=True, text=True)
        best = None  # (elapsed_seconds, node)
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) < 5:
                continue
            _, name, state, node, elapsed = parts
            if name != job_name or state != "R":
                continue
            # elapsed format H:MM:SS or D-HH:MM:SS
            secs = 0
            try:
                if "-" in elapsed:
                    days, rest = elapsed.split("-", 1)
                    h, m, s = map(int, rest.split(":"))
                    secs = int(days) * 86400 + h * 3600 + m * 60 + s
                else:
                    h, m, s = map(int, elapsed.split(":"))
                    secs = h * 3600 + m * 60 + s
            except Exception:
                pass
            best = max(best or (0, ""), (secs, node))
        if best and best[1] and best[1] != "(none)":
            node = best[1]
            url = f"http://{node}:{port}"
            return url
    except Exception:
        pass
    return None


def _discover_base_url() -> str:
    # 1) env
    url = os.environ.get("DEEPSEEK_API_URL")
    if url and _probe_ok(_normalize_base(url)):
        _write_cache(url)
        return url.rstrip("/")

    # 2) ~/.deepseek_url
    f = _read_file(Path.home() / ".deepseek_url")
    if f and _probe_ok(_normalize_base(f)):
        return f.rstrip("/")

    # 3) ~/.deepseek_node + ~/.deepseek_port
    node = _read_file(Path.home() / ".deepseek_node")
    port = _read_file(Path.home() / ".deepseek_port")
    if node and port:
        guess = f"http://{node}:{port}"
        if _probe_ok(_normalize_base(guess)):
            _write_cache(guess)
            return guess.rstrip("/")

    # 4) Slurm probe by job name
    guess = _slurm_find_server(DEFAULT_JOB_NAME, DEFAULT_PORT)
    if guess and _probe_ok(_normalize_base(guess)):
        _write_cache(guess)
        # Also backfill node/port helper files
        try:
            (Path.home() / ".deepseek_node").write_text(guess.split("//", 1)[1].split(":")[0] + "\n")
            (Path.home() / ".deepseek_port").write_text(str(DEFAULT_PORT) + "\n")
        except Exception:
            pass
        return guess.rstrip("/")

    # 5) Back-compat GPT-OSS discovery
    url = os.environ.get("GPT20B_API_URL")
    if url and _probe_ok(_normalize_base(url)):
        _write_cache(url)
        return url.rstrip("/")
    g = _read_file(Path.home() / ".gpt20b_url")
    if g and _probe_ok(_normalize_base(g)):
        return g.rstrip("/")
    node = _read_file(Path.home() / ".gpt20b_node")
    port = _read_file(Path.home() / ".gpt20b_port")
    if node and port:
        guess = f"http://{node}:{port}"
        if _probe_ok(_normalize_base(guess)):
            _write_cache(guess)
            return guess.rstrip("/")

    # 6) Local fallback
    return "http://127.0.0.1:8002"


class _OptimizedRemoteClient:
    def __init__(
        self,
        base_url: str,
        *,
        model_id: str = DEFAULT_MODEL_ID,
        timeout_s: float = 120.0,
        max_retries: int = 20,
        max_connections: int = 256,
        max_keepalive: int = 64,
        default_seed: Optional[int] = 12345,
    ):
        self.model_id = model_id
        self.base_v1 = _normalize_base(base_url)
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.default_seed = default_seed
        self.default_temperature = 0.0
        self.default_top_p = 1.0
        self.default_max_new_tokens = 1000

        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=30.0,
        )
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_s, connect=20.0, read=timeout_s),
            limits=limits,
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'EMPTY')}"},
        )

    def _sleep_backoff(self, attempt: int):
        time.sleep(min(2 ** (attempt - 1), 8) + random.random() * 0.25)

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
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = int(seed)

        url = f"{self.base_v1}/chat/completions"
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._client.post(url, json=payload)
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    self._sleep_backoff(attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                self._sleep_backoff(attempt)
        raise RuntimeError(f"LLM API request failed after {self.max_retries} attempts: {last_err}")

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
    _FINAL_TAG_RE = re.compile(
        r'^\s*(?:```[a-zA-Z]*\s*)?FINAL\s*[:=\-]?\s*([AB])\b',
        re.IGNORECASE | re.MULTILINE
    )

    def _parse_final_choice(self, text: str) -> Optional[str]:
        """
        Parse the model's output and return 'A' or 'B' if found, else None.
        Priority:
          1) Explicit "FINAL: A/B" tag (last occurrence wins)
          2) JSON-like {"final": "A"} or "final: B"
          3) Legacy tokens {A}/{B}
          4) Lone 'A' or 'B' on its own line
        """
        # 1) Explicit FINAL: X (prefer the last match)
        matches = self._FINAL_TAG_RE.findall(text)
        if matches:
            return matches[-1].upper()

        # 2) JSON-like or key-value "final"
        kv = re.findall(r'"?final"?\s*[:=]\s*"?([AB])"?', text, flags=re.IGNORECASE)
        if kv:
            return kv[-1].upper()

        # 3) Legacy brace tokens
        last_a = text.rfind("{A}")
        last_b = text.rfind("{B}")
        if last_a != -1 or last_b != -1:
            if last_a > last_b:
                return "A"
            elif last_b > last_a:
                return "B"
            # if only one exists, return that
            if last_a != -1:
                return "A"
            if last_b != -1:
                return "B"

        # 4) Lone A/B on its own line (take the last occurrence)
        lone = re.findall(r'^\s*([AB])\s*$', text, flags=re.IGNORECASE | re.MULTILINE)
        if lone:
            return lone[-1].upper()

        return None

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
        """
        Returns (choice, raw_text) where choice is 'A' or 'B' (best-effort parse).
        Now correctly handles outputs like:
            FINAL: A
            FINAL: B
        plus legacy formats.
        """
        temperature = getattr(self, "default_temperature", 0.2) if temperature is None else temperature
        top_p = getattr(self, "default_top_p", 0.95) if top_p is None else top_p
        max_new_tokens = getattr(self, "default_max_new_tokens", 256) if max_new_tokens is None else max_new_tokens
        seed = getattr(self, "default_seed", None) if seed is None else seed

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

        choice = self._parse_final_choice(text)
        if choice is None:
            # Absolute last resort: keep your previous heuristic
            # (first char if it's A/B; otherwise default to 'A')
            if text and text[0] in ("A", "B"):
                choice = text[0]
            else:
                choice = "A"

        return choice, text
    # Add this method to the _OptimizedRemoteClient class in remoteOss.py

    def generate_response(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generate a response for a given prompt.
        This method is compatible with ScheduleBatchExp's expectations.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_new_tokens: Maximum tokens to generate
            stop: Stop sequences
            seed: Random seed for reproducibility
        
        Returns:
            The raw text response from the model
        """
        temperature = self.default_temperature if temperature is None else temperature
        top_p = self.default_top_p if top_p is None else top_p
        max_new_tokens = self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
        seed = self.default_seed if seed is None else seed
        
        messages = [{"role": "user", "content": prompt}]
        
        text = self._post_chat(
            messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            seed=seed,
            stop=stop,
        )
        
        # Apply stop sequences if provided
        text = self._apply_stop(text.strip(), stop)
        
        return text

def get_local_client(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    force_full_precision: bool = None,  # kept for compatibility, unused
):
    """Backwards-compatible factory."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT

    base = _discover_base_url()
    # Force DeepSeek unless you explicitly set DEEPSEEK_RESPECT_MODEL=1
    if os.getenv("DEEPSEEK_RESPECT_MODEL", "0") != "1":
        model_id = DEFAULT_MODEL_ID

    print(f"[remoteOss] Using LLM API at: {base} (model={model_id})")
    _CLIENT = _OptimizedRemoteClient(base, model_id=model_id)
    return _CLIENT
