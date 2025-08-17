from abc import ABC, abstractmethod  # <-- NEW

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

# local_groq_like.py
import threading, time, itertools, re
from types import SimpleNamespace
from typing import List, Optional, Dict, Iterable, Tuple
from abc import ABC, abstractmethod  # <-- NEW



# =========================
# Abstract Oracle interface
# =========================
class Oracle(ABC):
    """Abstract oracle interface for pairwise A/B decision-making."""

    @abstractmethod
    def call_oracle(
        self,
        prompt: str,
        sched_a :dict, 
        sched_b :dict, 
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        stop: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """"""
