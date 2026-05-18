from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import random
import re


# ── Shared prompt-variant helpers ────────────────────────────────────────────

VALID_VARIANT_STRATEGIES = {"fixed", "round_robin", "random"}


def normalize_prompt_variant_config(
    raw_config: str | list | dict,
) -> Tuple[List[str], List[str], str, int]:
    """Normalize a prompt config (string, list, or dict) into variant lists.

    Returns (variants, names, strategy, default_index).
    """
    if isinstance(raw_config, str):
        template = raw_config.strip()
        if not template:
            raise ValueError("Prompt template cannot be empty.")
        return [template], ["default"], "fixed", 0

    if isinstance(raw_config, list):
        variants = [str(v).strip() for v in raw_config if str(v).strip()]
        if not variants:
            raise ValueError("Prompt template list must contain at least one non-empty template.")
        names = [f"variant_{i + 1}" for i in range(len(variants))]
        return variants, names, "round_robin", 0

    if not isinstance(raw_config, dict):
        raise TypeError("Prompt config must be a string, list, or dict.")

    strategy = str(raw_config.get("strategy") or raw_config.get("mode") or "fixed").strip().lower()
    default_variant = raw_config.get("default_variant")
    variants_cfg = raw_config.get("variants")
    if variants_cfg is None:
        variants_cfg = {
            k: v for k, v in raw_config.items()
            if k not in {"strategy", "mode", "default_variant", "seed"}
        }

    names: List[str] = []
    variants: List[str] = []

    if isinstance(variants_cfg, Mapping):
        for name, template in variants_cfg.items():
            if template is None:
                continue
            template_str = str(template).strip()
            if not template_str:
                continue
            names.append(str(name))
            variants.append(template_str)
    elif isinstance(variants_cfg, Sequence) and not isinstance(variants_cfg, (str, bytes)):
        for idx, entry in enumerate(variants_cfg):
            if isinstance(entry, Mapping):
                template = entry.get("template")
                if template is None:
                    continue
                template_str = str(template).strip()
                if not template_str:
                    continue
                name = str(entry.get("name") or f"variant_{idx + 1}")
                names.append(name)
                variants.append(template_str)
            else:
                template_str = str(entry).strip()
                if not template_str:
                    continue
                names.append(f"variant_{idx + 1}")
                variants.append(template_str)
    else:
        raise TypeError("Prompt config variants must be a mapping or sequence.")

    if not variants:
        raise ValueError("Prompt config must define at least one non-empty variant template.")

    default_index = 0
    if default_variant is not None:
        if isinstance(default_variant, int):
            if 0 <= default_variant < len(variants):
                default_index = default_variant
        else:
            default_name = str(default_variant)
            if default_name in names:
                default_index = names.index(default_name)

    if len(variants) == 1 and strategy == "round_robin":
        strategy = "fixed"

    return variants, names, strategy, default_index


class PromptVariantMixin:
    """Mixin that adds prompt-variant selection to any prompt template class."""

    def _init_variants(
        self,
        raw_config: str | list | dict,
        strategy_override: str | None = None,
        seed: int | None = None,
    ):
        (
            self._base_variants,
            self._base_variant_names,
            inferred_strategy,
            default_index,
        ) = normalize_prompt_variant_config(raw_config)

        strategy = (strategy_override or inferred_strategy or "fixed").strip().lower()
        if strategy not in VALID_VARIANT_STRATEGIES:
            raise ValueError(
                f"Invalid prompt variant strategy '{strategy}'. "
                f"Expected one of {sorted(VALID_VARIANT_STRATEGIES)}."
            )
        self._variant_strategy = strategy
        self._variant_rng = random.Random(seed)
        self._next_variant_index = default_index
        self._current_variant_index = default_index

    def _select_variant(self) -> str:
        """Pick the next variant template according to the strategy."""
        count = len(self._base_variants)
        if count == 1:
            self._current_variant_index = 0
            return self._base_variants[0]

        if self._variant_strategy == "random":
            idx = self._variant_rng.randrange(count)
        elif self._variant_strategy == "round_robin":
            idx = self._next_variant_index
            self._next_variant_index = (self._next_variant_index + 1) % count
        else:  # fixed
            idx = self._current_variant_index
        self._current_variant_index = idx
        return self._base_variants[idx]

    def get_variant_name(self) -> str:
        return self._base_variant_names[self._current_variant_index]


class PromptTemplateInterface(ABC):
    """
    Abstract base class defining the interface for all prompt templates.
    This ensures consistent API across different prompt template implementations.
    """
    
    @abstractmethod
    def __init__(self, reasoning: bool = True, **kwargs):
        """
        Initialize the prompt template.
        
        Args:
            reasoning: Whether to include reasoning in the prompt
            **kwargs: Additional implementation-specific parameters
        """
        self.reasoning = reasoning
        self.history = []
    
    @abstractmethod
    def format(self, *args, **kwargs) -> str:
        """
        Format the main prompt based on input data.
        
        Returns:
            Formatted prompt string ready for LLM consumption
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> str:
        """
        Parse the LLM response to extract the final choice.
        
        Args:
            response: Raw LLM response string
            
        Returns:
            Parsed choice (e.g., 'A', 'B')
        """
        pass
    
    def add_to_history(self, entry: Dict[str, Any], choice: str, reasoning: Optional[str] = None):
        """
        Add a decision to the history for few-shot learning.
        
        Args:
            entry: The data that was compared
            choice: The choice that was made
            reasoning: Optional reasoning for the choice
        """
        history_entry = {
            "data": entry,
            "choice": choice
        }
        if reasoning:
            history_entry["reasoning"] = reasoning
        self.history.append(history_entry)
    
    def clear_history(self):
        """Clear all history entries."""
        self.history = []
    
    def get_history_length(self) -> int:
        """Get the number of entries in history."""
        return len(self.history)
    
    def _parse_final_tag(self, response: str, valid_choices: List[str]) -> str:
        """
        Common parsing logic for FINAL: X pattern.
        
        Args:
            response: LLM response string
            valid_choices: List of valid choices (e.g., ['A', 'B'])
            
        Returns:
            Parsed choice
        """
        # Look for FINAL: X pattern
        pattern = r'FINAL:\s*([' + ''.join(valid_choices) + '])'
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # Fallback: look for isolated choice at the end
        lines = response.strip().split('\n')
        if lines:
            last_line = lines[-1].strip().upper()
            if last_line in valid_choices:
                return last_line
        
        raise ValueError(f"Could not parse response: {response}")
    
    @abstractmethod
    def get_base_prompt(self) -> str:
        """
        Get the base/system prompt for this template.
        
        Returns:
            Base prompt string
        """
        pass