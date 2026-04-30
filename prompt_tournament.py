from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Tuple

from promptTemplate import PromptTemplateInterface, normalize_prompt_variant_config, PromptVariantMixin


def _excel_column_label(n: int) -> str:
    """
    Generate Excel-style column labels: A, B, ..., Z, AA, AB, ..., AZ, BA, ...

    Args:
        n: 0-based index

    Returns:
        Label string (e.g., 0 -> 'A', 25 -> 'Z', 26 -> 'AA')
    """
    label = ""
    n += 1  # Convert to 1-based for Excel-style
    while n > 0:
        n -= 1  # Adjust for 0-based modulo
        label = chr(65 + (n % 26)) + label
        n //= 26
    return label


class ComparisonPromptTemplate(PromptTemplateInterface):
    """
    Base class for prompt templates that compare multiple items.
    This version renders items in a single JSON block to prevent number-copy errors.
    Subclasses must still implement format() and get_base_prompt().
    """

    MAX_HISTORY = 5  # Default max history items #TODO: make this a parameter

    def __init__(self, reasoning: bool = True, reasoning_history: bool = False, **kwargs):
        """
        Initialize comparison prompt template.

        Args:
            reasoning: Include reasoning in output
            reasoning_history: Include history (previous decisions) in the prompt
            **kwargs: Additional parameters
        """
        super().__init__(reasoning=reasoning)
        self.reasoning_history = reasoning_history

    @abstractmethod
    def format(self, *args, **kwargs) -> str:
        """Format the main prompt. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_base_prompt(self) -> str:
        """Get the base prompt. Must be implemented by subclasses."""
        pass
        # ---------- NEW: JSON-based comparison formatting ----------
    def format_comparison(
            self,
            items: List[Dict[str, Any]],
            item_formatter: callable = None,  # kept for compatibility; ignored in JSON mode
            option_labels: List[str] = None,
            metric_list: List[str] = None,
        ) -> str:
        """
        Render the comparison as JSON so the model reads numbers reliably.

        Args:
            items: list of schedule/item dicts (each contains metric -> value)
            item_formatter: (ignored) kept for API compatibility
            option_labels: optional labels to use for each item; defaults to A, B, C, ..., Z, AA, AB, ...
            metric_list: optional explicit key order for metrics in each option
        """
        if not items:
            raise ValueError("At least one item is required for comparison.")

        # Labels: A, B, C, ..., Z, AA, AB, AC, ... (Excel-style)
        n = len(items)
        if option_labels is None:
            option_labels = [_excel_column_label(i) for i in range(n)]
        if len(option_labels) != n:
            raise ValueError("option_labels length must match items length.")

        # Prepare JSON payload
        options_json = []
        for label, item in zip(option_labels, items):
            # Only include metrics specified in metric_list
            filtered_item = {k: item[k] for k in metric_list if k in item} if metric_list else dict(item)

            # Wrap each option with a label and its metrics
            options_json.append({
                "label": label,
                "metrics": filtered_item
            })

        # Optional HISTORY block (JSON)
        history_block = ""
        if self.reasoning_history and getattr(self, "history", None):
            hist = []
            for entry in reversed(self.history[-self.MAX_HISTORY:]):
                hist.append({
                    "choice": entry.get("choice"),
                    # keep rationale short if present
                    "reasoning": entry.get("reasoning", None)
                })
            history_block = (
                "=== PREVIOUS DECISIONS (JSON) ===\n"
                "```json\n" + self._to_minified_json({"history": hist}) + "\n```\n\n"
                "=== CURRENT DECISION ===\n\n"
            )

        # Base prompt + JSON ITEMS block
        prompt_parts = []
        prompt_parts.append(self.get_base_prompt())

        if history_block:
            prompt_parts.append(history_block)

        prompt_parts.append("=== ITEMS TO COMPARE (JSON) ===\n")
        prompt_parts.append("Return EXACTLY these numbers when reasoning; do not infer or re-calc.\n")
        prompt_parts.append("```json\n")
        prompt_parts.append(self._to_minified_json({"options": options_json}))
        prompt_parts.append("\n```\n")

        # Clear, strict instructions & FINAL tag menu
        prompt_parts.append(self._format_instructions(n))

        return "".join(prompt_parts)

    # ---------- Helpers ----------
    @staticmethod
    def _to_minified_json(obj: Any) -> str:
        import json
        # stable keys for determinism; no extra spaces
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=False)

    def _format_instructions(self, num_options: int) -> str:
        """Instructions that reference the JSON block above."""
        # Generate Excel-style labels for all options
        valid_choices = [_excel_column_label(i) for i in range(num_options)]
        choices_str = ", ".join(valid_choices)
        if self.reasoning:
            return (
                "\nDecide using ONLY the JSON metrics above. "
                "Briefly explain your reasoning using those exact numbers. "
                f"On the last line, output ONLY the single letter of your choice from: {choices_str}"
            )
        else:
            return (
                "\nDo not explain. Using ONLY the JSON metrics above, "
                f"respond with ONLY the single letter of your choice from: {choices_str}"
            )

    # (Kept for compatibility; no longer used to render items)
    def _default_item_formatter(self, item: Dict[str, Any]) -> str:
        import json
        return json.dumps(item, ensure_ascii=False, indent=2)
        # ---------- Robust FINAL tag parsing ----------
    def _parse_final_tag(self, response: str, valid_choices: list) -> str | None:
        import re
        response = response.strip()

        # 1. Exact match: response is just a single valid choice letter
        if response.upper() in valid_choices:
            return response.upper()

        # 2. Check last non-empty line for a bare choice letter
        lines = response.split('\n')
        for line in reversed(lines):
            s = line.strip()
            if s.upper() in valid_choices:
                return s.upper()
            # Match optional prefix like "Option" or "Choice" followed by the letter
            m = re.match(r'^(?:Option|Choice|Answer)?\s*([A-Z]+)$', s, re.IGNORECASE)
            if m and m.group(1).upper() in valid_choices:
                return m.group(1).upper()

        # 3. Legacy: FINAL tag patterns
        for pattern in [r'FINAL\s*[:=]\s*([A-Z]+)\b', r'FINAL\s+([A-Z]+)\b']:
            matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
            if matches:
                choice = matches[-1].upper()
                if choice in valid_choices:
                    return choice

        # 4. Answer-style patterns
        for pattern in [
            r'(?:answer|choice|select|choose|pick|best|option)\s+(?:is|would be|:)?\s*([A-Z]+)\b',
            r'Option\s+([A-Z]+)\s+is\s+(?:the\s+)?best',
        ]:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                choice = matches[-1].upper()
                if choice in valid_choices:
                    return choice

        # 5. Last resort: search for a valid choice in the last chunk
        last_chunk = response[-100:] if len(response) > 100 else response
        for c in valid_choices:
            patt = rf'(?<![A-Za-z]){re.escape(c)}(?![A-Za-z])'
            if re.search(patt, last_chunk, re.IGNORECASE):
                return c

        return None

    def parse_response(self, response: str, num_options: int = None) -> int | None:
        """
        Parse response to extract choice as 0-based index.

        Args:
            response: The LLM response text
            num_options: Number of options (required to validate choices)

        Returns:
            0-based index of the chosen option, or None if parsing failed
        """
        if not num_options:
            raise ValueError("num_options is required for parse_response")
        valid_choices = [_excel_column_label(i) for i in range(num_options)]
        letter = self._parse_final_tag(response, valid_choices)
        if letter is None:
            return None
        try:
            return valid_choices.index(letter)
        except ValueError:
            return None


class ComparisonPromptAdapter(PromptVariantMixin, ComparisonPromptTemplate):
    """
    A generic prompt adapter for the comparison algorithm.
    It is configured with a fully-formed base prompt, making it adaptable to any scenario.
    """

    def __init__(
        self,
        base_prompt: str | list | dict,
        reasoning: bool = True,
        reasoning_history: bool = False,
        metric_columns: list = None,
        prompt_variant_strategy: str | None = None,
        prompt_variant_seed: int | None = None,
    ):
        super().__init__(reasoning=reasoning, reasoning_history=reasoning_history)
        self._init_variants(base_prompt, strategy_override=prompt_variant_strategy, seed=prompt_variant_seed)

        # Backward-compat aliases used by format() and external callers
        self.base_prompt_variants = self._base_variants
        self.base_prompt_variant_names = self._base_variant_names
        self.prompt_variant_strategy = self._variant_strategy
        self.base_prompt = self._base_variants[self._current_variant_index]
        self.metric_columns = list(metric_columns or [])

    # Keep legacy static method as a thin wrapper for any external callers
    @staticmethod
    def _normalize_base_prompt_config(
        base_prompt: str | list | dict,
    ) -> Tuple[List[str], List[str], str, int]:
        return normalize_prompt_variant_config(base_prompt)

    def _select_base_prompt_for_call(self) -> str:
        """Choose which base prompt variant to use for this format() call."""
        return self._select_variant()

    def get_base_prompt(self) -> str:
        """Returns the configured base prompt."""
        return self._base_variants[self._current_variant_index]

    def get_base_prompt_variant_name(self) -> str:
        """Returns the active prompt variant name."""
        return self.get_variant_name()

    def format(self, items: list) -> str:
        """Formats the complete comparison prompt with a JSON block of items."""
        if not items:
            raise ValueError("At least one item must be provided for comparison.")
        self.base_prompt = self._select_base_prompt_for_call()
        return self.format_comparison(
            items,
            metric_list=self.metric_columns
        )
