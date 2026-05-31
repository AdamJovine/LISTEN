from __future__ import annotations
from typing import Dict, Any, Optional, Union, List
import json
import re

from prompt_template import PromptTemplateInterface, PromptVariantMixin


_PROMPT_VAR_DEFAULTS: dict = {
    "comparison_noun": "option",
    "utility_priorities": "your stated priorities",
    "utility_task_body": (
        "Use positive weights for metrics where higher is better and negative weights where lower is better.\n"
        "\n"
        "Each metric is first mapped to a normalized [0,1] preference score."
    ),
    "utility_domain": "this problem",
    "best_item_found": "OPTION FOUND",
    "best_item_ref": "best option",
    "analysis_questions": (
        "- Does this option seem reasonable given the stated preferences?\n"
        "- Are any metrics over- or under-weighted?\n"
        "- Should any penalties or bonuses be adjusted?"
    ),
    "utility_weight_rule": (
        "- Weights may be positive or negative. Use positive weights where higher is better"
        " and negative weights where lower is better."
    ),
    "utility_key_rule": "- Ensure every weight key exactly matches the metric names listed above.",
    "utility_key_rule_refinement": "- Ensure every weight key exactly matches the metric names in the initial prompt.",
}


def _apply_prompt_vars(template: str, prompt_vars: dict) -> str:
    """Replace scenario-specific placeholders in a prompt template.

    Keys are drawn from prompt_vars, falling back to _PROMPT_VAR_DEFAULTS.
    If a value resolves to an empty string the entire line holding the
    placeholder is removed to avoid stray blank lines.
    """
    merged = {**_PROMPT_VAR_DEFAULTS, **prompt_vars}
    for key, value in merged.items():
        placeholder = f"{{{key}}}"
        if placeholder not in template:
            continue
        value_str = str(value).rstrip("\n") if value is not None else ""
        if value_str:
            template = template.replace(placeholder, value_str)
        else:
            # Remove the whole line that contains an empty placeholder
            template = (
                template
                .replace(f"\n{placeholder}", "")
                .replace(f"{placeholder}\n", "")
                .replace(placeholder, "")
            )
    return template


def _build_weights_json(metric_columns: List[str], non_numeric_metrics: List[str] = None) -> str:
    """
    Build the JSON weight template block from metric_columns.
    Non-numeric metrics are excluded from the return format.
    Returns an unescaped JSON string (for utility_base).
    """
    non_numeric = set(non_numeric_metrics or [])
    numeric_cols = [col for col in metric_columns if col not in non_numeric]
    lines = ['{']
    lines.append('  "weights": {')
    for i, col in enumerate(numeric_cols):
        comma = "," if i < len(numeric_cols) - 1 else ""
        lines.append(f'  "{col}": 0.0{comma}')
    lines.append('  },')
    lines.append('  "formula": "utility = sum(weight_i * score_i) for all metrics (score_i \u2208 [0,1])",')
    lines.append('  "description": "Brief explanation of your weighting rationale"')
    lines.append('}')
    return '\n'.join(lines)


def _build_weights_json_escaped(metric_columns: List[str], non_numeric_metrics: List[str] = None) -> str:
    """
    Build the JSON weight template block for refinement prompts.
    Non-numeric metrics are excluded; numeric metrics get <adjusted_weight>.
    """
    non_numeric = set(non_numeric_metrics or [])
    numeric_cols = [col for col in metric_columns if col not in non_numeric]
    lines = ['{']
    lines.append('  "weights": {')
    for i, col in enumerate(numeric_cols):
        comma = "," if i < len(numeric_cols) - 1 else ""
        value = "<adjusted_weight>"
        lines.append(f'    "{col}": {value}{comma}')
    lines.append('  },')
    lines.append('  "formula": "utility = sum(weight_i * score_i) for all metrics (score_i \u2208 [0,1])",')
    lines.append('  "description": "Brief explanation of your adjustments and reasoning based on the best schedule shown above"')
    lines.append('}')
    return '\n'.join(lines)


class UtilityPromptTemplate(PromptVariantMixin, PromptTemplateInterface):
    """
    Prompt template for utility-based algorithms (LISTEN-U).
    Auto-generates JSON weight templates from metric_columns.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        reasoning: bool = True,
        prompt_variant_strategy: str | None = None,
        prompt_variant_seed: int | None = None,
        **kwargs
    ):
        super().__init__(reasoning=reasoning)

        self.prompts_config = config.get("prompts", {})
        if not self.prompts_config:
            raise ValueError("Config must contain a 'prompts' section.")

        self.scenario_header = self.prompts_config.get("scenario_header")
        if not self.scenario_header:
            raise ValueError("Prompts config must define a 'scenario_header'.")

        self.persona_context = self.prompts_config.get("persona_context", "")
        self.attribute_definitions = self.prompts_config.get("attribute_definitions", "")
        self.section_order = kwargs.get("section_order")

        # Get generic templates — supports both string (legacy) and dict (variants)
        utility_base_raw = self.prompts_config.get("utility_base")
        if not utility_base_raw:
            raise ValueError("Prompts config must define 'utility_base'.")
        self._init_variants(utility_base_raw, strategy_override=prompt_variant_strategy, seed=prompt_variant_seed)
        self.utility_base_template = self._base_variants[self._current_variant_index]

        self.utility_refinement_template = self.prompts_config.get("utility_refinement")
        if not self.utility_refinement_template:
            raise ValueError("Prompts config must define 'utility_refinement'.")

        self.prompt_vars = config.get("prompt_vars", {})

        # Build JSON weight blocks from metric_columns
        metric_columns = config.get("metric_columns", [])
        non_numeric_metrics = config.get("non_numeric_metrics", [])
        self.metric_columns = metric_columns
        self.non_numeric_metrics = set(non_numeric_metrics)
        self.metric_weights_json = _build_weights_json(metric_columns, non_numeric_metrics)
        self.metric_weights_json_escaped = _build_weights_json_escaped(metric_columns, non_numeric_metrics)

        # Track iteration state
        self.iteration = 0
        self.last_weights: Optional[Dict[str, float]] = None
        self.last_formula: Optional[str] = None
        self.last_description: Optional[str] = None
        self.last_best_solution: Optional[Any] = None
        self.policy_guidance: str = kwargs.get("policy_guidance", "")

    def format(
        self,
        iteration: int = 0,
        best_solution: Optional[Union[Dict[str, Any], Any]] = None,
        weights: Optional[Union[Dict[str, float], Any]] = None,
        formula: Optional[str] = None,
        description: Optional[str] = None,
        policy_guidance: Optional[str] = None
    ) -> str:
        self.iteration = iteration

        if iteration == 0:
            selected_template = self._select_variant()
            self.utility_base_template = selected_template
            prompt = selected_template.replace(
                "{scenario_header}",
                self._resolve_scenario_block()
            )
            prompt = _apply_prompt_vars(prompt, self.prompt_vars)
            prompt = prompt.replace("{metric_weights_json}", self.metric_weights_json)
            return prompt
        else:
            if best_solution is None or weights is None:
                raise ValueError(
                    "Refinement prompts require best_solution and weights from previous iteration"
                )

            weights_dict = weights
            if hasattr(weights, 'to_dict'):
                weights_dict = weights.to_dict()
            weights_numeric = {k: v for k, v in weights_dict.items() if k not in self.non_numeric_metrics}

            # Build display dict with all metric columns: non-numeric shown as 0.0
            weights_display = {}
            for col in self.metric_columns:
                if col in self.non_numeric_metrics:
                    weights_display[col] = 0.0
                else:
                    weights_display[col] = weights_numeric.get(col, 0.0)

            self.last_weights = weights_numeric
            self.last_formula = formula or "utility = sum(weight_i * score_i) for all metrics"
            self.last_description = description or "No description provided"
            self.last_best_solution = best_solution

            prompt = _apply_prompt_vars(self.utility_refinement_template, self.prompt_vars)
            prompt = prompt.replace("{iteration}", str(iteration))
            prompt = prompt.replace("{weights}", json.dumps(weights_display, indent=2))
            prompt = prompt.replace("{formula}", self.last_formula)
            prompt = prompt.replace("{description}", self.last_description)
            prompt = prompt.replace(
                "{best_solution}",
                self._format_solution(best_solution)
            )
            prompt = prompt.replace(
                "{policy_guidance}",
                policy_guidance or self.policy_guidance or ""
            )
            prompt = prompt.replace("{metric_weights_json_escaped}", self.metric_weights_json_escaped)

            return prompt

    def _format_solution(self, solution: Union[Dict[str, Any], Any]) -> str:
        if hasattr(solution, 'to_dict'):
            solution = solution.to_dict()
        if isinstance(solution, dict):
            if self.metric_columns:
                solution = {k: solution[k] for k in self.metric_columns if k in solution}
            return json.dumps(solution, indent=2)
        else:
            return str(solution)

    def parse_response(self, response: str) -> Dict[str, Any]:
        response = response.strip()
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.replace('<END_JSON>', '')
        response = response.strip()

        try:
            data = json.loads(response)
            if "weights" not in data:
                raise ValueError("Response must contain 'weights' key")
            self.last_weights = data.get("weights")
            self.last_formula = data.get("formula")
            self.last_description = data.get("description")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON response: {e}\nResponse: {response}")

    def get_base_prompt(self) -> str:
        template = self._base_variants[self._current_variant_index]
        prompt = template.replace("{scenario_header}", self._resolve_scenario_block())
        prompt = _apply_prompt_vars(prompt, self.prompt_vars)
        prompt = prompt.replace("{metric_weights_json}", self.metric_weights_json)
        return prompt

    def _resolve_scenario_block(self) -> str:
        # iter-0 scenario block is persona + attributes only; the priorities
        # text (policy_guidance) is injected by utility_refinement at iter >=1,
        # matching pre-section_order behavior. Persona and attributes are
        # joined with a single newline so the blank line sits after the block
        # (from the template's "{scenario_header}\n\n..."), matching the
        # original scenario_header layout byte-for-byte.
        if not self.section_order:
            return self.scenario_header
        sections = {
            "persona": self.persona_context,
            "attributes": self.attribute_definitions,
        }
        parts = [
            (sections.get(name) or "").rstrip("\n")
            for name in self.section_order
            if (sections.get(name) or "").strip()
        ]
        return "\n".join(parts) + "\n"

    def get_base_prompt_variant_name(self) -> str:
        return self.get_variant_name()

    def get_current_iteration(self) -> int:
        return self.iteration

    def get_last_weights(self) -> Optional[Dict[str, float]]:
        return self.last_weights

    def set_policy_guidance(self, guidance: str):
        self.policy_guidance = guidance


class UtilityPromptAdapter:
    """
    Adapter to maintain backward compatibility with code expecting PromptManager interface.
    """

    def __init__(self, config: Dict[str, Any], prompt_variant_strategy=None, prompt_variant_seed=None, **kwargs):
        self.template = UtilityPromptTemplate(
            config,
            prompt_variant_strategy=prompt_variant_strategy,
            prompt_variant_seed=prompt_variant_seed,
            **kwargs,
        )

    def get_utility_base(self) -> str:
        return self.template.format(iteration=0)

    def get_utility_refinement(
        self,
        best_solution: Dict[str, Any],
        weights: Dict[str, float],
        formula: Optional[str] = None,
        description: Optional[str] = None,
        iteration: int = 1,
        policy_guidance: Optional[str] = None
    ) -> str:
        return self.template.format(
            iteration=iteration,
            best_solution=best_solution,
            weights=weights,
            formula=formula,
            description=description,
            policy_guidance=policy_guidance
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
        return self.template.parse_response(response)


__all__ = ["UtilityPromptTemplate", "UtilityPromptAdapter"]
