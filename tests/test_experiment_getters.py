import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment import Experiment
from run_algorithm import load_config, _build_comparison_prompt_template
from prompt_utility import UtilityPromptTemplate


ARTIFACT = Path(__file__).parent / "fixtures" / "exam_tournament_fixture.json"

# All scenarios to test
SCENARIOS = ["exam", "headphones", "flights_chi_nyc", "flights_ithaca_reston"]


def _extract_comparison_templates(comparison_base) -> list[str]:
    if isinstance(comparison_base, str):
        return [comparison_base]
    if isinstance(comparison_base, list):
        return [str(item) for item in comparison_base]
    if isinstance(comparison_base, dict):
        variants = comparison_base.get("variants")
        if variants is None:
            variants = comparison_base
        if isinstance(variants, dict):
            return [str(item) for item in variants.values()]
        if isinstance(variants, list):
            out = []
            for item in variants:
                if isinstance(item, dict):
                    out.append(str(item.get("template", "")))
                else:
                    out.append(str(item))
            return out
    raise TypeError("Unsupported prompts.comparison_base shape")


def _extract_utility_base_templates(utility_base) -> list[str]:
    """Extract template strings from utility_base (string or dict with variants)."""
    if isinstance(utility_base, str):
        return [utility_base]
    if isinstance(utility_base, dict):
        variants = utility_base.get("variants")
        if variants is None:
            variants = {
                k: v for k, v in utility_base.items()
                if k not in {"strategy", "mode", "default_variant", "seed"}
            }
        if isinstance(variants, dict):
            return [str(item) for item in variants.values()]
    raise TypeError("Unsupported prompts.utility_base shape")


def load_experiment() -> Experiment:
    assert ARTIFACT.exists(), f"Missing artifact: {ARTIFACT}"
    return Experiment.from_json(ARTIFACT)


def get_num_batches(exp: Experiment) -> int:
    return len(exp._history["batch_comparisons"])


def test_get_winner_matches_last_iteration():
    exp = load_experiment()
    num_batches = get_num_batches(exp)
    assert exp.get_winner() == exp.get_ith_winner(num_batches)


def test_get_ith_winner_out_of_range_raises():
    exp = load_experiment()
    num_batches = get_num_batches(exp)
    with pytest.raises(KeyError):
        exp.get_ith_winner(num_batches + 1)


def test_get_nar_respects_missing_winner():
    exp = load_experiment()
    if exp.get_winner() is None:
        assert exp.get_nar() is None


def test_get_gtu_returns_zero_without_weights():
    exp = load_experiment()
    assert exp.get_gtu(0) == 0.0


def test_get_scenario_mode_algo():
    exp = load_experiment()
    assert exp.get_scenario() == "exam"
    assert exp.get_mode() == "REGISTRAR"
    assert exp.get_algo() == "TournamentExperiment"


def test_get_prompt_and_response_present():
    exp = load_experiment()
    prompt = exp.get_prompt()
    print('PROMPT:', prompt)
    response = exp.get_llm_response()
    print('RESPONSE:', response)
    assert prompt is not None
    assert response is not None
    assert "FINAL:" in response


def test_get_rank_returns_number():
    exp = load_experiment()
    assert isinstance(exp.get_rank(0), (int, float))


# ---------------------------------------------------------------------------
# Prompt integration tests: verify prompts are correctly assembled from
# the merged global config.yml + scenario YAML for each algorithm type.
# ---------------------------------------------------------------------------

class TestUtilityPrompts:
    """Test that utility prompts contain the right components for each scenario."""

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_utility_base_contains_scenario_header(self, scenario):
        config = load_config(scenario)
        pt = UtilityPromptTemplate(config=config)
        prompt = pt.format(iteration=0)
        header = config["prompts"]["scenario_header"].strip().splitlines()[0]
        assert header in prompt

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_utility_base_contains_all_metric_columns(self, scenario):
        config = load_config(scenario)
        pt = UtilityPromptTemplate(config=config)
        prompt = pt.format(iteration=0)
        for col in config["metric_columns"]:
            assert col in prompt, f"Metric '{col}' missing from utility base prompt"

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_utility_base_contains_json_template(self, scenario):
        config = load_config(scenario)
        pt = UtilityPromptTemplate(config=config)
        prompt = pt.format(iteration=0)
        assert '"weights"' in prompt
        assert '"formula"' in prompt
        assert '"description"' in prompt

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_utility_base_contains_strict_rules(self, scenario):
        config = load_config(scenario)
        pt = UtilityPromptTemplate(config=config)
        prompt = pt.format(iteration=0)
        assert "STRICT OUTPUT RULES" in prompt
        assert "<END_JSON>" in prompt

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_utility_base_no_unresolved_placeholders(self, scenario):
        config = load_config(scenario)
        pt = UtilityPromptTemplate(config=config)
        prompt = pt.format(iteration=0)
        assert "{scenario_header}" not in prompt
        assert "{metric_weights_json}" not in prompt

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_utility_refinement_contains_all_placeholders_resolved(self, scenario):
        config = load_config(scenario)
        pt = UtilityPromptTemplate(config=config)
        cols = config["metric_columns"]
        weights = {col: 1.0 for col in cols}
        prompt = pt.format(
            iteration=1,
            best_solution={cols[0]: 42},
            weights=weights,
            formula="test_formula",
            description="test_description",
        )
        assert "iteration 1" in prompt.lower() or "iteration 1" in prompt
        assert "test_formula" in prompt
        assert "test_description" in prompt
        assert "{iteration}" not in prompt
        assert "{weights}" not in prompt
        assert "{formula}" not in prompt
        assert "{description}" not in prompt
        assert "{best_solution}" not in prompt
        assert "{metric_weights_json_escaped}" not in prompt

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_utility_refinement_contains_adjusted_weight_template(self, scenario):
        config = load_config(scenario)
        pt = UtilityPromptTemplate(config=config)
        cols = config["metric_columns"]
        weights = {col: 1.0 for col in cols}
        prompt = pt.format(
            iteration=1,
            best_solution={cols[0]: 42},
            weights=weights,
        )
        assert "<adjusted_weight>" in prompt
        assert "STRICT OUTPUT RULES" in prompt


class TestComparisonPrompts:
    """Test that comparison prompts (tournament/full_batch/baseline) are correct."""

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_comparison_prompt_contains_scenario_header(self, scenario):
        config = load_config(scenario)
        prompt_template = _build_comparison_prompt_template(config)
        base = prompt_template.get_base_prompt()
        header = config["prompts"]["scenario_header"].strip().splitlines()[0]
        assert header in base

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_comparison_prompt_contains_task_instruction(self, scenario):
        config = load_config(scenario)
        prompt_template = _build_comparison_prompt_template(config)
        base = prompt_template.get_base_prompt()
        comparison_noun = config.get("prompt_vars", {}).get("comparison_noun", "option")
        assert f"Identify the SINGLE BEST {comparison_noun}" in base

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_comparison_prompt_no_unresolved_placeholders(self, scenario):
        config = load_config(scenario)
        prompt_template = _build_comparison_prompt_template(config)
        base = prompt_template.get_base_prompt()
        assert "{scenario_header}" not in base

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_comparison_format_produces_json_block(self, scenario):
        config = load_config(scenario)
        prompt_template = _build_comparison_prompt_template(config)
        # Build dummy items using the first few metric columns
        cols = config["metric_columns"]
        items = [
            {col: i + j for j, col in enumerate(cols)}
            for i in range(2)
        ]
        formatted = prompt_template.format(items)
        assert "ITEMS TO COMPARE (JSON)" in formatted
        assert "respond with ONLY the single letter" in formatted

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_comparison_with_policy_guidance(self, scenario):
        """When a mode has a prompt (policy_guidance), it should appear in the base prompt."""
        config = load_config(scenario)
        # Pick first mode with a non-empty prompt
        modes = config.get("modes", {})
        for _, mode_def in modes.items():
            guidance = (mode_def or {}).get("prompt", "")
            if guidance and guidance.strip():
                config["utility_prompt_text"] = guidance
                prompt_template = _build_comparison_prompt_template(config)
                base = prompt_template.get_base_prompt()
                assert "Policy guidance:" in base
                break


class TestUtilityPromptVariants:
    """Test that utility prompt variants actually change the layout ordering."""

    def _make_config_with_variant(self, variant_name):
        """Load exam config and pin to a specific utility variant."""
        import copy
        config = load_config("exam")
        prompts = copy.deepcopy(config["prompts"])
        utility_base = prompts["utility_base"]
        utility_base["default_variant"] = variant_name
        prompts["utility_base"] = utility_base
        config["prompts"] = prompts
        return config

    def _get_prompt_for_variant(self, variant_name):
        config = self._make_config_with_variant(variant_name)
        pt = UtilityPromptTemplate(config=config, prompt_variant_strategy="fixed")
        return pt.format(iteration=0)

    def test_header_then_task_puts_header_before_task(self):
        prompt = self._get_prompt_for_variant("header_then_task_v1")
        header_pos = prompt.index("exam")  # scenario header mentions exam
        task_pos = prompt.index("Your task:")
        assert header_pos < task_pos, "header_then_task should place header before task"

    def test_task_then_header_puts_task_before_header(self):
        prompt = self._get_prompt_for_variant("task_then_header_v1")
        header_pos = prompt.index("exam")  # scenario header mentions exam
        task_pos = prompt.index("Your task:")
        assert task_pos < header_pos, "task_then_header should place task before header"

    def test_structured_sections_has_section_labels(self):
        prompt = self._get_prompt_for_variant("structured_sections_v1")
        assert "Weighting Objective:" in prompt
        assert "Scenario:" in prompt

    def test_v1_v2_v3_have_different_task_wording(self):
        p1 = self._get_prompt_for_variant("header_then_task_v1")
        p2 = self._get_prompt_for_variant("header_then_task_v2")
        p3 = self._get_prompt_for_variant("header_then_task_v3")
        assert "Define a utility function" in p1
        assert "Assign numerical weights" in p2
        assert "Produce a weighted utility function" in p3

    def test_fixed_strategy_returns_same_variant(self):
        config = self._make_config_with_variant("header_then_task_v1")
        pt = UtilityPromptTemplate(config=config, prompt_variant_strategy="fixed")
        p1 = pt.format(iteration=0)
        p2 = pt.format(iteration=0)
        assert p1 == p2

    def test_round_robin_cycles_through_variants(self):
        config = load_config("exam")
        pt = UtilityPromptTemplate(config=config, prompt_variant_strategy="round_robin")
        names = set()
        for _ in range(9):
            pt.format(iteration=0)
            names.add(pt.get_base_prompt_variant_name())
        assert len(names) >= 2, "round_robin should cycle through multiple variants"

    @pytest.mark.parametrize("variant", [
        "header_then_task_v1", "header_then_task_v2", "header_then_task_v3",
        "task_then_header_v1", "task_then_header_v2", "task_then_header_v3",
        "structured_sections_v1", "structured_sections_v2", "structured_sections_v3",
    ])
    def test_all_variants_resolve_placeholders(self, variant):
        prompt = self._get_prompt_for_variant(variant)
        assert "{scenario_header}" not in prompt
        assert "{metric_weights_json}" not in prompt
        assert "{utility_priorities}" not in prompt
        assert "{utility_task_body}" not in prompt
        assert "{utility_weight_rule}" not in prompt
        assert "{utility_key_rule}" not in prompt
        assert "STRICT OUTPUT RULES" in prompt
        assert '"weights"' in prompt


class TestConfigMerge:
    """Test that load_config correctly deep-merges prompts from global + scenario."""

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_merged_config_has_scenario_header(self, scenario):
        config = load_config(scenario)
        assert "scenario_header" in config["prompts"]
        assert config["prompts"]["scenario_header"].strip()

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_merged_config_has_global_templates(self, scenario):
        config = load_config(scenario)
        assert "utility_base" in config["prompts"]
        assert "utility_refinement" in config["prompts"]
        assert "comparison_base" in config["prompts"]

    def test_global_comparison_base_has_multiple_variants(self):
        config = load_config("exam")
        comparison_templates = _extract_comparison_templates(config["prompts"]["comparison_base"])
        non_empty = [template for template in comparison_templates if template.strip()]
        assert len(non_empty) >= 2

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_merged_config_templates_have_placeholders(self, scenario):
        config = load_config(scenario)
        utility_templates = _extract_utility_base_templates(config["prompts"]["utility_base"])
        assert any("{scenario_header}" in t for t in utility_templates)
        assert any("{metric_weights_json}" in t for t in utility_templates)
        assert "{iteration}" in config["prompts"]["utility_refinement"]
        comparison_templates = _extract_comparison_templates(config["prompts"]["comparison_base"])
        assert any("{scenario_header}" in template for template in comparison_templates)
