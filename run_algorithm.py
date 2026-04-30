from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from baseline_algorithm import BaselineExperiment
from full_batch_algorithm import FullBatchExperiment
from gemini_client import GeminiPreferenceClient
from groq_client import FreeLLMPreferenceClient
from prompt_tournament import ComparisonPromptAdapter
from prompt_utility import UtilityPromptTemplate
from tournament_algorithm import TournamentExperiment
from utility_algorithm import UtilityExperiment

REPO_ROOT = Path(__file__).resolve().parent


def get_git_info() -> Dict[str, Any]:
    """Capture git state for reproducibility: commit hash, branch, dirty status."""
    info: Dict[str, Any] = {
        "git_hash": None,
        "git_branch": None,
        "git_dirty": None,
        "git_dirty_files": [],
    }
    try:
        info["git_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["git_dirty"] = bool(status)
        if status:
            info["git_dirty_files"] = status.splitlines()
            try:
                diff = subprocess.check_output(
                    ["git", "diff", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
                ).decode()
                if len(diff) > 50_000:
                    diff = diff[:50_000] + "\n... (truncated)"
                info["git_diff"] = diff
            except Exception:
                info["git_diff"] = None
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError:
        pass
    return info


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run LISTEN algorithms (tournament, utility, baseline, or full_batch)."
    )
    ap.add_argument(
        "--algo",
        default="tournament",
        choices=["tournament", "baseline", "utility", "full_batch"],
        help="Algorithm to run.",
    )
    ap.add_argument("--scenario", required=True, help="Scenario name matching configs/<scenario>.yml")
    ap.add_argument("--mode", help="Scenario mode; falls back to default_mode in config")
    ap.add_argument("--iterations", type=int, help="Total LLM calls (T >= 3). Defaults to config n_batches.")
    ap.add_argument("--batch-size", dest="batch_size", type=int, help="Batch size B for prelim rounds.")
    ap.add_argument("--seed", type=int, help="Random seed override.")
    ap.add_argument("--api-model", help="API model key from config model_configs (groq or gemini).")
    ap.add_argument("--model-name", help="Concrete model name override.")
    ap.add_argument("--reasoning", action="store_true", help="Include reasoning in prompts.")
    ap.add_argument("--use-history", action="store_true", help="Include prompt history.")
    ap.add_argument("--unique-rank-batch", dest="unique_rank_batch", action="store_true",
                     help="Ensure each batch contains at least one human_sol index (tournament only).")
    ap.add_argument(
        "--comparison-prompt-strategy",
        choices=["fixed", "round_robin", "random"],
        help="Override comparison prompt variant strategy.",
    )
    ap.add_argument(
        "--comparison-prompt-variant",
        help="Override comparison prompt default variant (name or index).",
    )
    ap.add_argument(
        "--comparison-prompt-seed",
        type=int,
        help="Seed used for random comparison prompt variant selection.",
    )
    ap.add_argument(
        "--utility-prompt-strategy",
        choices=["fixed", "round_robin", "random"],
        help="Override utility prompt variant strategy.",
    )
    ap.add_argument(
        "--utility-prompt-variant",
        help="Override utility prompt default variant (name or index).",
    )
    ap.add_argument(
        "--utility-prompt-seed",
        type=int,
        help="Seed used for random utility prompt variant selection.",
    )
    ap.add_argument(
        "--output-root",
        help="Directory to place output files. Default: outputs/<scenario>.",
    )
    ap.add_argument(
        "--run-stamp",
        dest="run_stamp",
        help="Timestamp to embed in filenames (YYYYMMDD_HHMMSS). Defaults to current UTC time.",
    )
    ap.add_argument(
        "--temperature", type=float,
        help="Override temperature for the active api-model.",
    )
    return ap.parse_args()


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.api_model:
        cfg["api_model"] = args.api_model
    if args.model_name:
        am = cfg.get("api_model")
        if am:
            cfg.setdefault("model_configs", {})
            cfg["model_configs"].setdefault(am, {})
            cfg["model_configs"][am]["model_name"] = args.model_name
    if args.batch_size is not None:
        cfg["tournament_batch_size"] = args.batch_size
    if args.iterations is not None:
        cfg["n_batches"] = args.iterations
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.comparison_prompt_strategy:
        cfg["comparison_prompt_strategy_override"] = args.comparison_prompt_strategy
    if args.comparison_prompt_variant is not None:
        variant_raw = args.comparison_prompt_variant
        if isinstance(variant_raw, str) and variant_raw.strip().isdigit():
            cfg["comparison_prompt_variant_override"] = int(variant_raw.strip())
        else:
            cfg["comparison_prompt_variant_override"] = variant_raw
    if args.comparison_prompt_seed is not None:
        cfg["comparison_prompt_seed_override"] = args.comparison_prompt_seed
    if args.utility_prompt_strategy:
        cfg["utility_prompt_strategy_override"] = args.utility_prompt_strategy
    if args.utility_prompt_variant is not None:
        variant_raw = args.utility_prompt_variant
        if isinstance(variant_raw, str) and variant_raw.strip().isdigit():
            cfg["utility_prompt_variant_override"] = int(variant_raw.strip())
        else:
            cfg["utility_prompt_variant_override"] = variant_raw
    if args.utility_prompt_seed is not None:
        cfg["utility_prompt_seed_override"] = args.utility_prompt_seed
    cfg["reasoning"] = bool(args.reasoning)
    cfg["use_history"] = bool(args.use_history)
    if args.temperature is not None:
        am = cfg.get("api_model")
        if am:
            cfg.setdefault("model_configs", {})
            cfg["model_configs"].setdefault(am, {})
            cfg["model_configs"][am]["temperature"] = args.temperature


def _load_env_file():
    base_dir = Path(__file__).resolve().parent
    env_path = base_dir / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
    except Exception as e:
        print(f"[WARN] Failed to load .env: {e}")


def _load_global_defaults() -> dict:
    cfg_dir = Path(__file__).resolve().parent / "configs"
    cfg_path = cfg_dir / "config.yml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f) or {}


def _find_scenario_yaml(scenario: str) -> Path:
    cfg_dir = Path(__file__).resolve().parent / "configs"
    candidates = []
    if scenario.endswith((".yaml", ".yml")):
        candidates.append(cfg_dir / scenario)
    else:
        candidates.append(cfg_dir / f"{scenario}.yml")
        candidates.append(cfg_dir / f"{scenario}.yaml")
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Scenario '{scenario}' not found in configs/")


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(scenario: str) -> dict:
    cfg = _load_global_defaults()
    scenario_path = _find_scenario_yaml(scenario)
    override = _load_yaml(str(scenario_path))
    for k, v in override.items():
        if k == "model_configs":
            cfg["model_configs"] = {**cfg.get("model_configs", {}), **v}
        elif k == "prompts":
            cfg["prompts"] = {**cfg.get("prompts", {}), **v}
        else:
            cfg[k] = v
    for req in ["metric_columns", "data_csv", "tag"]:
        if req not in cfg:
            raise ValueError(f"Scenario config missing '{req}'")
    return cfg


def resolve_mode(cfg: dict, mode_cli: str | None):
    modes = cfg.get("modes")
    if not modes:
        if mode_cli:
            cfg["mode"] = mode_cli
        return
    selected = mode_cli or cfg.get("mode") or cfg.get("default_mode")
    if not selected:
        raise ValueError("Mode not specified; pass --mode or set default_mode in config")
    if selected not in modes:
        raise ValueError(f"Mode '{selected}' not found in scenario modes: {list(modes.keys())}")
    mode_def = modes[selected] or {}
    cfg["mode"] = selected
    cfg["mode_weights"] = mode_def.get("weights")
    cfg["utility_prompt_text"] = mode_def.get("prompt") or mode_def.get("utility_prompt_text")


def _build_llm_params(cfg: Dict[str, Any], api_model: str | None) -> Dict[str, Any]:
    if not api_model:
        return {}
    mc = cfg.get("model_configs", {}).get(api_model, {}) or {}
    params: Dict[str, Any] = {}
    for key in ("temperature", "top_p", "stop", "seed"):
        if mc.get(key) is not None:
            params[key] = mc[key]
    max_out = mc.get("max_output_tokens") if mc.get("max_output_tokens") is not None else mc.get("max_tokens")
    if max_out is not None:
        params["max_new_tokens"] = max_out
    return params


def _render_base_prompt_text(
    template: str,
    scenario_header: str,
    policy_guidance: str = "",
    prompt_vars: Optional[Dict[str, Any]] = None,
) -> str:
    from prompt_utility import _apply_prompt_vars

    rendered = template.replace("{scenario_header}", scenario_header)
    rendered = _apply_prompt_vars(rendered, prompt_vars or {})
    if policy_guidance:
        rendered = f"{rendered.strip()}\n\nPolicy guidance: {policy_guidance.strip()}\n"
    return rendered


def _render_comparison_prompt_config(
    comparison_template: Any,
    scenario_header: str,
    policy_guidance: str = "",
    prompt_vars: Optional[Dict[str, Any]] = None,
) -> Any:
    """Render comparison prompt config while preserving shape (str/list/dict)."""
    if isinstance(comparison_template, str):
        return _render_base_prompt_text(comparison_template, scenario_header, policy_guidance, prompt_vars)

    if isinstance(comparison_template, list):
        return [
            _render_base_prompt_text(str(tpl), scenario_header, policy_guidance, prompt_vars)
            for tpl in comparison_template
        ]

    if isinstance(comparison_template, dict):
        rendered_cfg = dict(comparison_template)
        variants = rendered_cfg.get("variants")

        if variants is None:
            bare_mapping = {
                k: _render_base_prompt_text(str(v), scenario_header, policy_guidance, prompt_vars)
                for k, v in comparison_template.items()
                if k not in {"strategy", "mode", "default_variant", "seed"}
            }
            if not bare_mapping:
                raise ValueError("prompts.comparison_base must include at least one template variant.")
            return bare_mapping

        if isinstance(variants, dict):
            rendered_cfg["variants"] = {
                name: _render_base_prompt_text(str(tpl), scenario_header, policy_guidance, prompt_vars)
                for name, tpl in variants.items()
            }
            return rendered_cfg

        if isinstance(variants, list):
            rendered_variants = []
            for entry in variants:
                if isinstance(entry, dict):
                    entry_copy = dict(entry)
                    if "template" not in entry_copy:
                        raise ValueError("Each dict entry in prompts.comparison_base.variants must include 'template'.")
                    entry_copy["template"] = _render_base_prompt_text(
                        str(entry_copy["template"]),
                        scenario_header,
                        policy_guidance,
                        prompt_vars,
                    )
                    rendered_variants.append(entry_copy)
                else:
                    rendered_variants.append(
                        _render_base_prompt_text(str(entry), scenario_header, policy_guidance, prompt_vars)
                    )
            rendered_cfg["variants"] = rendered_variants
            return rendered_cfg

        raise TypeError("prompts.comparison_base.variants must be a dict or list.")

    raise TypeError("prompts.comparison_base must be a string, list, or dict.")


def _build_comparison_prompt_template(config: Dict[str, Any]) -> ComparisonPromptAdapter:
    """Build a ComparisonPromptAdapter from config (used by tournament, full_batch)."""
    prompts_config = config.get("prompts", {})
    scenario_header = prompts_config.get("scenario_header", "")
    comparison_template = prompts_config.get("comparison_base", "{scenario_header}")
    policy_guidance = config.get("utility_prompt_text", "")

    strategy_override = config.get("comparison_prompt_strategy_override")
    variant_override = config.get("comparison_prompt_variant_override")
    variant_seed_override = config.get("comparison_prompt_seed_override")
    if variant_override is not None:
        if not isinstance(comparison_template, dict):
            raise ValueError(
                "comparison prompt variant override requires prompts.comparison_base to be a dict with variants."
            )
        comparison_template = dict(comparison_template)
        comparison_template["default_variant"] = variant_override
        if not strategy_override:
            strategy_override = "fixed"

    rendered_prompt_config = _render_comparison_prompt_config(
        comparison_template,
        scenario_header=scenario_header,
        policy_guidance=policy_guidance,
        prompt_vars=config.get("prompt_vars", {}),
    )

    return ComparisonPromptAdapter(
        base_prompt=rendered_prompt_config,
        reasoning_history=config.get("use_history", False),
        metric_columns=config["metric_columns"],
        reasoning=config.get("reasoning", False),
        prompt_variant_strategy=strategy_override,
        prompt_variant_seed=variant_seed_override,
    )


def _build_utility_prompt_template(config: Dict[str, Any]) -> UtilityPromptTemplate:
    """Build a UtilityPromptTemplate from config with variant overrides."""
    strategy_override = config.get("utility_prompt_strategy_override")
    variant_override = config.get("utility_prompt_variant_override")
    variant_seed_override = config.get("utility_prompt_seed_override")

    if variant_override is not None:
        prompts = config.get("prompts", {})
        utility_base = prompts.get("utility_base")
        if not isinstance(utility_base, dict):
            raise ValueError(
                "utility prompt variant override requires prompts.utility_base to be a dict with variants."
            )
        utility_base = dict(utility_base)
        utility_base["default_variant"] = variant_override
        if not strategy_override:
            strategy_override = "fixed"
        prompts["utility_base"] = utility_base
        config["prompts"] = prompts

    return UtilityPromptTemplate(
        config=config,
        reasoning=config.get("reasoning", True),
        policy_guidance=config.get("utility_prompt_text", ""),
        prompt_variant_strategy=strategy_override,
        prompt_variant_seed=variant_seed_override,
    )


def create_client(api_model: str, model_configs: dict):
    mc = dict(model_configs.get(api_model) or {})
    if api_model == "groq":
        return FreeLLMPreferenceClient(
            provider="groq",
            api_key=mc.get("api_key") or os.getenv(mc.get("api_key_env", "GROQ_API_KEY")),
            model_name=mc.get("model_name", "llama-3.3-70b-versatile"),
            max_tokens=mc.get("max_output_tokens", 8192),
            max_retries=mc.get("max_retries", 20),
            rate_limit_delay=mc.get("rate_limit_delay", 0.1),
            default_seed=mc.get("seed"),
        )
    if api_model == "gemini":
        api_key = mc.get("api_key") or os.getenv(mc.get("api_key_env", "GEMINI_API_KEY")) or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not found. Set GEMINI_API_KEY/GOOGLE_API_KEY or model_configs.gemini.api_key.")
        return GeminiPreferenceClient(
            api_key=api_key,
            model_name=mc.get("model_name", "gemini-1.5-pro"),
            simple=False,
            rate_limit_delay=mc.get("rate_limit_delay", 0.1),
            max_tokens=mc.get("max_output_tokens", 8192),
            max_retries=mc.get("max_retries", 4),
            default_seed=mc.get("seed"),
        )
    raise ValueError(f"Unsupported api_model '{api_model}'. Supported: groq, gemini.")


def build_output_filename(
    scenario: str,
    algo: str,
    mode_name: str,
    api_model: str,
    model_name_short: str,
    seed: int | None,
    run_stamp: str | None,
    max_iters: int,
    batch_size: int | None = None,
    comparison_settings: dict | None = None,
    base_output_dir: Path | None = None,
):
    """Build a unique identifier for a run and return the target file path."""
    if run_stamp is None:
        run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    parts = [
        f"{scenario}__{algo}__{mode_name}",
        f"api{api_model}",
        f"llm{model_name_short}",
    ]
    if comparison_settings:
        parts.append("__".join(f"{k}{v}" for k, v in comparison_settings.items()))
    if batch_size is not None:
        parts.append(f"B{batch_size}")
    parts.append(f"iters{max_iters}")
    if seed is not None:
        parts.append(f"seed{seed}")
    parts.append(run_stamp)

    identifier = "__".join(parts)

    if base_output_dir:
        output_dir = Path(base_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{identifier}.json"

    return Path(f"{identifier}.json")


def main():
    _load_env_file()
    args = parse_args()
    config = load_config(args.scenario)
    apply_overrides(config, args)
    resolve_mode(config, args.mode)

    data_path = Path(__file__).resolve().parent / config["data_csv"]
    non_numeric = set(config.get("non_numeric_metrics") or config.get("non_numerical_metrics") or [])
    numeric_metric_columns = [c for c in config["metric_columns"] if c not in non_numeric]
    df = pd.read_csv(data_path).dropna(subset=numeric_metric_columns)

    client = create_client(config["api_model"], config["model_configs"])

    iterations = int(args.iterations or config.get("n_batches", 25))
    batch_size_default = config.get("tournament_batch_size", config.get("batch_size", 50))
    batch_size = int(args.batch_size) if args.batch_size is not None else int(batch_size_default)

    mode_def = (config.get("modes") or {}).get(config.get("mode"), {})
    weights = mode_def.get("weights") or config.get("mode_weights") or config.get("weights") or {}
    human_sol = mode_def.get("human_sol") or config.get("human_sol") or []

    if args.algo == "baseline":
        algo = BaselineExperiment(
            {
                "solutions_df": df,
                "metric_columns": config["metric_columns"],
                "non_numeric_metrics": config.get("non_numeric_metrics", []),
                "metric_signs": config.get("metric_signs", {}),
                "seed": config.get("seed"),
                "iterations": iterations,
                "gtu_weights": weights,
                "human_sol": human_sol,
            }
        )
    elif args.algo == "tournament":
        prompt_template = _build_comparison_prompt_template(config)
        algo = TournamentExperiment({
            "solutions_df": df,
            "metric_columns": config["metric_columns"],
            "non_numeric_metrics": config.get("non_numeric_metrics", []),
            "llm_client": client,
            "prompt_template": prompt_template,
            "batch_size": batch_size,
            "iterations": iterations,
            "seed": config.get("seed"),
            "llm_params": _build_llm_params(config, config.get("api_model")),
            "gtu_weights": weights,
            "human_sol": human_sol,
            "unique_rank_batch": args.unique_rank_batch,
        })
    elif args.algo == "utility":
        prompt_template = _build_utility_prompt_template(config)
        algo = UtilityExperiment({
            "solutions_df": df,
            "metric_columns": config["metric_columns"],
            "non_numeric_metrics": config.get("non_numeric_metrics", []),
            "llm_client": client,
            "prompt_template": prompt_template,
            "iterations": iterations,
            "seed": config.get("seed"),
            "llm_params": _build_llm_params(config, config.get("api_model")),
            "gtu_weights": weights,
            "human_sol": human_sol,
            "prompts": config.get("prompts", {}),
        })
    elif args.algo == "full_batch":
        prompt_template = _build_comparison_prompt_template(config)
        full_batch_size = int(args.batch_size) if args.batch_size is not None else None
        algo = FullBatchExperiment(
            {
                "solutions_df": df,
                "metric_columns": config["metric_columns"],
                "non_numeric_metrics": config.get("non_numeric_metrics", []),
                "llm_client": client,
                "prompt_template": prompt_template,
                "seed": config.get("seed"),
                "gtu_weights": weights,
                "human_sol": human_sol,
                "scenario": args.scenario,
                "batch_size": full_batch_size,
            }
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algo}")

    results = algo.run()

    winner_idx = results.get("final_winner_idx")
    if args.algo == "full_batch":
        effective_batch_size = full_batch_size or len(df)
    else:
        effective_batch_size = batch_size

    git_info = get_git_info()
    if git_info["git_dirty"]:
        print(f"[WARNING] Git working tree is dirty ({len(git_info['git_dirty_files'])} files modified). "
              "Results may not be reproducible from the committed code.")

    tag = config.get("tag") or Path(args.scenario).stem
    meta = {
        "scenario": tag,
        "algo": args.algo,
        "mode": config.get("mode"),
        "api_model": config.get("api_model"),
        "model_name": config.get("model_configs", {}).get(config.get("api_model"), {}).get("model_name"),
        "seed": config.get("seed"),
        "max_iters": iterations,
        "batch_size": effective_batch_size,
        "gtu": algo.get_gtu(winner_idx) if winner_idx is not None else None,
        "nar": algo.get_nar(),
        "config": config,
        "git": git_info,
    }

    run_stamp = args.run_stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir_path = (
        Path(args.output_root)
        if args.output_root
        else Path(__file__).resolve().parent / "outputs" / tag
    )
    outdir_path.mkdir(parents=True, exist_ok=True)
    model_name_short = meta["model_name"] or meta["api_model"] or "model"
    model_name_short = model_name_short.split("/")[-1]

    output_file = build_output_filename(
        scenario=tag,
        algo=args.algo,
        mode_name=config.get("mode") or "MODE",
        api_model=config.get("api_model") or "api",
        model_name_short=model_name_short,
        seed=config.get("seed"),
        run_stamp=run_stamp,
        max_iters=iterations,
        batch_size=effective_batch_size,
        comparison_settings=None,
        base_output_dir=outdir_path,
    )

    payload = {"meta": meta, "results": results}
    output_file.write_text(json.dumps(payload, indent=2, default=str))

    print("\n" + "=" * 64)
    print(f"Scenario: {args.scenario} | Mode: {meta['mode']} | Algo: {args.algo}")
    print(f"Winner idx: {winner_idx}")
    print(f"NAR: {meta['nar']} | GTU: {meta['gtu']}")
    print(f"Output: {output_file}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
