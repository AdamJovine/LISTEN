import pandas as pd
import numpy as np
import concurrent.futures
import time
import random
from collections import defaultdict, Counter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from typing import Dict, List, Optional, Tuple, Set
from LLM import FreeLLMPreferenceClient
from AcqStrategy import AcquisitionStrategy
from prefAcq import PrefAcquisitionConfig, PreferenceAcquisition
from Logistic import LinearLogisticModel
from GP import GPUtilityModel

# batch_pref_learning.py
import pandas as pd
import numpy as np
import concurrent.futures
import time
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from sklearn.impute import SimpleImputer

from prefAcq import PrefAcquisitionConfig, PreferenceAcquisition
from Utilitymodel import UtilityModel


class BatchPrefLearning:
    """
    Batch-based preference learning using a pluggable UtilityModel (logistic or GP).
    """

    def __init__(
        self,
        schedules_df: pd.DataFrame,
        llm_client,
        metric_columns: Optional[List[str]] = None,
        m_samples: int = 5,  # LLM voters per comparison
        batch_size: int = 5,  # pairs per batch
        max_workers: int = 5,
        reflexion: bool = False,
        reasoning_history: bool = False,
        history_file: str = "batch_preference_history.csv",
        voter_history_file: str = "voter_reflection_histories.csv",
        global_history_file: str = "global_reflection_history.csv",
        # NEW: choose model + acquisition
        model_type: str = "logistic",  # "logistic" | "gp"
        acq_mode: str = "eubo",  # "eubo" | "info_gain" | "logistic_entropy" | "random" | "fixed"
        n_theta_samples: int = 64,
        exploration_rate: float = 0.0,
        utility_cols: Optional[int] = None,
    ):
        # Basic config
        self.batch_size = batch_size
        self.m_samples = m_samples
        self.max_workers = max_workers
        self.reflexion = reflexion
        self.reasoning_history = reasoning_history
        self.history_file = history_file
        self.voter_history_file = voter_history_file
        self.global_history_file = global_history_file

        # Data & client
        self.df = schedules_df.reset_index(drop=True)
        self.metrics = metric_columns
        self.client = llm_client

        # Features (impute)
        raw_feat = self.df[self.metrics]
        imputer = SimpleImputer(strategy="mean")
        self.feat = imputer.fit_transform(raw_feat.to_numpy(dtype=float))
        self.all_idx = list(range(len(self.df)))

        # Acquisition
        self.acquisition = PreferenceAcquisition(
            PrefAcquisitionConfig(
                mode=acq_mode,
                n_theta_samples=n_theta_samples,
                exploration_rate=exploration_rate,
                utility_cols=utility_cols,
            )
        )

        # Utility model
        self.model_type = model_type
        if model_type == "logistic":
            self.utility_model: UtilityModel = LinearLogisticModel(C=1.0)
        elif model_type == "gp":
            self.utility_model = GPUtilityModel(kernel=Matern(nu=2.5))
            # prepare item store for GP counts
            self.utility_model.ensure_item_store(
                n_items=len(self.feat), X_items=self.feat
            )
        else:
            raise ValueError("model_type must be 'logistic' or 'gp'")

        # Training storage for duels (common)
        self.X_delta: List[np.ndarray] = []
        self.y01: List[int] = []

        # Tracking
        self.history: List[Dict] = []
        self.batch_history: List[Dict] = []
        self.voter_reflections = {i: [] for i in range(m_samples)}
        self.global_reflection_history: List[str] = []
        self.compared_pairs = set()
        self.previous_winners: List[int] = []

        # Cumulative votes per item (also used by GP pseudo-utility)
        self.cumulative_votes = defaultdict(lambda: {"for": 0, "against": 0})

    # ---------------------------
    # Pair selection
    # ---------------------------
    def _select_initial_batch(self) -> List[Tuple[int, int]]:
        if hasattr(self, "initial_pairs"):
            return self.initial_pairs
        random.seed(42)
        np.random.seed(42)
        pairs = []
        chosen = set()
        while len(pairs) < self.batch_size:
            i, j = np.random.choice(self.all_idx, 2, replace=False)
            p = (min(i, j), max(i, j))
            if p not in chosen:
                pairs.append((i, j))
                chosen.add(p)
        self.initial_pairs = pairs
        return pairs

    def _select_batch_with_model(self) -> List[Tuple[int, int]]:
        print(f"Previous winners: {self.previous_winners}")

        if self.previous_winners:
            candidate_pairs: List[Tuple[int, int]] = []
            for winner in self.previous_winners:
                challengers_all = [
                    j
                    for j in self.all_idx
                    if j != winner
                    and ((min(winner, j), max(winner, j)) not in self.compared_pairs)
                ]
                if not challengers_all:
                    continue
                challengers = random.sample(
                    challengers_all, min(500, len(challengers_all))
                )

                scored = []
                for challenger in challengers:
                    score = self.acquisition.compute_score(
                        winner, challenger, self.feat, self.utility_model, None
                    )
                    scored.append(((winner, challenger), score))
                scored.sort(key=lambda t: t[1], reverse=True)
                if scored:
                    candidate_pairs.append(scored[0][0])
            if candidate_pairs:
                return candidate_pairs

        # fallback: if model not ready, random; else, sample and score pairs
        if not self.utility_model.ready():
            return self._select_initial_batch()

        pool = []
        tries = 0
        target = self.batch_size * 10
        while len(pool) < target and tries < target * 5:
            a, b = np.random.choice(self.all_idx, 2, replace=False)
            p = (min(a, b), max(a, b))
            if p not in self.compared_pairs:
                pool.append((a, b))
            tries += 1

        scored_pool = [
            (
                (i, j),
                self.acquisition.compute_score(
                    i, j, self.feat, self.utility_model, None
                ),
            )
            for (i, j) in pool
        ]
        scored_pool.sort(key=lambda t: t[1], reverse=True)
        return [
            p for p, _ in scored_pool[: self.batch_size]
        ] or self._select_initial_batch()

    # ---------------------------
    # Voting & history
    # ---------------------------
    def _collect_batch_comparisons(
        self, pairs: List[Tuple[int, int]], prompt_init: Optional[str] = None
    ) -> List[Dict]:
        print(f"\nCollecting comparisons for batch of {len(pairs)} pairs...")
        batch_results: List[Dict] = []
        batch_winners: List[int] = []

        for k, (idx_a, idx_b) in enumerate(pairs):
            print(f"\n  Pair {k+1}/{len(pairs)}: {idx_a} vs {idx_b}")
            votes, responses, refl_summary, full_prompt, stats = (
                self._make_multiple_llm_votes(idx_a, idx_b, self.m_samples, prompt_init)
            )
            if stats.get("error"):
                print(f"    Skipping due to: {stats['error']}")
                continue

            champion = (
                idx_a
                if stats["winner"] == "A"
                else (idx_b if stats["winner"] == "B" else None)
            )
            if champion is not None:
                batch_winners.append(champion)

            result = {
                "idx_a": idx_a,
                "idx_b": idx_b,
                "champion_idx": champion,
                "votes": votes,
                "responses": responses,
                "group_reflection": refl_summary,
                "full_prompt": full_prompt,
                **stats,
            }
            batch_results.append(result)
            print(f"    Votes: {votes}, Winner: {champion}")

            # Update cumulative votes and compared set
            self._update_cumulative_votes(idx_a, idx_b, votes)
            self.compared_pairs.add((min(idx_a, idx_b), max(idx_a, idx_b)))

        self.previous_winners = list(set(batch_winners))
        return batch_results

    def _update_cumulative_votes(self, idx_a: int, idx_b: int, votes: Dict[str, int]):
        self.cumulative_votes[idx_a]["for"] += votes["A"]
        self.cumulative_votes[idx_a]["against"] += votes["B"]
        self.cumulative_votes[idx_b]["for"] += votes["B"]
        self.cumulative_votes[idx_b]["against"] += votes["A"]

    # ---------------------------
    # Model updates
    # ---------------------------
    def _update_model_batch(self, batch_results: List[Dict]):
        """Accumulate duels and fit/refresh the utility model."""
        print(f"\nUpdating model with {len(batch_results)} new comparisons...")

        # Accumulate new duels
        new_duels: List[Tuple[int, int, int]] = []
        for r in batch_results:
            i, j = r["idx_a"], r["idx_b"]
            if r["winner"] == "A":
                y01 = 1
            elif r["winner"] == "B":
                y01 = 0
            else:
                continue  # skip ties
            self.X_delta.append(self.feat[i] - self.feat[j])
            self.y01.append(y01)
            new_duels.append((i, j, y01))

        if not new_duels:
            return

        # If GP, update item win/loss counts first
        if isinstance(self.utility_model, GPUtilityModel):
            self.utility_model.observe_duels(new_duels)

        # Fit/update model on duels
        X = np.vstack(self.X_delta)
        y = np.array(self.y01, dtype=int)
        self.utility_model.fit_on_duels(X, y)

    # ---------------------------
    # Rankings
    # ---------------------------
    def _get_current_rankings(self) -> pd.DataFrame:
        mu = (
            self.utility_model.posterior_mean_util(self.feat)
            if self.utility_model.ready()
            else np.zeros(len(self.feat))
        )
        sigma = np.zeros_like(mu)  # placeholder; expose predictive std if you add it

        rows = []
        for idx in range(len(self.df)):
            cv = self.cumulative_votes[idx]
            total = cv["for"] + cv["against"]
            rows.append(
                {
                    **self.df.iloc[idx].to_dict(),
                    "schedule_idx": idx,
                    "model_score": float(mu[idx]),
                    "model_uncertainty": float(sigma[idx]),
                    "votes_for": cv["for"],
                    "votes_against": cv["against"],
                    "total_votes": total,
                    "win_rate": (cv["for"] / total) if total > 0 else 0.5,
                }
            )
        rows.sort(key=lambda r: r["model_score"], reverse=True)
        return pd.DataFrame(rows)

    # ---------------------------
    # Main loop
    # ---------------------------
    def run(
        self,
        n_batches: int = 20,
        prompt_init: Optional[str] = None,
        save_history: bool = True,
    ) -> pd.DataFrame:
        print(f"Starting batch preference learning (model={self.model_type})")
        print(
            f"Batches: {n_batches}, Batch size: {self.batch_size}, Voters: {self.m_samples}"
        )

        for batch_num in range(1, n_batches + 1):
            print(f"\n{'='*60}\nBATCH {batch_num}/{n_batches}\n{'='*60}")

            if batch_num == 1:
                pairs = self._select_initial_batch()
                print(f"Initial batch: selected {len(pairs)} random pairs")
            else:
                pairs = self._select_batch_with_model()
                print(
                    f"Model-guided batch: selected {len(pairs)} pairs using {self.model_type}"
                )

            batch_results = self._collect_batch_comparisons(pairs, prompt_init)
            self._update_model_batch(batch_results)

            batch_summary = {
                "batch_num": batch_num,
                "n_comparisons": len(batch_results),
                "total_comparisons_so_far": len(self.X_delta),
                "pairs": pairs,
                "results": batch_results,
            }
            self.batch_history.append(batch_summary)

            for i, r in enumerate(batch_results):
                r["batch_num"] = batch_num
                r["comparison_num_in_batch"] = i + 1
                r["total_comparison_num"] = len(self.history) + 1
                self.history.append(r)

            print(f"\nCurrent top 5 schedules after batch {batch_num}:")
            current_rankings = self._get_current_rankings()
            print(
                current_rankings.head()[
                    ["schedule_idx", "model_score", "win_rate", "total_votes"]
                ]
            )

        final_rankings = self._get_current_rankings()
        if save_history:
            self._save_history()
            self._save_batch_summary()
        print(
            f"\nCompleted {n_batches} batches with {len(self.history)} total comparisons"
        )
        return final_rankings

    # ---------------------------
    # (Voting code: unchanged from your version)
    # ---------------------------
    def _make_single_comparison(
        self, sched_a: Dict, sched_b: Dict, prompt_init: Optional[str], sample_id: int
    ):
        start_time = time.time()
        try:
            reflection_context = self._build_global_reflection_context()
            reasoning_context = self._build_reasoning_history_context(sample_id)

            raw_prompt = self.client._format_prompt_pairwise(sched_a, sched_b)
            full_prompt_parts = []
            if reasoning_context:
                full_prompt_parts.append(reasoning_context)
            if reflection_context:
                full_prompt_parts.append(reflection_context)
            if prompt_init:
                full_prompt_parts.append(prompt_init)
                full_prompt_parts.append("\n\n")
            full_prompt_parts.append(raw_prompt)
            full_prompt = "".join(full_prompt_parts)
            raw = self.client._call_api(full_prompt)
            choice, reason = self.client._parse_pairwise(raw)

            self.voter_reflections[sample_id].append(
                {
                    "sched_a": sched_a,
                    "sched_b": sched_b,
                    "winner": choice,
                    "reason": reason.strip(),
                }
            )
            if choice in ["A", "B"]:
                elapsed = time.time() - start_time
                print(f"    Voter {sample_id}: {choice} ({elapsed:.1f}s)")
                return sample_id, (choice, reason, None), full_prompt
            else:
                return sample_id, None, full_prompt

        except Exception as e:
            print(f"    Voter {sample_id}: Error - {e}")
            return sample_id, None, "full_prompt"

    def _make_multiple_llm_votes(
        self, idx_a: int, idx_b: int, n_samples: int, prompt_init: Optional[str] = None
    ):
        sched_a = dict(zip(self.metrics, self.feat[idx_a]))
        sched_b = dict(zip(self.metrics, self.feat[idx_b]))
        votes = {"A": 0, "B": 0}
        responses = []

        print(f"    Launching {n_samples} parallel LLM votes...")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(n_samples, self.max_workers)
        ) as ex:
            futures = [
                ex.submit(
                    self._make_single_comparison, sched_a, sched_b, prompt_init, i
                )
                for i in range(n_samples)
            ]
            for fut in concurrent.futures.as_completed(futures):
                sample_id, result, full_prompt = fut.result()
                if result is not None:
                    choice, reason, reflection = result
                    votes[choice] += 1
                    responses.append((choice, reason, reflection))

        total_votes = votes["A"] + votes["B"]
        if total_votes == 0:
            return votes, responses, None, None, {"error": "No valid votes received"}

        vote_ratio_a = votes["A"] / total_votes
        vote_ratio_b = votes["B"] / total_votes
        entropy = 0.0
        if vote_ratio_a > 0:
            entropy -= vote_ratio_a * np.log2(vote_ratio_a)
        if vote_ratio_b > 0:
            entropy -= vote_ratio_b * np.log2(vote_ratio_b)
        vote_margin = abs(votes["A"] - votes["B"]) / total_votes
        confidence = abs(vote_ratio_a - 0.5) * 2

        stats = {
            "total_votes": total_votes,
            "vote_ratio_a": vote_ratio_a,
            "vote_ratio_b": vote_ratio_b,
            "entropy": entropy,
            "vote_margin": vote_margin,
            "confidence": confidence,
            "winner": (
                "A"
                if votes["A"] > votes["B"]
                else ("B" if votes["B"] > votes["A"] else "tie")
            ),
        }
        reflection_summary = None
        if self.reflexion:
            pass

        print(
            f"    Result: {votes['A']}-{votes['B']} (confidence: {confidence:.2f}, entropy: {entropy:.2f})"
        )
        return votes, responses, reflection_summary, full_prompt, stats

    def _build_reasoning_history_context(
        self, voter_id: int, max_history: int = 10
    ) -> str:
        if self.reasoning_history:
            entries = self.voter_reflections.get(voter_id, [])
            if not entries:
                return ""
            recent = entries[-max_history:]
            lines = []
            for e in recent:
                lines.append(
                    f"Compared Schedule A ({e.get('sched_a','')}) and "
                    f"Schedule B ({e.get('sched_b','')}); Winner: {e.get('winner','')}; "
                    f"Reasoning: {e.get('reason','').strip()}"
                )
            return "\n".join(lines) + "\n\n"
        return ""

    def _build_global_reflection_context(self, max_reflections: int = 10) -> str:
        return ""

    # ---------------------------
    # Persistence
    # ---------------------------
    def _save_history(self):
        rows = []
        for rec in self.history:
            base = {
                "batch_num": rec.get("batch_num"),
                "comparison_num": rec.get("total_comparison_num"),
                "idx_a": rec["idx_a"],
                "idx_b": rec["idx_b"],
                "champion_idx": rec.get("champion_idx"),
                "total_votes": rec["total_votes"],
                "vote_ratio_a": rec["vote_ratio_a"],
                "vote_ratio_b": rec["vote_ratio_b"],
                "entropy": rec["entropy"],
                "confidence": rec["confidence"],
                "winner": rec["winner"],
            }
            for i, (choice, reason, reflection) in enumerate(rec["responses"]):
                row = dict(base)
                row["voter_id"] = i + 1
                row["choice"] = choice
                row["reason"] = reason
                rows.append(row)
        pd.DataFrame(rows).to_csv(self.history_file, index=False)
        print(f"Saved detailed history to {self.history_file}")

    def _save_batch_summary(self):
        batch_rows = []
        for batch in self.batch_history:
            vote_margins, entropies, confidences = [], [], []
            for r in batch["results"]:
                vote_margins.append(abs(r["vote_ratio_a"] - r["vote_ratio_b"]))
                entropies.append(r["entropy"])
                confidences.append(r["confidence"])
            batch_rows.append(
                {
                    "batch_num": batch["batch_num"],
                    "n_comparisons": batch["n_comparisons"],
                    "total_comparisons_so_far": batch["total_comparisons_so_far"],
                    "avg_vote_margin": np.mean(vote_margins) if vote_margins else 0,
                    "avg_entropy": np.mean(entropies) if entropies else 0,
                    "avg_confidence": np.mean(confidences) if confidences else 0,
                    "min_confidence": min(confidences) if confidences else 0,
                    "max_confidence": max(confidences) if confidences else 0,
                }
            )
        pd.DataFrame(batch_rows).to_csv("batch_summary.csv", index=False)
        print("Saved batch summary to batch_summary.csv")
