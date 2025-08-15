import re
from typing import Dict, Tuple, List, Optional
import pandas as pd
from oracle import Oracle

class SimpleUtilityPreferenceClient(Oracle):
    """
    Client that picks schedules based on maximizing utility score.
    Utility is calculated as the weighted sum of schedule metrics.
    
    Positive weights indicate desirable metrics, negative weights indicate undesirable ones.
    Always chooses the schedule with the highest total utility.
    
    Maintains the same interface as the LLM version for drop-in replacement.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, **kwargs):
        """
        Initialize with weights for utility calculation.
        
        Args:
            weights: Dictionary mapping metric names to weights.
                     Positive weights = desirable, negative = undesirable.
                     If None, defaults to minimizing back-to-back exams.
            **kwargs: Additional parameters (ignored for compatibility)
        """
        # Default weights if none provided (mimics original b2b minimization)
        if weights is None:
            self.weights = {
                "conflicts": -1000,  # Very bad
                "quints": -100,
                "quads": -50,
                "four in five slots": -40,
                "triple in 24h (no gaps)": -30,
                "triple in same day (no gaps)": -30,
                "three in four slots": -20,
                "evening/morning b2b": -1,
                "other b2b": -1,
                "two in three slots": -10,
            }
        else:
            self.weights = weights.copy()
        
        # Ignore all other parameters for compatibility

    # ---- Oracle interface ----
    def call_oracle(
        self,
        prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        stop: Optional[List[str]] = None,
    ) -> Tuple[str, str]:
        """
        Parse a prompt with 'Schedule A:' and 'Schedule B:' lines, compute utilities,
        and return ('A'|'B', explanation). Sampling args are ignored.
        """
        try:
            schedule_a, schedule_b = self._parse_prompt_schedules(prompt)
            utility_a = self._calculate_utility(schedule_a)
            utility_b = self._calculate_utility(schedule_b)

            summary_a = self._format_schedule_summary(schedule_a, "Schedule A", utility_a)
            summary_b = self._format_schedule_summary(schedule_b, "Schedule B", utility_b)

            if utility_a >= utility_b:
                choice = "A"
                explanation = f"{summary_a}\n{summary_b}\nChoice: A (higher utility: {utility_a:.2f} vs {utility_b:.2f})"
            else:
                choice = "B"
                explanation = f"{summary_a}\n{summary_b}\nChoice: B (higher utility: {utility_b:.2f} vs {utility_a:.2f})"

            return choice, explanation
        except Exception as e:
            # Default to A with minimal reasoning if parsing fails
            return "A", f"Parsing failed ({e}); defaulting to A."

    # ---- helpers / parsing ----
    def _calculate_utility(self, schedule_dict: Dict) -> float:
        """
        Calculate total utility for a schedule based on weights.
        
        Args:
            schedule_dict: Dictionary with metric names as keys and values
            
        Returns:
            Total utility score (higher is better)
        """
        utility = 0.0
        
        for metric_name, weight in self.weights.items():
            # Try different variations of the metric name for matching
            metric_value = self._get_metric_value(schedule_dict, metric_name)
            utility += weight * metric_value
        
        return utility

    def _get_metric_value(self, schedule_dict: Dict, metric_name: str) -> float:
        """
        Get metric value from schedule dictionary, handling various naming conventions.
        
        Args:
            schedule_dict: Schedule metrics dictionary
            metric_name: Name of the metric to find
            
        Returns:
            Metric value (0.0 if not found)
        """
        # Direct match first
        if metric_name in schedule_dict:
            return float(schedule_dict[metric_name])
        
        # Try lowercase comparison
        metric_lower = metric_name.lower()
        for key, value in schedule_dict.items():
            if key.lower() == metric_lower:
                return float(value)
        
        # Try with underscores instead of spaces
        metric_underscore = metric_name.replace(' ', '_')
        if metric_underscore in schedule_dict:
            return float(schedule_dict[metric_underscore])
        
        # Try various formatting variations
        metric_variations = [
            metric_name.replace(' ', '_'),
            metric_name.replace('_', ' '),
            metric_name.replace('-', '_'),
            metric_name.replace('_', '-'),
        ]
        
        for variation in metric_variations:
            if variation in schedule_dict:
                return float(schedule_dict[variation])
            # Also try lowercase
            for key, value in schedule_dict.items():
                if key.lower() == variation.lower():
                    return float(value)
        
        # Not found, return 0
        return 0.0

    def _format_schedule_summary(self, schedule_dict: Dict, label: str, utility: float) -> str:
        """Format schedule for logging/debugging with utility score."""
        components = []
        
        # Show all metrics that contribute to utility
        for metric_name, weight in self.weights.items():
            if weight != 0:  # Only show metrics with non-zero weights
                value = self._get_metric_value(schedule_dict, metric_name)
                if value != 0:  # Only show non-zero values
                    contribution = weight * value
                    components.append(f"{metric_name}={value} (weight={weight}, contrib={contribution:.2f})")
        
        if components:
            details = ", ".join(components)
            return f"{label}: {details}, total utility={utility:.2f}"
        else:
            return f"{label}: total utility={utility:.2f}"

    def _parse_prompt_schedules(self, prompt: str) -> Tuple[Dict, Dict]:
        """
        Parse a prompt containing Schedule A and Schedule B data.
        Returns: (schedule_a_dict, schedule_b_dict)
        """
        schedule_a = {}
        schedule_b = {}
        
        # Find Schedule A and B lines
        lines = prompt.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Schedule A:'):
                # Parse the metrics after "Schedule A:"
                data_part = line[11:].strip()  # Remove "Schedule A:"
                schedule_a = self._parse_schedule_line(data_part)
            elif line.startswith('Schedule B:'):
                # Parse the metrics after "Schedule B:"
                data_part = line[11:].strip()  # Remove "Schedule B:"
                schedule_b = self._parse_schedule_line(data_part)
        
        return schedule_a, schedule_b

    def _parse_schedule_line(self, data_line: str) -> Dict:
        """
        Parse a line like: "conflicts=1.0, quints=0.0, evening/morning b2b=331.0, other b2b=1375.0"
        Returns: dict with metric names as keys and float values
        """
        schedule_dict = {}
        
        # Split by commas and parse each key=value pair
        parts = data_line.split(',')
        for part in parts:
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                try:
                    schedule_dict[key] = float(value)
                except ValueError:
                    # If can't convert to float, store as string
                    schedule_dict[key] = value
        
        return schedule_dict

    def _call_api(self, prompt: str, stream: Optional[bool] = None) -> str:
        """
        Legacy method retained for compatibility with code paths
        that expect a '{A}' / '{B}' string.
        """
        try:
            schedule_a, schedule_b = self._parse_prompt_schedules(prompt)
            utility_a = self._calculate_utility(schedule_a)
            utility_b = self._calculate_utility(schedule_b)
            return '{A}' if utility_a >= utility_b else '{B}'
        except Exception as e:
            return '{A}'

    def _parse_pairwise(self, response: str) -> Tuple[str, str]:
        """
        Parse the response from _call_api to extract choice.
        Returns: (choice, reason) where reason is empty string.
        """
        response = response.strip()
        
        # Look for {A} or {B} in the response
        if '{A}' in response:
            return 'A', ''
        elif '{B}' in response:
            return 'B', ''
        else:
            # Fallback - just look for A or B
            if 'A' in response:
                return 'A', ''
            else:
                return 'B', ''

    # ------------------ Pairwise API ------------------
    def get_preference(
        self, sched_a: Dict, sched_b: Dict, stream: Optional[bool] = None
    ) -> Tuple[str, str]:
        """
        Compare two schedules and return the one with higher utility.
        Returns: (choice, explanation)
        """
        utility_a = self._calculate_utility(sched_a)
        utility_b = self._calculate_utility(sched_b)
        
        summary_a = self._format_schedule_summary(sched_a, "Schedule A", utility_a)
        summary_b = self._format_schedule_summary(sched_b, "Schedule B", utility_b)
        
        if utility_a >= utility_b:
            choice = "A"
            explanation = f"{summary_a}\n{summary_b}\nChoice: A (higher utility: {utility_a:.2f} vs {utility_b:.2f})"
        else:
            choice = "B"
            explanation = f"{summary_a}\n{summary_b}\nChoice: B (higher utility: {utility_b:.2f} vs {utility_a:.2f})"
        
        print(f"[Utility comparison] {explanation}")
        return choice, explanation

    # ------------------ Batch choose-one (up to 100) ------------------
    def choose_best_in_batch(
        self, ids: List[str], dicts: List[Dict], stream: Optional[bool] = None
    ) -> Tuple[str, str]:
        """
        Pick the schedule with the highest utility from a batch.
        Returns: (winning_id, explanation)
        """
        if not ids or not dicts:
            raise ValueError("Empty batch provided")
        
        best_id = None
        best_utility = float('-inf')
        summaries = []
        
        for schedule_id, schedule_dict in zip(ids, dicts):
            utility = self._calculate_utility(schedule_dict)
            summary = self._format_schedule_summary(schedule_dict, schedule_id, utility)
            summaries.append(summary)
            
            if utility > best_utility:
                best_utility = utility
                best_id = schedule_id
        
        explanation = f"Batch comparison:\n" + "\n".join(summaries) + f"\nWinner: {best_id} with utility {best_utility:.2f}"
        print(f"[Utility batch choice] {explanation}")
        return best_id, explanation

    # ------------------ Final top-K (compare favorites from all batches) ------------------
    def choose_top_k(
        self, ids: List[str], dicts: List[Dict], k: int, stream: Optional[bool] = None
    ) -> Tuple[List[str], str]:
        """
        Pick the K schedules with the highest utilities.
        Returns: (top_k_ids, explanation)
        """
        if k > len(ids):
            raise ValueError(f"Requested k={k} but only {len(ids)} candidates available")
        
        # Calculate utilities and sort
        candidates = []
        for schedule_id, schedule_dict in zip(ids, dicts):
            utility = self._calculate_utility(schedule_dict)
            candidates.append((utility, schedule_id, schedule_dict))
        
        # Sort by utility (descending - highest first)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Take top K
        top_k = candidates[:k]
        top_k_ids = [item[1] for item in top_k]
        
        # Create explanation
        summaries = []
        for utility, schedule_id, schedule_dict in candidates:
            summary = self._format_schedule_summary(schedule_dict, schedule_id, utility)
            summaries.append(summary)
        
        explanation = f"Top-K selection (k={k}):\n" + "\n".join(summaries) + f"\nSelected: {top_k_ids}"
        print(f"[Utility top-K choice] {explanation}")
        return top_k_ids, explanation


# Example usage:
if __name__ == "__main__":
    # Example weights configuration
    weights = {
        "conflicts": -1000,  # Very bad - avoid conflicts
        "quints": 0,
        "quads": 0,
        "four in five slots": 0,
        "triple in 24h (no gaps)": 0,
        "triple in same day (no gaps)": 0,
        "three in four slots": 0,
        "evening/morning b2b": -1,  # Bad - minimize back-to-back
        "other b2b": -1,  # Bad - minimize back-to-back
        "two in three slots": 0,
    }
    
    # Create client with custom weights
    client = SimpleUtilityPreferenceClient(weights=weights)
    
    # Example schedules
    schedule_1 = {
        "conflicts": 0,
        "evening/morning b2b": 10,
        "other b2b": 20,
    }
    
    schedule_2 = {
        "conflicts": 1,
        "evening/morning b2b": 5,
        "other b2b": 10,
    }
    
    # Compare schedules
    choice, explanation = client.get_preference(schedule_1, schedule_2)
    print(f"Chosen: {choice}")
    print(f"Reason: {explanation}")