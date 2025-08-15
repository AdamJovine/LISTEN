import re
from typing import Dict, Tuple, List, Optional
import pandas as pd
from oracle import Oracle

class SimpleB2BPreferenceClient(Oracle):
    """
    Client that picks schedules based on minimizing back-to-back exams.
    Always chooses the schedule with the lowest total of:
    - "evening/morning b2b" + "other b2b"
    
    Maintains the same interface as the LLM version for drop-in replacement.
    """

    def __init__(self, **kwargs):
        """
        Simplified constructor - ignores all LLM-related parameters.
        Maintains interface compatibility.
        """
        # Ignore all the LLM parameters for compatibility
        pass

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
        Parse a prompt with 'Schedule A:' and 'Schedule B:' lines, compute totals,
        and return ('A'|'B', explanation). Sampling args are ignored.
        """
        try:
            schedule_a, schedule_b = self._parse_prompt_schedules(prompt)
            total_a = self._extract_b2b_total(schedule_a)
            total_b = self._extract_b2b_total(schedule_b)

            summary_a = self._format_schedule_summary(schedule_a, "Schedule A")
            summary_b = self._format_schedule_summary(schedule_b, "Schedule B")

            if total_a <= total_b:
                choice = "A"
                explanation = f"{summary_a}\n{summary_b}\nChoice: A (fewer total back-to-backs: {total_a} vs {total_b})"
            else:
                choice = "B"
                explanation = f"{summary_a}\n{summary_b}\nChoice: B (fewer total back-to-backs: {total_b} vs {total_a})"

            return choice, explanation
        except Exception as e:
            # Default to A with minimal reasoning if parsing fails
            return "A", f"Parsing failed ({e}); defaulting to A."

    # ---- helpers / parsing ----
    def _extract_b2b_total(self, schedule_dict: Dict) -> float:
        """
        Extract total back-to-back exams from schedule dictionary.
        Looks for keys containing "evening/morning b2b" and "other b2b".
        """
        evening_morning_b2b = 0.0
        other_b2b = 0.0
        
        for key, value in schedule_dict.items():
            key_lower = key.lower()
            if "evening/morning b2b" in key_lower or "evening_morning_b2b" in key_lower:
                evening_morning_b2b = float(value)
            elif "other b2b" in key_lower or "other_b2b" in key_lower:
                other_b2b = float(value)
        
        return evening_morning_b2b + other_b2b

    def _format_schedule_summary(self, schedule_dict: Dict, label: str) -> str:
        """Format schedule for logging/debugging."""
        evening_morning = 0.0
        other = 0.0
        
        for key, value in schedule_dict.items():
            key_lower = key.lower()
            if "evening/morning b2b" in key_lower or "evening_morning_b2b" in key_lower:
                evening_morning = float(value)
            elif "other b2b" in key_lower or "other_b2b" in key_lower:
                other = float(value)
        
        total = evening_morning + other
        return f"{label}: evening/morning b2b={evening_morning}, other b2b={other}, total b2b={total}"

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
            total_a = self._extract_b2b_total(schedule_a)
            total_b = self._extract_b2b_total(schedule_b)
            return '{A}' if total_a <= total_b else '{B}'
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
        Compare two schedules and return the one with fewer back-to-back exams.
        Returns: (choice, explanation)
        """
        total_a = self._extract_b2b_total(sched_a)
        total_b = self._extract_b2b_total(sched_b)
        
        summary_a = self._format_schedule_summary(sched_a, "Schedule A")
        summary_b = self._format_schedule_summary(sched_b, "Schedule B")
        
        if total_a <= total_b:
            choice = "A"
            explanation = f"{summary_a}\n{summary_b}\nChoice: A (fewer total back-to-backs: {total_a} vs {total_b})"
        else:
            choice = "B"
            explanation = f"{summary_a}\n{summary_b}\nChoice: B (fewer total back-to-backs: {total_b} vs {total_a})"
        
        print(f"[B2B comparison] {explanation}")
        return choice, explanation

    # ------------------ Batch choose-one (up to 100) ------------------
    def choose_best_in_batch(
        self, ids: List[str], dicts: List[Dict], stream: Optional[bool] = None
    ) -> Tuple[str, str]:
        """
        Pick the schedule with the lowest back-to-back total from a batch.
        Returns: (winning_id, explanation)
        """
        if not ids or not dicts:
            raise ValueError("Empty batch provided")
        
        best_id = None
        best_total = float('inf')
        summaries = []
        
        for schedule_id, schedule_dict in zip(ids, dicts):
            total = self._extract_b2b_total(schedule_dict)
            summary = self._format_schedule_summary(schedule_dict, schedule_id)
            summaries.append(summary)
            
            if total < best_total:
                best_total = total
                best_id = schedule_id
        
        explanation = f"Batch comparison:\n" + "\n".join(summaries) + f"\nWinner: {best_id} with {best_total} total back-to-backs"
        print(f"[B2B batch choice] {explanation}")
        return best_id, explanation

    # ------------------ Final top-K (compare favorites from all batches) ------------------
    def choose_top_k(
        self, ids: List[str], dicts: List[Dict], k: int, stream: Optional[bool] = None
    ) -> Tuple[List[str], str]:
        """
        Pick the K schedules with the lowest back-to-back totals.
        Returns: (top_k_ids, explanation)
        """
        if k > len(ids):
            raise ValueError(f"Requested k={k} but only {len(ids)} candidates available")
        
        # Calculate totals and sort
        candidates = []
        for schedule_id, schedule_dict in zip(ids, dicts):
            total = self._extract_b2b_total(schedule_dict)
            candidates.append((total, schedule_id, schedule_dict))
        
        # Sort by total back-to-backs (ascending)
        candidates.sort(key=lambda x: x[0])
        
        # Take top K
        top_k = candidates[:k]
        top_k_ids = [item[1] for item in top_k]
        
        # Create explanation
        summaries = []
        for total, schedule_id, schedule_dict in candidates:
            summary = self._format_schedule_summary(schedule_dict, schedule_id)
            summaries.append(summary)
        
        explanation = f"Top-K selection (k={k}):\n" + "\n".join(summaries) + f"\nSelected: {top_k_ids}"
        print(f"[B2B top-K choice] {explanation}")
        return top_k_ids, explanation