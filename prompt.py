
class PromptTemplate:
    """
    A template class for generating prompts for LLM-based schedule preference collection.
    Optionally maintains history of previous decisions and reasoning.
    """
    
    def __init__(self, reasoning_history: bool = False, utility_prompt: str = '' , metric_columns:list = []):
        """
        Initialize the PromptTemplate.
        
        Args:
            reasoning_history: If True, store and append previous schedules, decisions, 
                             and reasoning to prompts
            utility_prompt: Optional additional prompt text about utility preferences
        """
        self.reasoning_history = reasoning_history
        self.utility_prompt = utility_prompt
        self.metric_columns = metric_columns
        # Base prompt explaining the task and definitions
        self.base_prompt = (
            "You are an expert university registrar deciding between final exam schedules. "
            "Pick the best schedule. "
            "Definitions: conflicts is the number of students with exams scheduled at the same time; "
            "quints is 5 exams in a row; quads is 4 in a row; triples is 3 in a row; "
            "b2b is back-to-back exams; two in three is 2 exams within 24 hours; "
            "three in 24 is 3 exams within 24 hours."
        )
        
        # History storage if enabled
        if self.reasoning_history:
            self.history = []
    
    def format(self, schedule_a: dict, schedule_b: dict) -> str:
        """
        Format the complete prompt with schedules and optional history/utility.
        
        Args:
            schedule_a: Dictionary representation of first schedule
            schedule_b: Dictionary representation of second schedule
            
        Returns:
            Complete formatted prompt string
        """
        prompt_parts = []
        
        # Add history if enabled and exists
        if self.reasoning_history and hasattr(self, 'history') and self.history:
            prompt_parts.append("=== PREVIOUS DECISIONS ===\n")
            for i, entry in enumerate(self.history, 1):
                prompt_parts.append(f"\nDecision {i}:  ")
                prompt_parts.append(f"Schedule A: {entry['schedule_a']}")
                prompt_parts.append(f"Schedule B: {entry['schedule_b']}")
                prompt_parts.append(f"Choice: {entry['choice']}")
                if 'reasoning' in entry:
                    prompt_parts.append(f"  Reasoning: {entry['reasoning']}")
            prompt_parts.append("\n=== CURRENT DECISION ===\n")
        
        # Add base prompt
        prompt_parts.append(self.base_prompt)
        
        # Add utility prompt if provided
        if self.utility_prompt:
            prompt_parts.append(" " + self.utility_prompt)
        
        # Add current schedules for comparison
        prompt_parts.append("\n\nPlease compare these two schedules:")
        prompt_parts.append(f"\n\nSchedule A:\n{self._format_schedule(schedule_a)}")
        prompt_parts.append(f"\n\nSchedule B:\n{self._format_schedule(schedule_b)}")
        
        # Add decision request
        prompt_parts.append("\n\nWhich schedule is better? Please provide your choice (A or B) and reasoning. write your decision after you finish reasoning in this format: {A} or {B}. If you pick schedule A write {A} if you pick schedule B write {B}")
        
        return "".join(prompt_parts)
    
    def add_to_history(self, schedule_a: dict, schedule_b: dict, choice: str, reasoning: str = None):
        """
        Add a decision to the history if history tracking is enabled.
        
        Args:
            schedule_a: Dictionary representation of first schedule
            schedule_b: Dictionary representation of second schedule
            choice: The chosen schedule ('A' or 'B')
            reasoning: Optional reasoning for the choice
        """
        if self.reasoning_history and hasattr(self, 'history'):
            entry = {
                'schedule_a': schedule_a,
                'schedule_b': schedule_b,
                'choice': choice
            }
            if reasoning:
                entry['reasoning'] = reasoning
            self.history.append(entry)
    
    def clear_history(self):
        """Clear the decision history."""
        if self.reasoning_history and hasattr(self, 'history'):
            self.history = []
    
    def _format_schedule(self, schedule: dict) -> str:
        """
        Format a schedule dictionary for display in the prompt.
        
        Args:
            schedule: Dictionary representation of a schedule
            
        Returns:
            Formatted string representation of the schedule
        """
        lines = []
        #print('schedule !, ' , schedule )
        for key, value in schedule.items():
            if key in self.metric_columns : 
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)
    
    def get_history_length(self) -> int:
        """Get the number of decisions in history."""
        if self.reasoning_history and hasattr(self, 'history'):
            return len(self.history)
        return 0

"""
# Example usage:
if __name__ == "__main__":
    # Initialize without history
    template_no_history = PromptTemplate(reasoning_history=False)
    
    # Initialize with history and utility prompt
    template_with_history = PromptTemplate(
        reasoning_history=True,
        utility_prompt="Prioritize minimizing conflicts and consecutive exams."
    )
    
    # Example schedules
    schedule_a = {
        'conflicts': 5,
        'quints': 0,
        'quads': 1,
        'triples': 3,
        'b2b': 15,
        'two_in_three': 20,
        'three_in_24': 8
    }
    
    schedule_b = {
        'conflicts': 3,
        'quints': 1,
        'quads': 0,
        'triples': 2,
        'b2b': 18,
        'two_in_three': 22,
        'three_in_24': 10
    }
    
    # Generate prompt without history
    prompt = template_no_history.format(schedule_a, schedule_b)
    print("Prompt without history:")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    # Generate prompt with history
    # First add some history
    template_with_history.add_to_history(
        {'conflicts': 10, 'b2b': 20},
        {'conflicts': 8, 'b2b': 25},
        'B',
        'Schedule B has fewer conflicts which is more important than b2b exams.'
    )
    
    prompt_with_history = template_with_history.format(schedule_a, schedule_b)
    print("Prompt with history:")
    print(prompt_with_history)
"""