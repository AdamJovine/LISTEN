from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import re
import random


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


class ComparisonPromptTemplate(PromptTemplateInterface):
    """
    Base class for prompt templates that compare multiple items.
    This is still abstract - subclasses must implement format() and get_base_prompt().
    """
    
    MAX_HISTORY = 5  # Default max history items
    
    def __init__(self, reasoning: bool = True, reasoning_history: bool = False, **kwargs):
        """
        Initialize comparison prompt template.
        
        Args:
            reasoning: Include reasoning in output
            reasoning_history: Include history in prompt
            **kwargs: Additional parameters
        """
        super().__init__(reasoning=reasoning)
        self.reasoning_history = reasoning_history
    
    @abstractmethod
    def format(self, *args, **kwargs) -> str:
        """
        Format the main prompt. Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def get_base_prompt(self) -> str:
        """
        Get the base prompt. Must be implemented by subclasses.
        """
        pass
    
    
    def format_comparison(self, items: List[Dict[str, Any]], 
                         item_formatter: callable = None) -> str:
        """
        Helper method to format multiple items for comparison.
        Can be used by subclasses in their format() implementation.
        
        Args:
            items: List of items to compare
            item_formatter: Optional custom formatter for each item
            
        Returns:
            Formatted comparison string
        """
        parts = []
        
        # Add history if enabled
        if self.reasoning_history and self.history:
            parts.append(self._format_history())
        
        # Add base prompt
        parts.append(self.get_base_prompt())
        
        # Format each item
        parts.append("\n=== ITEMS TO COMPARE ===\n\n")
        for i, item in enumerate(items):
            label = chr(65 + i)  # A, B, C, etc.
            parts.append(f"Option {label}:\n")
            if item_formatter:
                parts.append(item_formatter(item))
            else:
                parts.append(self._default_item_formatter(item))
            parts.append("\n\n")
        
        # Add instructions
        parts.append(self._format_instructions(len(items)))
        
        return "".join(parts)
    
    def _format_history(self) -> str:
        """Format history entries for few-shot learning."""
        parts = ["=== PREVIOUS DECISIONS (most recent first) ===\n"]
        for i, entry in enumerate(reversed(self.history[-self.MAX_HISTORY:]), 1):
            parts.append(f"Decision {i}:\n")
            parts.append(f"  Choice: {entry['choice']}")
            if "reasoning" in entry and entry["reasoning"]:
                parts.append(f"\n  Rationale: {entry['reasoning']}")
            parts.append("\n\n")
        parts.append("=== CURRENT DECISION ===\n\n")
        return "".join(parts)
    
    def _format_instructions(self, num_options: int) -> str:
        """Format the instructions based on reasoning flag."""
        valid_choices = [chr(65 + i) for i in range(num_options)]
        
        if self.reasoning:
            parts = [
                "\nProvide a brief explanation of your reasoning. "
                "Then on the last line, output only a final tag in this exact format:\n"
            ]
        else:
            parts = [
                "\nDo not explain. Output only one line in this exact format:\n"
            ]
        
        for i, choice in enumerate(valid_choices):
            if i == 0:
                parts.append(f"FINAL: {choice}\n")
            else:
                parts.append(f"or\nFINAL: {choice}\n")
        
        return "".join(parts)
    
    def _default_item_formatter(self, item: Dict[str, Any]) -> str:
        """Default formatter for items."""
        lines = []
        for key, value in item.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
    
    def parse_response(self, response: str) -> str:
        """Parse response to extract choice."""
        # Determine valid choices based on history or assume binary
        if self.history and "data" in self.history[-1]:
            num_options = len(self.history[-1]["data"])
        else:
            num_options = 2  # Default to binary choice
        
        valid_choices = [chr(65 + i) for i in range(num_options)]
        return self._parse_final_tag(response, valid_choices)


# Adapter for the existing PromptTemplate class
class SchedulePromptTemplateAdapter(ComparisonPromptTemplate):
    """
    Adapter to make the existing PromptTemplate conform to the interface.
    """
    
    def __init__(self, reasoning: bool = True, reasoning_history: bool = False,
                 utility_prompt: str = "", metric_columns: list = None):
        super().__init__(reasoning=reasoning, reasoning_history=reasoning_history)
        self.utility_prompt = utility_prompt.strip()
        self.metric_columns = list(metric_columns or [])
        
    def get_base_prompt(self) -> str:
        return (
            "You are an expert university registrar choosing the better final-exam schedule.\n"
            "Use these definitions (lower is better unless stated otherwise):\n"
            "• conflicts: students with overlapping exams (must be minimized)\n"
            "• quints: 5 exams in a row\n"
            "• quads: 4 in a row\n"
            "• four in five slots: 4 exams within 5 slots\n"
            "• triple in 24h (no gaps): 3 exams within 24 hours (strictly consecutive blocks)\n"
            "• triple in same day (no gaps): 3 exams within the same calendar day consecutively\n"
            "• three in four slots: 3 exams within 4 slots\n"
            "• evening/morning b2b: back-to-back from an evening into next-morning slot\n"
            "• other b2b: any other back-to-back pair\n"
            "• two in three slots: 2 exams within 3 slots\n"
        )
    
    def format(self, schedules: list) -> str:
        """Format schedules for comparison."""
        if not schedules:
            raise ValueError("At least one schedule must be provided")
        
        def schedule_formatter(schedule):
            keys = self.metric_columns or list(schedule.keys())
            lines = []
            for k in keys:
                if k in schedule:
                    lines.append(f"  {k}: {schedule[k]}")
            return "\n".join(lines)
        
        # Build the comparison prompt
        prompt = self.format_comparison(schedules, item_formatter=schedule_formatter)
        
        # Add utility prompt if provided
        if self.utility_prompt:
            parts = prompt.split("\n=== ITEMS TO COMPARE ===")
            prompt = parts[0] + f"\nPolicy guidance: {self.utility_prompt}\n" + "\n=== ITEMS TO COMPARE ===" + parts[1]
        
        return prompt
    
    def add_to_history(self, schedules: list, choice: str, reasoning: str = None):
        """Override to match original interface."""
        super().add_to_history({"schedules": schedules}, choice, reasoning)



# Adapter for the BeautyPreferencePromptTemplate
class BeautyPromptTemplateAdapter(ComparisonPromptTemplate):
    """
    Adapter to make the BeautyPreferencePromptTemplate conform to the interface.
    """
    
    def __init__(self, train_df, test_pairs_file: str = 'beauty_split_test_pairs.json',
                 include_product_features: bool = True, include_review_text: bool = True,
                 reasoning: bool = True, max_review_text_length: int = 200, temperature :int = 0 , max_new_tokens :int = 1024 , seed :int = 42 ):
        super().__init__(reasoning=reasoning, reasoning_history=True, temperature = temperature , max_new_tokens = max_new_tokens, seed = seed)
        self.train_df = train_df
        self.include_product_features = include_product_features
        self.include_review_text = include_review_text
        self.max_review_text_length = max_review_text_length
        
        # Load test pairs
        import json
        with open(test_pairs_file, 'r') as f:
            self.test_pairs = json.load(f)
        
        # Create user lookup
        self.user_reviews = {}
        for user_id, user_data in train_df.groupby('user_id'):
            self.user_reviews[user_id] = user_data.sort_values('timestamp').to_dict('records')
    
    def get_base_prompt(self) -> str:
        return (
            "You are an expert at predicting user preferences for beauty products based on their review history.\n"
            "You will analyze a user's past product reviews to understand their preferences, then predict which "
            "of two new products they would prefer.\n\n"
            "Consider factors such as:\n"
            "• Product categories and types they've enjoyed\n"
            "• Brands they trust or dislike\n"
            "• Specific features they value (scent, texture, effectiveness)\n"
            "• Common complaints or praise in their reviews\n"
            "• Price sensitivity\n"
            "• Skin/hair type compatibility mentioned\n"
        )
    def format(self, user_id: str, product_a: Dict = None, product_b: Dict = None,
            pair_index: Optional[int] = None) -> str:
        """Format beauty product comparison with randomized product order."""
        
        # Get products from test pairs if index provided
        if pair_index is not None:
            pair = self.test_pairs[pair_index]
            if pair['user_id'] != user_id:
                raise ValueError(f"User ID mismatch: {user_id} != {pair['user_id']}")
            product_a = pair['positive_review']
            product_b = pair['negative_review']
        
        if not product_a or not product_b:
            raise ValueError("Both product_a and product_b must be provided")
        
        # Build user history section
        history_section = self._format_user_history(user_id)
        
        # Format products for comparison (shuffle order)
        products = [product_a, product_b]
        random.shuffle(products)
        comparison = self.format_comparison(products, item_formatter=self._format_product)
        
        # Combine sections
        parts = comparison.split("\n=== ITEMS TO COMPARE ===")
        final_prompt = parts[0] + history_section + "\n=== PRODUCTS TO COMPARE ===" + parts[1]
        
        return final_prompt
    
    def _format_user_history(self, user_id: str) -> str:
        """Format user's review history."""
        parts = ["\n=== USER REVIEW HISTORY ===\n"]
        parts.append(f"User ID: {user_id}\n\n")
        
        if user_id in self.user_reviews:
            reviews = self.user_reviews[user_id][-10:]  # Last 10 reviews
            parts.append(f"Showing last {len(reviews)} reviews (most recent first):\n\n")
            
            for i, review in enumerate(reversed(reviews), 1):
                parts.append(f"Review {i}:\n")
                parts.append(f"  Product: {review.get('product_title', 'Unknown')}\n")
                parts.append(f"  Rating: {'⭐' * int(review.get('review_rating', 0))} ({review.get('review_rating')}/5)\n")
                parts.append(f"  Product Features : {review.get('product_features', 'Unknown')}")
                parts.append(f"  Price : {review.get('product_price', 'Unknown')}")
                
                if self.include_review_text and review.get('review_text'):
                    text = str(review['review_text'])[:self.max_review_text_length]
                    parts.append(f"  Review: \"{text}\"\n")
                
                parts.append("\n")
        else:
            parts.append("No review history available for this user.\n\n")
        
        return "".join(parts)
    
    def _format_product(self, product: Dict) -> str:
        """Format a single product."""
        lines = []
        
        #if 'product_title' in product:
        lines.append(f"  Product: {product.get('product_title', 'Unknown')}")
        #elif 'title' in product:
        #    lines.append(f"  Title: {product.get('title', 'Unknown')}")
        lines.append(f"  Product Features : {product.get('product_features', 'Unknown')}")
        lines.append(f"  Price : {product.get('product_price', 'Unknown')}")  
        #if 'rating' in product:
         #   rating = product.get('rating', 0)
         #   lines.append(f"  User Rating: {'⭐' * int(rating)} ({rating}/5)")
        
        #if self.include_review_text and 'review_text' in product:
        #    text = str(product.get('review_text', ''))[:self.max_review_text_length]
        #    if text:
        #        lines.append(f"  Review Text: \"{text}\"")
        
        return "\n".join(lines)
    
    def format_from_test_pair(self, pair_index: int) -> Tuple[str, Dict]:
        """
        Format prompt directly from a test pair index.
        
        Args:
            pair_index: Index in the test_pairs list
            
        Returns:
            Tuple of (prompt, pair_info)
        """
        if pair_index >= len(self.test_pairs):
            raise ValueError(f"Invalid pair index: {pair_index} (max: {len(self.test_pairs)-1})")
        
        pair = self.test_pairs[pair_index]
        user_id = pair['user_id']
        
        prompt = self.format(
            user_id=user_id,
            product_a=pair['positive_review'],
            product_b=pair['negative_review']
        )
        
        return prompt, pair
    
    def evaluate_pair(self, pair_index: int) -> Dict:
        """
        Get evaluation info for a test pair.
        
        Args:
            pair_index: Index in test_pairs
            
        Returns:
            Dictionary with ground truth and pair info
        """
        pair = self.test_pairs[pair_index]
        
        # Ground truth: A (positive) should be preferred over B (negative)
        return {
            'user_id': pair['user_id'],
            'ground_truth': 'A',  # Positive product is always A in our format
            'product_a_rating': pair['positive_review']['rating'],
            'product_b_rating': pair['negative_review']['rating'],
            'product_a_title': pair['positive_review'].get('product_title', 'Unknown'),
            'product_b_title': pair['negative_review'].get('product_title', 'Unknown')
        }
    
    def parse_response(self, response: str) -> str:
        """Parse beauty preference response."""
        return self._parse_final_tag(response, ['A', 'B'])
# Adapter for Industrial & Scientific Products
class IndustrialPromptTemplateAdapter(BeautyPromptTemplateAdapter):
    """
    Adapter for predicting user preferences for industrial and scientific products.
    Inherits from BeautyPromptTemplateAdapter and only modifies the base prompt.
    """
    
    def get_base_prompt(self) -> str:
        return (
            "You are an expert at predicting user preferences for industrial and scientific products based on their review history.\n"
            "You will analyze a user's past product reviews to understand their preferences, then predict which "
            "of two new products they would prefer.\n\n"
            "Consider factors such as:\n"
            "• Product quality and durability expectations\n"
            "• Precision and accuracy requirements\n"
            "• Material preferences (steel, plastic, copper, etc.)\n"
            "• Brand reliability and manufacturer reputation\n"
            "• Price-to-performance ratio expectations\n"
            "• Specific technical specifications they value\n"
            "• Professional vs. hobbyist use cases\n"
            "• Safety and compliance standards importance\n"
            "• Ease of use vs. professional features balance\n"
            "• Common complaints about build quality or functionality\n"
        )