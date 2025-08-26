"""
Usage examples for the refactored PreferenceLLMEvaluator with different prompt template adapters.
"""

import pandas as pd
import json
import time
from datetime import datetime
from pathlib import Path

# Import the evaluator and prompt templates
from prompt import BeautyPromptTemplateAdapter, IndustrialPromptTemplateAdapter
import pandas as pd
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from tqdm import tqdm
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import the LLM client
from remoteOss import get_local_client


class PreferenceLLMEvaluator:
    """
    Evaluates LLM performance on preference prediction with batch processing support.
    Works with any prompt template adapter that follows the expected interface.
    """
    
    def __init__(
        self,
        prompt_template_adapter: Any,  # Accept any prompt template adapter
        model_id: str = None,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        seed: int = 42,
        batch_size: int = 32,
        max_concurrent_batches: int = 2
    ):
        """
        Initialize the evaluator with batch processing support.
        
        Args:
            prompt_template_adapter: Instance of any prompt template adapter class
                                    (e.g., BeautyPromptTemplateAdapter, IndustrialPromptTemplateAdapter)
            model_id: Model ID for the LLM (None uses default)
            temperature: LLM temperature setting
            max_new_tokens: Max tokens for LLM response
            seed: Random seed for reproducibility
            batch_size: Number of queries to process in a single batch
            max_concurrent_batches: Maximum number of concurrent batch requests
        """
        # Use the provided prompt template adapter
        self.prompt_template = prompt_template_adapter
        
        # Initialize LLM client
        if model_id:
            self.llm_client = get_local_client(model_id=model_id)
        else:
            self.llm_client = get_local_client()
        
        # LLM parameters
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        
        # Batch processing parameters
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        
        # Results storage
        self.results = []
        self.errors = []
    
    def prepare_batch_data(self, pair_indices: List[int]) -> Tuple[List[str], List[Dict]]:
        """
        Prepare batch data for multiple test pairs.
        
        Args:
            pair_indices: List of test pair indices
            
        Returns:
            Tuple of (prompts, pair_infos) for the batch
        """
        prompts = []
        pair_infos = []
        
        for idx in pair_indices:
            try:
                prompt, pair_info = self.prompt_template.format_from_test_pair(idx)
                eval_info = self.prompt_template.evaluate_pair(idx)
                
                # Combine pair_info with eval_info for complete context
                complete_info = {
                    'pair_index': idx,
                    'prompt': prompt,
                    **pair_info,
                    **eval_info
                }
                
                prompts.append(prompt)
                pair_infos.append(complete_info)
            except Exception as e:
                self.errors.append({
                    'pair_index': idx,
                    'error': f"Error preparing pair: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                })
        
        return prompts, pair_infos
    
    def process_batch_responses(
        self, 
        responses: List[Tuple[str, str]], 
        pair_infos: List[Dict],
        inference_time: float
    ) -> List[Dict]:
        """
        Process batch responses from the LLM.
        
        Args:
            responses: List of (choice, raw_response) tuples from LLM
            pair_infos: List of pair information dictionaries
            inference_time: Total inference time for the batch
            
        Returns:
            List of result dictionaries
        """
        batch_results = []
        avg_inference_time = inference_time / len(responses) if responses else 0
        
        for (choice, raw_response), pair_info in zip(responses, pair_infos):
            try:
                # Check if prediction is correct
                is_correct = (choice == pair_info['ground_truth'])
                
                # Build result dict with flexible field names
                result = {
                    'pair_index': pair_info['pair_index'],
                    'user_id': pair_info.get('user_id'),
                    'predicted': choice,
                    'ground_truth': pair_info['ground_truth'],
                    'correct': is_correct,
                    'raw_response': raw_response,
                    'inference_time': avg_inference_time,
                    'prompt_length': len(pair_info['prompt']),
                    'response_length': len(raw_response)
                }
                
                # Add product-specific fields dynamically
                # Handle different field names from different adapters
                for key, value in pair_info.items():
                    if key not in result and key not in ['prompt', 'ground_truth']:
                        result[key] = value
                
                batch_results.append(result)
                
            except Exception as e:
                self.errors.append({
                    'pair_index': pair_info.get('pair_index', -1),
                    'error': f"Error processing response: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                })
        
        return batch_results
    
    def call_batch_oracle(self, prompts: List[str], pair_infos: List[Dict]) -> List[Tuple[str, str]]:
        """
        Call the LLM with a batch of prompts.
        
        Args:
            prompts: List of prompts to process
            pair_infos: List of pair information for context
            
        Returns:
            List of (choice, raw_response) tuples
        """
        # Check if the LLM client supports batch processing
        if hasattr(self.llm_client, 'call_batch_oracle'):
            # Use batch method if available
            schedules_a = []
            schedules_b = []
            
            for info in pair_infos:
                # Handle different field names from different adapters
                sched_a = {}
                sched_b = {}
                
                # Look for product/item fields with various naming conventions
                for field_prefix in ['product', 'item', 'option']:
                    a_title_key = f'{field_prefix}_a_title'
                    b_title_key = f'{field_prefix}_b_title'
                    a_name_key = f'{field_prefix}_a_name'
                    b_name_key = f'{field_prefix}_b_name'
                    a_key = f'{field_prefix}_a'
                    b_key = f'{field_prefix}_b'
                    
                    # Try different key combinations
                    if a_title_key in info:
                        sched_a['product'] = info[a_title_key]
                    elif a_name_key in info:
                        sched_a['product'] = info[a_name_key]
                    elif a_key in info:
                        sched_a['product'] = info[a_key]
                    
                    if b_title_key in info:
                        sched_b['product'] = info[b_title_key]
                    elif b_name_key in info:
                        sched_b['product'] = info[b_name_key]
                    elif b_key in info:
                        sched_b['product'] = info[b_key]
                
                # Look for rating/score fields
                for field in ['rating', 'score', 'value']:
                    a_field_key = f'{field}_a'
                    b_field_key = f'{field}_b'
                    a_alt_key = f'product_a_{field}'
                    b_alt_key = f'product_b_{field}'
                    
                    if a_field_key in info:
                        sched_a['rating'] = info[a_field_key]
                    elif a_alt_key in info:
                        sched_a['rating'] = info[a_alt_key]
                    
                    if b_field_key in info:
                        sched_b['rating'] = info[b_field_key]
                    elif b_alt_key in info:
                        sched_b['rating'] = info[b_alt_key]
                
                schedules_a.append(sched_a)
                schedules_b.append(sched_b)
            
            return self.llm_client.call_batch_oracle(
                prompts=prompts,
                schedules_a=schedules_a,
                schedules_b=schedules_b,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                seed=self.seed,
                stop=["===", "---", "\n\n\n"]
            )
        else:
            # Fallback to sequential processing if batch method not available
            responses = []
            for prompt, info in zip(prompts, pair_infos):
                # Similar field extraction logic as above
                sched_a = {}
                sched_b = {}
                
                for field_prefix in ['product', 'item', 'option']:
                    a_title_key = f'{field_prefix}_a_title'
                    b_title_key = f'{field_prefix}_b_title'
                    if a_title_key in info:
                        sched_a['product'] = info[a_title_key]
                        sched_b['product'] = info[b_title_key]
                        break
                
                for field in ['rating', 'score', 'value']:
                    a_field_key = f'{field}_a'
                    b_field_key = f'{field}_b'
                    if a_field_key in info:
                        sched_a['rating'] = info[a_field_key]
                        sched_b['rating'] = info[b_field_key]
                        break
                
                choice, raw_response = self.llm_client.call_oracle(
                    prompt=prompt,
                    sched_a=sched_a,
                    sched_b=sched_b,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    seed=self.seed,
                    stop=["===", "---", "\n\n\n"]
                )
                responses.append((choice, raw_response))
            
            return responses
    
    def evaluate_batch_concurrent(
        self,
        pair_indices: List[int] = None,
        sample_size: int = None,
        verbose: bool = True,
        save_intermediate: bool = True,
        output_dir: str = 'evaluation_results'
    ) -> Dict:
        """
        Evaluate multiple test pairs using batch processing with concurrency.
        
        Args:
            pair_indices: Specific indices to evaluate
            sample_size: Random sample size (if pair_indices not provided)
            verbose: Print progress
            save_intermediate: Save results periodically
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Determine which pairs to evaluate
        if pair_indices is None:
            total_pairs = len(self.prompt_template.test_pairs)
            if sample_size:
                import random
                pair_indices = random.sample(range(total_pairs), min(sample_size, total_pairs))
            else:
                pair_indices = list(range(total_pairs))
        else:
            pair_indices = list(pair_indices)
        
        # Create output directory
        if save_intermediate:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Split into batches
        batches = [pair_indices[i:i + self.batch_size] 
                  for i in range(0, len(pair_indices), self.batch_size)]
        
        if verbose:
            print(f"Processing {len(pair_indices)} pairs in {len(batches)} batches of size {self.batch_size}")
        
        # Process batches with concurrency control
        with ThreadPoolExecutor(max_workers=self.max_concurrent_batches) as executor:
            futures = []
            
            # Submit batches for processing
            for batch_idx, batch_indices in enumerate(batches):
                future = executor.submit(self._process_single_batch, batch_indices, batch_idx)
                futures.append(future)
            
            # Process completed batches
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Processing batches", disable=not verbose):
                try:
                    batch_results = future.result()
                    self.results.extend(batch_results)
                    
                    # Save intermediate results
                    if save_intermediate and len(self.results) % (self.batch_size * 2) == 0:
                        self._save_intermediate_results(output_dir)
                        
                except Exception as e:
                    print(f"Batch processing error: {e}")
        
        # Final save
        if save_intermediate:
            self._save_intermediate_results(output_dir)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        if verbose:
            self.print_metrics(metrics)
        
        return metrics
    
    def _process_single_batch(self, batch_indices: List[int], batch_idx: int) -> List[Dict]:
        """
        Process a single batch of test pairs.
        
        Args:
            batch_indices: Indices for this batch
            batch_idx: Batch number (for logging)
            
        Returns:
            List of results for this batch
        """
        try:
            # Prepare batch data
            prompts, pair_infos = self.prepare_batch_data(batch_indices)
            
            if not prompts:
                return []
            
            # Call LLM with batch
            start_time = time.time()
            responses = self.call_batch_oracle(prompts, pair_infos)
            inference_time = time.time() - start_time
            
            # Process responses
            batch_results = self.process_batch_responses(responses, pair_infos, inference_time)
            
            return batch_results
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            self.errors.append({
                'batch_idx': batch_idx,
                'batch_indices': batch_indices,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return []
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate evaluation metrics from results.
        
        Returns:
            Dictionary with various metrics
        """
        if not self.results:
            return {'error': 'No results to evaluate'}
        
        df = pd.DataFrame(self.results)
        
        # Basic accuracy
        accuracy = df['correct'].mean()
        
        # Per-user accuracy (if user_id exists)
        user_metrics = {}
        if 'user_id' in df.columns and df['user_id'].notna().any():
            user_accuracy = df.groupby('user_id')['correct'].agg(['mean', 'count'])
            user_metrics = {
                'unique_users': df['user_id'].nunique(),
                'avg_accuracy_per_user': user_accuracy['mean'].mean(),
                'std_accuracy_per_user': user_accuracy['mean'].std(),
                'min_user_accuracy': user_accuracy['mean'].min(),
                'max_user_accuracy': user_accuracy['mean'].max()
            }
        
        # Try to find rating difference columns
        rating_metrics = {}
        rating_cols = [col for col in df.columns if 'rating' in col.lower() or 'score' in col.lower()]
        
        if len(rating_cols) >= 2:
            # Assume first two are for options A and B
            col_a = sorted([c for c in rating_cols if 'a' in c.lower()])[0] if any('a' in c.lower() for c in rating_cols) else rating_cols[0]
            col_b = sorted([c for c in rating_cols if 'b' in c.lower()])[0] if any('b' in c.lower() for c in rating_cols) else rating_cols[1]
            
            if col_a in df.columns and col_b in df.columns:
                df['rating_diff'] = df[col_a] - df[col_b]
                df['rating_diff_abs'] = df['rating_diff'].abs()
                
                accuracy_by_diff = df.groupby('rating_diff_abs')['correct'].agg(['mean', 'count'])
                
                rating_metrics = {
                    'accuracy_by_diff': accuracy_by_diff.to_dict() if not accuracy_by_diff.empty else {},
                    'avg_rating_diff': df['rating_diff'].mean(),
                }
                
                for diff in [1, 2, 3, 4]:
                    key = f'accuracy_when_diff_{diff}'
                    subset = df[df['rating_diff_abs'] == diff]
                    if len(subset) > 0:
                        rating_metrics[key] = subset['correct'].mean()
                    else:
                        rating_metrics[key] = None
        
        # Inference time statistics
        avg_inference_time = df['inference_time'].mean()
        median_inference_time = df['inference_time'].median()
        
        # Response analysis
        avg_response_length = df['response_length'].mean()
        avg_prompt_length = df['prompt_length'].mean()
        
        metrics = {
            'overall_accuracy': accuracy,
            'total_evaluated': len(df),
            'total_errors': len(self.errors),
            'batch_size_used': self.batch_size,
            'max_concurrent_batches': self.max_concurrent_batches,
            'performance_metrics': {
                'avg_inference_time_per_item': avg_inference_time,
                'median_inference_time_per_item': median_inference_time,
                'total_inference_time': df['inference_time'].sum(),
                'avg_response_length': avg_response_length,
                'avg_prompt_length': avg_prompt_length,
                'throughput_items_per_second': len(df) / df['inference_time'].sum() if df['inference_time'].sum() > 0 else 0
            },
            'prediction_distribution': {
                'predicted_A': (df['predicted'] == 'A').sum(),
                'predicted_B': (df['predicted'] == 'B').sum(),
                'ground_truth_A': (df['ground_truth'] == 'A').sum(),
                'ground_truth_B': (df['ground_truth'] == 'B').sum()
            }
        }
        
        if user_metrics:
            metrics['user_metrics'] = user_metrics
        
        if rating_metrics:
            metrics['rating_difference_metrics'] = rating_metrics
        
        # Add confusion matrix
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            
            cm = confusion_matrix(df['ground_truth'], df['predicted'], labels=['A', 'B'])
            metrics['confusion_matrix'] = {
                'true_A_pred_A': int(cm[0, 0]),
                'true_A_pred_B': int(cm[0, 1]),
                'true_B_pred_A': int(cm[1, 0]),
                'true_B_pred_B': int(cm[1, 1])
            }
            
            # Add classification report
            report = classification_report(
                df['ground_truth'], 
                df['predicted'], 
                labels=['A', 'B'],
                output_dict=True
            )
            metrics['classification_report'] = report
        except:
            pass
        
        return metrics
    
    def print_metrics(self, metrics: Dict):
        """
        Print formatted evaluation metrics.
        
        Args:
            metrics: Dictionary of metrics to print
        """
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        #print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.2%}")
        #print(f"Total Pairs Evaluated: {metrics['total_evaluated']}")
        #print(f"Total Errors: {metrics['total_errors']}")
        #print(f"Batch Size: {metrics['batch_size_used']}")
        #print(f"Max Concurrent Batches: {metrics['max_concurrent_batches']}")
        
        if 'user_metrics' in metrics:
            print("\n--- User Metrics ---")
            user_m = metrics['user_metrics']
            print(f"Unique Users: {user_m['unique_users']}")
            print(f"Avg Accuracy per User: {user_m['avg_accuracy_per_user']:.2%}")
            print(f"Std Accuracy per User: {user_m['std_accuracy_per_user']:.2%}")
            print(f"Min User Accuracy: {user_m['min_user_accuracy']:.2%}")
            print(f"Max User Accuracy: {user_m['max_user_accuracy']:.2%}")
        
        if 'rating_difference_metrics' in metrics:
            print("\n--- Accuracy by Rating Difference ---")
            diff_m = metrics['rating_difference_metrics']
            for diff in [1, 2, 3, 4]:
                key = f'accuracy_when_diff_{diff}'
                if key in diff_m and diff_m[key] is not None:
                    print(f"Rating Diff = {diff}: {diff_m[key]:.2%}")
        
        print("\n--- Performance Metrics ---")
        perf_m = metrics['performance_metrics']
        print(f"Avg Inference Time per Item: {perf_m['avg_inference_time_per_item']:.3f}s")
        print(f"Median Inference Time per Item: {perf_m['median_inference_time_per_item']:.3f}s")
        print(f"Total Inference Time: {perf_m['total_inference_time']:.1f}s")
        print(f"Throughput: {perf_m['throughput_items_per_second']:.1f} items/second")
        print(f"Avg Response Length: {perf_m['avg_response_length']:.0f} chars")
        print(f"Avg Prompt Length: {perf_m['avg_prompt_length']:.0f} chars")
        
        print("\n--- Prediction Distribution ---")
        pred_d = metrics['prediction_distribution']
        print(f"Predicted A: {pred_d['predicted_A']} ({pred_d['predicted_A']/metrics['total_evaluated']:.1%})")
        print(f"Predicted B: {pred_d['predicted_B']} ({pred_d['predicted_B']/metrics['total_evaluated']:.1%})")
        
        if 'confusion_matrix' in metrics:
            print("\n--- Confusion Matrix ---")
            cm = metrics['confusion_matrix']
            print("           Predicted")
            print("           A      B")
            print(f"True A:  {cm['true_A_pred_A']:4d}  {cm['true_A_pred_B']:4d}")
            print(f"True B:  {cm['true_B_pred_A']:4d}  {cm['true_B_pred_B']:4d}")
    
    def _save_intermediate_results(self, output_dir: str):
        """Save intermediate results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        if self.results:
            results_file = Path(output_dir) / f"results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        
        # Save errors
        if self.errors:
            errors_file = Path(output_dir) / f"errors_{timestamp}.json"
            with open(errors_file, 'w') as f:
                json.dump(self.errors, f, indent=2)
        
        # Save metrics
        if self.results:
            metrics = self.calculate_metrics()
            metrics_file = Path(output_dir) / f"metrics_{timestamp}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
    
    def analyze_failures(self, n_examples: int = 5) -> Dict:
        """
        Analyze failure cases to understand where the model fails.
        
        Args:
            n_examples: Number of example failures to show
            
        Returns:
            Dictionary with failure analysis
        """
        if not self.results:
            return {'error': 'No results to analyze'}
        
        df = pd.DataFrame(self.results)
        failures = df[~df['correct']]
        
        if failures.empty:
            return {'message': 'No failures found!'}
        
        # Find rating/score columns dynamically
        rating_col = None
        for col in df.columns:
            if 'rating' in col.lower() or 'score' in col.lower():
                rating_col = col
                break
        
        analysis = {
            'total_failures': len(failures),
            'failure_rate': len(failures) / len(df),
            'example_failures': []
        }
        
        if rating_col:
            analysis['failures_by_rating'] = failures.groupby(rating_col).size().to_dict()
        
        # Get example failures
        for _, row in failures.head(n_examples).iterrows():
            example = {
                'predicted': row['predicted'],
                'should_be': row['ground_truth'],
                'response_snippet': row['raw_response'][:200] + "..." if len(row['raw_response']) > 200 else row['raw_response']
            }
            
            # Add all other fields dynamically
            for col in row.index:
                if col not in ['predicted', 'ground_truth', 'raw_response', 'correct'] and pd.notna(row[col]):
                    example[col] = row[col]
            
            analysis['example_failures'].append(example)
        
        return analysis
    

def evaluate_beauty_preferences():
    """
    Example: Evaluate using BeautyPromptTemplateAdapter
    """
    print("="*60)
    print("BEAUTY PREFERENCE EVALUATION")
    print("="*60)
    
    # Load training data
    print("\nLoading beauty training data...")
    train_df = pd.read_csv('input/beauty_split_train.csv')
    print(f"Loaded {len(train_df)} training reviews from {train_df['user_id'].nunique()} users")
    
    # Initialize the Beauty prompt template adapter
    beauty_prompt_adapter = BeautyPromptTemplateAdapter(
        train_df=train_df,
        test_pairs_file='input/beauty_split_test_pairs.json',
        include_product_features=True,
        include_review_text=True,
        reasoning=True,
        max_review_text_length=150
    )
    
    # Initialize evaluator with the beauty prompt adapter
    print("\nInitializing evaluator with BeautyPromptTemplateAdapter...")
    evaluator = PreferenceLLMEvaluator(
        prompt_template_adapter=beauty_prompt_adapter,
        model_id=None,  # Use default model
        temperature=0.0,
        max_new_tokens=2048,
        seed=42,
        batch_size=32,
        max_concurrent_batches=2
    )
    
    # Run evaluation
    print("\nStarting beauty preference evaluation...")
    metrics = evaluator.evaluate_batch_concurrent(
        sample_size=100,  # Evaluate 100 pairs
        verbose=True,
        save_intermediate=True,
        output_dir='evaluation_results_beauty'
    )
    
    # Analyze failures
    print("\nAnalyzing failures...")
    failure_analysis = evaluator.analyze_failures(n_examples=3)
    
    return metrics, failure_analysis


def evaluate_industrial_preferences():
    """
    Example: Evaluate using IndustrialPromptTemplateAdapter
    """
    print("="*60)
    print("INDUSTRIAL PREFERENCE EVALUATION")
    print("="*60)
    
    # Load training data
    print("\nLoading industrial training data...")
    train_df = pd.read_csv('input/industrial_split_train.csv')
    print(f"Loaded {len(train_df)} training records")
    
    # Initialize the Industrial prompt template adapter
    # Note: Adjust parameters based on IndustrialPromptTemplateAdapter's constructor
    industrial_prompt_adapter = IndustrialPromptTemplateAdapter(
        train_df=train_df,
        test_pairs_file='input/industrial_split_test_pairs.json',
        include_product_features=True,  # Adjust based on adapter's params
        include_review_text=True,       # Adjust based on adapter's params
        reasoning=True,
        max_review_text_length=150      # Adjust based on adapter's params
    )
    
    # Initialize evaluator with the industrial prompt adapter
    print("\nInitializing evaluator with IndustrialPromptTemplateAdapter...")
    evaluator = PreferenceLLMEvaluator(
        prompt_template_adapter=industrial_prompt_adapter,
        model_id=None,  # Use default model
        temperature=0.0,
        max_new_tokens=2048,
        seed=42,
        batch_size=16,
        max_concurrent_batches=2
    )
    
    # Run evaluation
    print("\nStarting industrial preference evaluation...")
    metrics = evaluator.evaluate_batch_concurrent(
        sample_size=32,  # Evaluate 100 pairs
        verbose=True,
        save_intermediate=True,
        output_dir='evaluation_results_industrial'
    )
    
    # Analyze failures
    print("\nAnalyzing failures...")
    failure_analysis = evaluator.analyze_failures(n_examples=3)
    
    return metrics, failure_analysis


def compare_evaluations():
    """
    Example: Compare evaluations across different domains
    """
    print("\n" + "="*60)
    print("COMPARATIVE EVALUATION")
    print("="*60)
    
    results = {}
    
    # Evaluate beauty preferences
    print("\n[1/2] Evaluating Beauty Preferences...")
    beauty_metrics, beauty_failures = evaluate_beauty_preferences()
    results['beauty'] = {
        'metrics': beauty_metrics,
        'failures': beauty_failures
    }
    
    # Evaluate industrial preferences
    print("\n[2/2] Evaluating Industrial Preferences...")
    industrial_metrics, industrial_failures = evaluate_industrial_preferences()
    results['industrial'] = {
        'metrics': industrial_metrics,
        'failures': industrial_failures
    }
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    print("\n--- Accuracy Comparison ---")
    print(f"Beauty Accuracy: {results['beauty']['metrics']['overall_accuracy']:.2%}")
    print(f"Industrial Accuracy: {results['industrial']['metrics']['overall_accuracy']:.2%}")
    
    print("\n--- Performance Comparison ---")
    beauty_perf = results['beauty']['metrics']['performance_metrics']
    industrial_perf = results['industrial']['metrics']['performance_metrics']
    
    print(f"Beauty Throughput: {beauty_perf['throughput_items_per_second']:.1f} items/sec")
    print(f"Industrial Throughput: {industrial_perf['throughput_items_per_second']:.1f} items/sec")
    
    print(f"Beauty Avg Response Length: {beauty_perf['avg_response_length']:.0f} chars")
    print(f"Industrial Avg Response Length: {industrial_perf['avg_response_length']:.0f} chars")
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f'comparison_results_{timestamp}.json'
    
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nComparison results saved to {comparison_file}")
    
    return results


def custom_prompt_adapter_example():
    """
    Example: Using a custom prompt template adapter
    """
    print("\n" + "="*60)
    print("CUSTOM PROMPT ADAPTER EXAMPLE")
    print("="*60)
    
    # Define a minimal custom prompt adapter
    class CustomPromptAdapter:
        def __init__(self, test_pairs_file):
            with open(test_pairs_file, 'r') as f:
                self.test_pairs = json.load(f)
        
        def format_from_test_pair(self, idx):
            """Format a prompt from test pair at index idx"""
            pair = self.test_pairs[idx]
            
            # Create your custom prompt
            prompt = f"""
            Which option would you prefer?
            
            Option A: {pair['option_a']['name']}
            Option B: {pair['option_b']['name']}
            
            Please choose A or B.
            """
            
            # Return prompt and pair info
            pair_info = {
                'user_id': pair.get('user_id'),
                'option_a_name': pair['option_a']['name'],
                'option_b_name': pair['option_b']['name'],
                'option_a_value': pair['option_a'].get('value', 0),
                'option_b_value': pair['option_b'].get('value', 0)
            }
            
            return prompt, pair_info
        
        def evaluate_pair(self, idx):
            """Get evaluation info for test pair at index idx"""
            pair = self.test_pairs[idx]
            return {
                'ground_truth': pair['preferred']  # 'A' or 'B'
            }
    
    # Use the custom adapter
    print("\nInitializing custom prompt adapter...")
    custom_adapter = CustomPromptAdapter('input/custom_test_pairs.json')
    
    # Initialize evaluator with custom adapter
    print("Initializing evaluator with custom adapter...")
    evaluator = PreferenceLLMEvaluator(
        prompt_template_adapter=custom_adapter,
        temperature=0.0,
        max_new_tokens=256,
        batch_size=16,
        max_concurrent_batches=1
    )
    
    # Run evaluation
    print("Running evaluation with custom adapter...")
    metrics = evaluator.evaluate_batch_concurrent(
        sample_size=50,
        verbose=True,
        output_dir='evaluation_results_custom'
    )
    
    return metrics


def main():
    """
    Main function to demonstrate different usage patterns
    """
    print("PREFERENCE LLM EVALUATOR - USAGE EXAMPLES")
    print("="*60)
    
    # Ask user which evaluation to run
    print("\nSelect evaluation type:")
    print("1. Beauty Preferences")
    print("2. Industrial Preferences")
    print("3. Compare Both")
    print("4. Custom Adapter Example")
    print("5. Run All")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        evaluate_beauty_preferences()
    elif choice == '2':
        evaluate_industrial_preferences()
    elif choice == '3':
        compare_evaluations()
    elif choice == '4':
        custom_prompt_adapter_example()
    elif choice == '5':
        print("\nRunning all evaluations...")
        evaluate_beauty_preferences()
        print("\n" + "-"*60 + "\n")
        evaluate_industrial_preferences()
        print("\n" + "-"*60 + "\n")
        compare_evaluations()
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()