import os
from typing import Dict, Any, List
import numpy as np
import torch
import wandb
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class EvaluationAnalyst:
    """Evaluation analyst agent for the LAM system."""
    
    def __init__(self, config: Dict[str, Any], model: Any):
        """Initialize EvaluationAnalyst."""
        self.config = config
        self.model = model
        self.metrics = {}
        self.baseline_metrics = {}
        
        # Initialize wandb only if not in test mode
        if not config.get('test_mode', False):
            try:
                wandb.init(project="large-action-model")
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
        
    def calculate_task_success_rate(self, results: List[Dict[str, bool]], ground_truth: List[Dict[str, bool]] = None) -> float:
        """Calculate Task Success Rate (TSR)."""
        if not results:
            return 0.0
            
        successes = sum(1 for result in results if result.get('success', False))
        return successes / len(results)
        
    def evaluate_model(self, eval_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()
        metrics = {
            'accuracy': 0.0,
            'loss': 0.0
        }

        try:
            with torch.no_grad():
                # Process each batch
                input_ids = eval_data['input_ids'].to(self.model.device)
                outputs = self.model(input_ids)

                # Calculate metrics
                if 'action_type_logits' in outputs and 'action_type' in eval_data:
                    action_type = eval_data['action_type'].to(self.model.device)
                    pred_action = torch.argmax(outputs['action_type_logits'], dim=-1)
                    metrics['accuracy'] = (pred_action == action_type).float().mean().item()
                    metrics['loss'] = torch.nn.functional.cross_entropy(outputs['action_type_logits'], action_type).item()

            # Only log to wandb if not in test mode and wandb is available
            if not self.config.get('test_mode', False):
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(metrics)
                except (ImportError, wandb.Error):
                    pass

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            metrics['error'] = str(e)

        return metrics
        
    def preprocess_text(self, text: str) -> List[int]:
        """Preprocess text input."""
        # Mock implementation for testing
        return [1] * 10  # Return a sequence of 10 ones
        
    def _calculate_step_accuracy(self, predicted_steps: List[Dict], actual_steps: List[Dict]) -> float:
        """Calculate accuracy of predicted steps against actual steps."""
        if not predicted_steps or not actual_steps:
            return 0.0
            
        total_steps = len(actual_steps)
        correct_steps = sum(1 for pred, actual in zip(predicted_steps, actual_steps)
                          if self._compare_steps(pred, actual))
        return correct_steps / total_steps
        
    def _compare_steps(self, predicted_step: Dict, actual_step: Dict) -> bool:
        """Compare if predicted step matches actual step."""
        return (predicted_step.get('action_type') == actual_step.get('action_type') and
                all(abs(p - a) < 0.1 for p, a in zip(
                    predicted_step.get('continuous_action', []),
                    actual_step.get('continuous_action', [])
                )))

if __name__ == "__main__":
    # Test evaluation analyst functionality
    evaluator = EvaluationAnalyst({'test_mode': True}, None)
    # Add test cases here
