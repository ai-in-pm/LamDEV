import os
import sys
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.model import LAMModel
from config.lam_config import LAM_CONFIG
from core.utils import setup_logging, normalize_continuous_action, denormalize_continuous_action

class LAMInference:
    def __init__(self, model_path: str, config: dict):
        """Initialize LAM inference."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])
        
        # Load model
        self.model = LAMModel(config).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def process_input(self, text: str):
        """Process input text and return model predictions."""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config['model']['max_sequence_length'],
            return_tensors='pt'
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                return_value=True
            )
            
            # Sample actions
            action_type, continuous_action = self.model.sample_action(
                outputs['action_type_logits'],
                outputs['action_mean'],
                outputs['action_log_std']
            )
            
        # Process outputs
        action_type = action_type.cpu().numpy()
        continuous_action = continuous_action.cpu().numpy()
        
        # Denormalize continuous action
        continuous_action = denormalize_continuous_action(
            continuous_action,
            self.config['environment']['action_space']
        )
        
        # Get action type name
        action_type_name = self.config['environment']['action_space']['discrete_actions'][action_type[0][0]]
        
        # Create response
        response = {
            'action_type': action_type_name,
            'continuous_action': continuous_action[0].tolist(),
            'value': outputs['value'].cpu().numpy()[0][0]
        }
        
        return response

def main():
    """Example script to run inference with a trained LAM model."""
    # Set up logging
    setup_logging('logs')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize inference
        model_path = os.path.join('models', 'lam_model', 'checkpoint_final.pt')
        inference = LAMInference(model_path, LAM_CONFIG)
        
        # Example inputs
        example_inputs = [
            "Move to the red box and pick it up",
            "Navigate to coordinates (2.5, 1.0) and wait",
            "Grab the blue sphere and place it on the platform"
        ]
        
        # Process each input
        for text in example_inputs:
            logger.info(f"\nProcessing input: {text}")
            
            response = inference.process_input(text)
            
            logger.info("Model response:")
            logger.info(f"Action type: {response['action_type']}")
            logger.info(f"Continuous action: {response['continuous_action']}")
            logger.info(f"Value estimate: {response['value']:.4f}")
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
