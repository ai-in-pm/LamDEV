import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, PreTrainedTokenizer
import torch
import logging

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

class DataEngineer:
    """Data engineering agent for collecting and preprocessing data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DataEngineer."""
        self.config = config
        self.data_sources = config.get('data', {}).get('data_sources', [])
        self.raw_data_path = os.path.join('data', 'raw')
        self.processed_data_path = os.path.join('data', 'processed')
        self.tokenizer = None
        
    def collect_data(self, data_sources: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Collect data from specified sources."""
        if data_sources is None:
            data_sources = self.config.get('data', {}).get('data_sources', [])
        
        collected_data = []
        
        for source in data_sources:
            try:
                # Implement data collection logic for each source
                if source == 'documentation':
                    data = self._collect_from_documentation()
                elif source == 'wikihow':
                    data = self._collect_from_wikihow()
                elif source == 'expert_demonstrations':
                    data = self._collect_from_expert_demonstrations()
                else:
                    logger.warning(f"Unknown data source: {source}")
                    continue
                
                collected_data.extend(data)
                
            except Exception as e:
                logger.error(f"Error collecting data from {source}: {e}")
                continue
        
        # Split into train and validation sets
        train_size = int(len(collected_data) * 0.8)
        return {
            'train': collected_data[:train_size],
            'validation': collected_data[train_size:]
        }
        
    def preprocess_data(self, raw_data: List[Dict[str, Any]], tokenizer: Optional[PreTrainedTokenizer] = None) -> List[Dict[str, Any]]:
        """Preprocess raw data."""
        if not tokenizer and not self.tokenizer:
            # Initialize tokenizer if not already done
            self.setup_tokenizer()
        
        if tokenizer:
            self.tokenizer = tokenizer
            
        processed_data = []
        
        for item in raw_data:
            try:
                # Handle string inputs by converting to dict
                if isinstance(item, str):
                    item = {'text': item}
                    
                # Tokenize text
                text = item.get('text', '')
                tokenized = self.tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.config.get('model', {}).get('max_length', 128),
                    return_tensors='pt'
                )
                
                # Process actions
                action_type = item.get('action_type', 0)  # Default to 0 (no action)
                continuous_action = item.get('continuous_action', [0.0, 0.0])  # Default to [0, 0]
                
                processed_item = {
                    'input_ids': tokenized['input_ids'].squeeze(0),
                    'attention_mask': tokenized['attention_mask'].squeeze(0),
                    'action_type': torch.tensor(action_type),
                    'continuous_action': torch.tensor(continuous_action)
                }
                
                processed_data.append(processed_item)
                
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                continue
                
        return processed_data
        
    def validate_data(self, data: List[Dict[str, Any]]) -> bool:
        """Validate data quality."""
        if not data or not isinstance(data, list):
            return False
            
        required_fields = ['text', 'action_type', 'continuous_action']
        for item in data:
            if not all(field in item for field in required_fields):
                return False
            if not isinstance(item['text'], str):
                return False
            if not isinstance(item['action_type'], int):
                return False
            if not isinstance(item['continuous_action'], list):
                return False
        return True
        
    def augment_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Augment data with additional features or examples."""
        augmented_data = []
        for item in data:
            augmented_data.append(item)
            # Add simple augmentation by copying with slight modifications
            augmented_item = item.copy()
            augmented_item['continuous_action'] = [-x for x in item['continuous_action']]
            augmented_data.append(augmented_item)
        return augmented_data
        
    def save_processed_data(self, data: List[Dict[str, Any]], filename: str) -> None:
        """Save processed data to disk."""
        os.makedirs(self.processed_data_path, exist_ok=True)
        torch.save(data, os.path.join(self.processed_data_path, filename))

    def setup_tokenizer(self):
        """Setup the tokenizer."""
        model_name = self.config.get('model', {}).get('base_model', 'gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _collect_from_documentation(self):
        # Mock implementation for testing
        return [
            {'text': 'example text', 'action_type': 0, 'continuous_action': [0.5, -0.5]},
            {'text': 'another example', 'action_type': 1, 'continuous_action': [-0.5, 0.5]}
        ]

    def _collect_from_wikihow(self):
        # Mock implementation for testing
        return [
            {'text': 'wikihow example', 'action_type': 0, 'continuous_action': [0.3, -0.3]},
        ]

    def _collect_from_expert_demonstrations(self):
        # Mock implementation for testing
        return [
            {'text': 'expert demonstration example', 'action_type': 1, 'continuous_action': [0.2, -0.2]},
        ]

if __name__ == "__main__":
    # Test data engineer functionality
    data_engineer = DataEngineer({'data': {'data_sources': []}})
    # Add test cases here
