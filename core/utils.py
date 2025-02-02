import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

class LAMDataset(Dataset):
    """Dataset class for LAM training."""

    def __init__(self, data: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, max_length: int = 512):
        """Initialize LAMDataset."""
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an item from the dataset."""
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create batch dictionary
        batch = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'action_type': torch.tensor(item['action_type']),
            'continuous_action': torch.tensor(item['continuous_action'])
        }
        
        return batch

def create_dataloaders(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    # Set default values if not in config
    if 'model' not in config:
        config['model'] = {'max_sequence_length': 512}
    if 'training' not in config:
        config['training'] = {'batch_size': 32}

    # Create datasets
    train_dataset = LAMDataset(
        train_data,
        tokenizer,
        max_length=config['model']['max_sequence_length']
    )

    val_dataset = LAMDataset(
        val_data,
        tokenizer,
        max_length=config['model']['max_sequence_length']
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader."""
    # Initialize batch dictionary
    collated = {
        'input_ids': [],
        'attention_mask': [],
        'action_type': [],
        'continuous_action': []
    }
    
    # Collect items from each sample in batch
    for sample in batch:
        collated['input_ids'].append(sample['input_ids'])
        collated['attention_mask'].append(sample['attention_mask'])
        collated['action_type'].append(sample['action_type'])
        collated['continuous_action'].append(sample['continuous_action'])
    
    # Stack tensors
    collated['input_ids'] = torch.stack(collated['input_ids'])
    collated['attention_mask'] = torch.stack(collated['attention_mask'])
    collated['action_type'] = torch.stack(collated['action_type'])
    collated['continuous_action'] = torch.stack(collated['continuous_action'])
    
    return collated

def normalize_continuous_action(
    action: np.ndarray,
    action_space: Dict[str, Any]
) -> np.ndarray:
    """Normalize continuous actions to [-1, 1] range."""
    low = np.array(action_space['low'])
    high = np.array(action_space['high'])
    return 2.0 * (action - low) / (high - low) - 1.0

def denormalize_continuous_action(
    normalized_action: np.ndarray,
    action_space: Dict[str, Any]
) -> np.ndarray:
    """Denormalize continuous actions from [-1, 1] range."""
    low = np.array(action_space['low'])
    high = np.array(action_space['high'])
    return low + (normalized_action + 1.0) * 0.5 * (high - low)

def compute_returns(
    rewards: List[float],
    gamma: float = 0.99
) -> List[float]:
    """Compute discounted returns."""
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
        
    return returns

def save_json(data: Dict[str, Any], path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
        
def load_json(path: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
        
def setup_logging(log_dir: str, level: int = logging.INFO) -> None:
    """Set up logging configuration."""
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_file = os.path.join(log_dir, 'lam.log')
    
    # Clear previous handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

class MetricTracker:
    """Track and log training metrics."""
    
    def __init__(self):
        self.metrics = {}
        
    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            
    def get_average(self, metric: str) -> float:
        """Get average value for a metric."""
        values = self.metrics.get(metric, [])
        return sum(values) / len(values) if values else 0.0
        
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics."""
        return {
            key: sum(values) / len(values)
            for key, values in self.metrics.items()
        }

def create_env_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create environment configuration."""
    return {
        'observation_space': {
            'visual': config['environment']['observation_space']['visual'],
            'text': config['environment']['observation_space']['text'],
            'state': config['environment']['observation_space']['state']
        },
        'action_space': {
            'discrete': {
                'n': len(config['environment']['action_space']['discrete_actions'])
            },
            'continuous': {
                'low': [-1.0] * len(config['environment']['action_space']['continuous_actions']),
                'high': [1.0] * len(config['environment']['action_space']['continuous_actions'])
            }
        },
        'max_steps': config['environment']['max_steps'],
        'reward_scale': config['environment']['reward_scale']
    }
