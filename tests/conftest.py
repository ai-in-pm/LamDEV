import pytest
from unittest.mock import Mock
import torch
from transformers import PreTrainedTokenizer

@pytest.fixture
def config():
    """Test configuration fixture."""
    return {
        'data': {
            'data_sources': ['documentation', 'wikihow', 'expert_demonstrations'],
            'max_examples': 100000,
            'train_test_split': 0.8
        },
        'model': {
            'model_name': 'gpt2',
            'model_type': 'gpt2',
            'max_sequence_length': 512
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'ppo_epochs': 4,
            'clip_epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5
        },
        'deployment': {
            'update_frequency': 1000,
            'safety_checks': ['bias_detection', 'content_filtering', 'action_verification'],
            'update_threshold': 50
        },
        'test_mode': True  # This will disable wandb initialization
    }

@pytest.fixture
def mock_model():
    """Mock model fixture."""
    model = Mock()
    model.to.return_value = model
    model.train.return_value = None
    model.eval.return_value = None
    return model

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer fixture."""
    tokenizer = Mock(spec=PreTrainedTokenizer)
    tokenizer.encode.return_value = torch.tensor([1] * 10)
    tokenizer.pad_token_id = 0
    tokenizer.__call__.return_value = {
        'input_ids': torch.tensor([1] * 10),
        'attention_mask': torch.tensor([1] * 10)
    }
    return tokenizer

@pytest.fixture
def sample_data():
    """Sample training data fixture."""
    return [
        {
            'text': 'example text 1',
            'action_type': 0,
            'continuous_action': [0.5, -0.5]
        },
        {
            'text': 'example text 2',
            'action_type': 1,
            'continuous_action': [-0.5, 0.5]
        }
    ]
