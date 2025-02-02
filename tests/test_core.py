import os
import pytest
import torch
import numpy as np
from core.model import LAMModel, ActionBuffer
from core.training import LAMTrainer
from core.utils import (
    LAMDataset,
    normalize_continuous_action,
    denormalize_continuous_action,
    compute_returns,
    MetricTracker
)
from config.lam_config import LAM_CONFIG

@pytest.fixture
def model_config():
    """Fixture for model configuration."""
    return LAM_CONFIG

@pytest.fixture
def device():
    """Fixture for torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def model(model_config, device):
    """Fixture for LAM model."""
    model = LAMModel(model_config).to(device)
    return model

@pytest.fixture
def trainer(model_config):
    """Fixture for LAM trainer."""
    return LAMTrainer(model_config)

@pytest.fixture
def action_buffer():
    """Fixture for action buffer."""
    return ActionBuffer(capacity=100)

class TestLAMModel:
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert isinstance(model, LAMModel)
        assert hasattr(model, 'transformer')
        assert hasattr(model, 'action_type_head')
        assert hasattr(model, 'continuous_action_head')
        assert hasattr(model, 'value_head')
        
    def test_forward_pass(self, model, device):
        """Test model forward pass."""
        batch_size = 4
        seq_length = 128
        
        # Create dummy input
        input_ids = torch.randint(0, 1000, (batch_size, seq_length)).to(device)
        attention_mask = torch.ones((batch_size, seq_length)).to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask, return_value=True)
        
        # Check outputs
        assert 'action_type_logits' in outputs
        assert 'action_mean' in outputs
        assert 'action_log_std' in outputs
        assert 'value' in outputs
        
        # Check shapes
        assert outputs['action_type_logits'].shape == (batch_size, len(LAM_CONFIG['environment']['action_space']['discrete_actions']))
        assert outputs['action_mean'].shape == (batch_size, len(LAM_CONFIG['environment']['action_space']['continuous_actions']))
        assert outputs['action_log_std'].shape == (batch_size, len(LAM_CONFIG['environment']['action_space']['continuous_actions']))
        assert outputs['value'].shape == (batch_size, 1)
        
    def test_action_sampling(self, model, device):
        """Test action sampling."""
        batch_size = 4
        
        # Create dummy action distributions
        action_type_logits = torch.randn(batch_size, len(LAM_CONFIG['environment']['action_space']['discrete_actions'])).to(device)
        action_mean = torch.randn(batch_size, len(LAM_CONFIG['environment']['action_space']['continuous_actions'])).to(device)
        action_log_std = torch.zeros_like(action_mean).to(device)
        
        # Sample actions
        action_type, continuous_action = model.sample_action(
            action_type_logits,
            action_mean,
            action_log_std
        )
        
        # Check shapes
        assert action_type.shape == (batch_size, 1)
        assert continuous_action.shape == action_mean.shape

class TestLAMTrainer:
    def test_trainer_initialization(self, trainer):
        """Test trainer initialization."""
        assert isinstance(trainer, LAMTrainer)
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'optimizer')
        assert hasattr(trainer, 'action_buffer')
        
    def test_supervised_training_step(self, trainer, device):
        """Test supervised training step."""
        batch_size = 4
        seq_length = 128
        
        # Create dummy batch
        batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_length)).to(device),
            'attention_mask': torch.ones((batch_size, seq_length)).to(device),
            'action_type': torch.randint(0, 4, (batch_size,)).to(device),
            'continuous_action': torch.randn(batch_size, len(LAM_CONFIG['environment']['action_space']['continuous_actions'])).to(device)
        }
        
        # Execute training step
        metrics = trainer.supervised_training_step(batch)
        
        # Check metrics
        assert 'loss' in metrics
        assert 'action_type_loss' in metrics
        assert 'continuous_action_loss' in metrics

class TestUtils:
    def test_normalize_denormalize_actions(self):
        """Test action normalization and denormalization."""
        action_space = {
            'low': [-1.0, -2.0, 0.0],
            'high': [1.0, 2.0, 5.0]
        }
        
        original_action = np.array([0.0, 0.0, 2.5])
        
        # Normalize
        normalized = normalize_continuous_action(original_action, action_space)
        assert np.all(normalized >= -1.0) and np.all(normalized <= 1.0)
        
        # Denormalize
        denormalized = denormalize_continuous_action(normalized, action_space)
        assert np.allclose(original_action, denormalized)
        
    def test_compute_returns(self):
        """Test return computation."""
        rewards = [1.0, 0.0, 2.0, -1.0]
        gamma = 0.99
        
        returns = compute_returns(rewards, gamma)
        
        assert len(returns) == len(rewards)
        assert returns[0] > returns[1]  # First return should be highest
        
    def test_metric_tracker(self):
        """Test metric tracking."""
        tracker = MetricTracker()
        
        # Update metrics
        metrics = {'loss': 1.0, 'accuracy': 0.8}
        tracker.update(metrics)
        
        # Get average
        assert abs(tracker.get_average('loss') - 1.0) < 1e-6
        assert abs(tracker.get_average('accuracy') - 0.8) < 1e-6
        
        # Update again
        metrics = {'loss': 0.5, 'accuracy': 0.9}
        tracker.update(metrics)
        
        # Check averages
        assert abs(tracker.get_average('loss') - 0.75) < 1e-6
        assert abs(tracker.get_average('accuracy') - 0.85) < 1e-6
        
        # Reset
        tracker.reset()
        assert len(tracker.metrics) == 0

if __name__ == "__main__":
    pytest.main([__file__])
