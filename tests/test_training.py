import os
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from core.training import LAMTrainer
from core.model import LAMModel
from config.lam_config import LAM_CONFIG
from torch.utils.data import DataLoader, TensorDataset

@pytest.fixture
def config():
    """Fixture for configuration."""
    return LAM_CONFIG

@pytest.fixture
def device():
    """Fixture for torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def mock_model():
    """Fixture for mock model."""
    model = Mock(spec=LAMModel)
    model.to.return_value = model
    model.train.return_value = None
    model.eval.return_value = None
    return model

@pytest.fixture
def trainer(config, mock_model):
    """Fixture for LAMTrainer instance."""
    return LAMTrainer(config)

@pytest.fixture
def dummy_batch(device):
    """Fixture for dummy batch data."""
    batch_size = 4
    seq_length = 128
    return {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length)).to(device),
        'attention_mask': torch.ones(batch_size, seq_length).to(device),
        'action_type': torch.randint(0, 4, (batch_size,)).to(device),
        'continuous_action': torch.randn(batch_size, 2).to(device)
    }

@pytest.fixture
def dummy_dataloader(dummy_batch):
    """Fixture for dummy dataloader."""
    dataset = TensorDataset(
        dummy_batch['input_ids'],
        dummy_batch['attention_mask'],
        dummy_batch['action_type'],
        dummy_batch['continuous_action']
    )
    return DataLoader(dataset, batch_size=2)

class TestLAMTrainer:
    """Tests for LAMTrainer class."""
    
    def test_initialization(self, trainer):
        """Test trainer initialization."""
        assert isinstance(trainer.model, LAMModel)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert hasattr(trainer, 'action_buffer')
        
    def test_supervised_training_step(self, trainer, dummy_batch):
        """Test supervised training step."""
        metrics = trainer.supervised_training_step(dummy_batch)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'action_type_loss' in metrics
        assert 'continuous_action_loss' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
        
    def test_rl_training_step(self, trainer, device):
        """Test reinforcement learning training step."""
        buffer_data = {
            'observations': {
                'input_ids': torch.randint(0, 1000, (4, 128)).to(device),
                'attention_mask': torch.ones(4, 128).to(device)
            },
            'action_types': torch.randint(0, 4, (4,)).to(device),
            'continuous_actions': torch.randn(4, 2).to(device),
            'values': torch.randn(4, 1).to(device),
            'action_type_log_probs': torch.randn(4).to(device),
            'continuous_action_log_probs': torch.randn(4).to(device),
            'rewards': torch.randn(4).to(device),
            'masks': torch.ones(4).to(device)
        }
        
        metrics = trainer.rl_training_step(buffer_data)
        
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'action_loss' in metrics
        assert 'value_loss' in metrics
        assert 'entropy_loss' in metrics
        assert all(isinstance(v, float) for v in metrics.values())
        
    def test_compute_advantages(self, trainer):
        """Test advantage computation."""
        rewards = torch.tensor([1.0, 0.0, 2.0, -1.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.5])
        masks = torch.ones_like(rewards)
        
        advantages = trainer._compute_advantages(rewards, values, masks)
        
        assert isinstance(advantages, torch.Tensor)
        assert advantages.shape == rewards.shape
        assert torch.all(torch.isfinite(advantages))
        
    @pytest.mark.parametrize("num_epochs", [1, 2])
    def test_train_supervised(self, trainer, dummy_dataloader, num_epochs):
        """Test supervised training loop."""
        val_dataloader = dummy_dataloader
        
        with patch('wandb.log') as mock_wandb_log:
            trainer.train_supervised(dummy_dataloader, val_dataloader, num_epochs)
            
            # Check that wandb.log was called at least once per epoch
            assert mock_wandb_log.call_count >= num_epochs
            
    def test_save_load_checkpoint(self, trainer, tmp_path):
        """Test checkpoint saving and loading."""
        checkpoint_path = os.path.join(tmp_path, "checkpoint.pt")
        
        # Save checkpoint
        trainer.save_checkpoint(checkpoint_path)
        assert os.path.exists(checkpoint_path)
        
        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)
        
        # Verify model state
        assert isinstance(trainer.model, LAMModel)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        
    def test_training_with_gradient_clipping(self, trainer, dummy_batch):
        """Test training with gradient clipping."""
        # Set a very small max_grad_norm to force clipping
        trainer.config['training']['max_grad_norm'] = 0.1
        
        metrics = trainer.supervised_training_step(dummy_batch)
        
        assert isinstance(metrics, dict)
        assert metrics['loss'] > 0
        
        # Verify gradients are clipped
        max_grad = max(
            p.grad.abs().max()
            for p in trainer.model.parameters()
            if p.grad is not None
        )
        assert max_grad <= 0.1 + 1e-6  # Allow small numerical error
