import os
import pytest
import torch
import json
import logging
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from transformers import PreTrainedTokenizer
from core.utils import (
    LAMDataset,
    create_dataloaders,
    normalize_continuous_action,
    denormalize_continuous_action,
    compute_returns,
    save_json,
    load_json,
    setup_logging,
    MetricTracker,
    create_env_config
)
from config.lam_config import LAM_CONFIG

@pytest.fixture
def mock_tokenizer():
    """Fixture for mock tokenizer."""
    tokenizer = Mock(spec=PreTrainedTokenizer)
    tokenizer.return_value = {
        'input_ids': torch.ones(1, 10),
        'attention_mask': torch.ones(1, 10)
    }
    return tokenizer

@pytest.fixture
def sample_data():
    """Fixture for sample data."""
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

class TestLAMDataset:
    """Tests for LAMDataset class."""
    
    def test_initialization(self, sample_data, mock_tokenizer):
        """Test dataset initialization."""
        dataset = LAMDataset(sample_data, mock_tokenizer)
        assert len(dataset) == len(sample_data)
        assert dataset.tokenizer == mock_tokenizer
        assert dataset.max_length == 512
        
    def test_getitem(self, sample_data, mock_tokenizer):
        """Test dataset item retrieval."""
        dataset = LAMDataset(sample_data, mock_tokenizer)
        item = dataset[0]
        
        assert isinstance(item, dict)
        assert 'input_ids' in item
        assert 'attention_mask' in item
        assert 'action_type' in item
        assert 'continuous_action' in item
        assert isinstance(item['action_type'], torch.Tensor)
        assert isinstance(item['continuous_action'], torch.Tensor)

class TestDataLoaders:
    """Tests for dataloader creation functions."""
    
    def test_create_dataloaders(self, sample_data, mock_tokenizer):
        """Test dataloader creation."""
        train_loader, val_loader = create_dataloaders(
            sample_data,
            sample_data,
            mock_tokenizer,
            LAM_CONFIG
        )
        
        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(val_loader, torch.utils.data.DataLoader)
        assert len(train_loader) > 0
        assert len(val_loader) > 0

class TestActionProcessing:
    """Tests for action processing functions."""
    
    def test_normalize_continuous_action(self):
        """Test action normalization."""
        action = np.array([0.5, 1.5])
        action_space = {'low': [0.0, 0.0], 'high': [1.0, 2.0]}
        
        normalized = normalize_continuous_action(action, action_space)
        assert isinstance(normalized, np.ndarray)
        assert normalized.shape == action.shape
        assert np.all(normalized >= -1.0) and np.all(normalized <= 1.0)
        
    def test_denormalize_continuous_action(self):
        """Test action denormalization."""
        normalized_action = np.array([0.0, 0.0])
        action_space = {'low': [0.0, 0.0], 'high': [1.0, 2.0]}
        
        denormalized = denormalize_continuous_action(normalized_action, action_space)
        assert isinstance(denormalized, np.ndarray)
        assert denormalized.shape == normalized_action.shape
        assert np.allclose(denormalized, [0.5, 1.0])
        
    def test_normalization_roundtrip(self):
        """Test normalization followed by denormalization."""
        original_action = np.array([0.5, 1.5])
        action_space = {'low': [0.0, 0.0], 'high': [1.0, 2.0]}
        
        normalized = normalize_continuous_action(original_action, action_space)
        denormalized = denormalize_continuous_action(normalized, action_space)
        
        assert np.allclose(original_action, denormalized)

class TestReturnsComputation:
    """Tests for return computation functions."""
    
    @pytest.mark.parametrize("gamma", [0.99, 0.95])
    def test_compute_returns(self, gamma):
        """Test return computation with different discount factors."""
        rewards = [1.0, 0.0, 2.0, -1.0]
        returns = compute_returns(rewards, gamma)
        
        assert len(returns) == len(rewards)
        assert returns[0] > returns[1]  # First return should be highest
        
        # Manually compute expected returns
        expected_returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            expected_returns.insert(0, R)
            
        assert np.allclose(returns, expected_returns)

class TestFileOperations:
    """Tests for file operation functions."""
    
    def test_save_load_json(self, tmp_path):
        """Test JSON file saving and loading."""
        data = {'key': 'value', 'number': 42}
        file_path = os.path.join(tmp_path, 'test.json')
        
        # Save JSON
        save_json(data, file_path)
        assert os.path.exists(file_path)
        
        # Load JSON
        loaded_data = load_json(file_path)
        assert loaded_data == data

class TestLogging:
    """Tests for logging setup functions."""
    
    def test_setup_logging(self, tmp_path):
        """Test logging setup."""
        log_dir = os.path.join(tmp_path, 'logs')
        setup_logging(log_dir)
        
        logger = logging.getLogger(__name__)
        logger.info('Test message')
        
        log_file = os.path.join(log_dir, 'lam.log')
        assert os.path.exists(log_file)
        
        with open(log_file, 'r') as f:
            content = f.read()
            assert 'Test message' in content

class TestMetricTracker:
    """Tests for MetricTracker class."""
    
    def test_update_and_average(self):
        """Test metric updating and averaging."""
        tracker = MetricTracker()
        
        metrics1 = {'loss': 1.0, 'accuracy': 0.8}
        metrics2 = {'loss': 0.5, 'accuracy': 0.9}
        
        tracker.update(metrics1)
        assert abs(tracker.get_average('loss') - 1.0) < 1e-6
        
        tracker.update(metrics2)
        assert abs(tracker.get_average('loss') - 0.75) < 1e-6
        
    def test_reset(self):
        """Test metric reset."""
        tracker = MetricTracker()
        tracker.update({'loss': 1.0})
        tracker.reset()
        assert len(tracker.metrics) == 0
        
    def test_get_summary(self):
        """Test summary generation."""
        tracker = MetricTracker()
        metrics = {'loss': 1.0, 'accuracy': 0.8}
        tracker.update(metrics)
        
        summary = tracker.get_summary()
        assert isinstance(summary, dict)
        assert set(summary.keys()) == set(metrics.keys())

class TestEnvironmentConfig:
    """Tests for environment configuration functions."""
    
    def test_create_env_config(self):
        """Test environment configuration creation."""
        config = create_env_config(LAM_CONFIG)
        
        assert isinstance(config, dict)
        assert 'observation_space' in config
        assert 'action_space' in config
        assert 'max_steps' in config
        assert 'reward_scale' in config
        
        action_space = config['action_space']
        assert 'discrete' in action_space
        assert 'continuous' in action_space
        assert len(action_space['continuous']['low']) == len(action_space['continuous']['high'])
