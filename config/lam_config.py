"""Configuration settings for the Large Action Model (LAM) system."""

# Model Configuration
MODEL_CONFIG = {
    'model_name': 'gpt2',  # Base model to start with
    'model_size': 'base',  # Model size/variant
    'max_sequence_length': 512,  # Maximum sequence length for input
    'batch_size': 4,  # Training batch size
    'learning_rate': 2e-5,  # Learning rate for training
}

# Training Configuration
TRAINING_CONFIG = {
    'num_epochs': 3,  # Number of training epochs
    'warmup_steps': 500,  # Number of warmup steps
    'gradient_accumulation_steps': 4,  # Number of steps to accumulate gradients
    'max_grad_norm': 1.0,  # Maximum gradient norm for clipping
    'weight_decay': 0.01,  # Weight decay for regularization
    'evaluation_steps': 1000,  # Steps between evaluations
    'save_steps': 5000,  # Steps between model saves
}

# Data Configuration
DATA_CONFIG = {
    'data_sources': [
        'documentation',
        'wikihow',
        'expert_demonstrations'
    ],
    'train_test_split': 0.8,  # Ratio of training data
    'validation_split': 0.1,  # Ratio of validation data
    'max_examples': 100000,  # Maximum number of training examples
}

# Environment Configuration
ENVIRONMENT_CONFIG = {
    'max_steps': 100,  # Maximum steps per episode
    'reward_scale': 1.0,  # Scaling factor for rewards
    'action_space': {
        'discrete_actions': ['move', 'grab', 'drop', 'use'],
        'continuous_actions': ['position', 'rotation']
    },
    'observation_space': {
        'visual': True,
        'text': True,
        'state': True
    }
}

# Memory System Configuration
MEMORY_CONFIG = {
    'short_term_size': 1000,  # Size of short-term memory buffer
    'long_term_size': 10000,  # Size of long-term memory storage
    'memory_update_frequency': 100,  # Steps between memory updates
}

# Deployment Configuration
DEPLOYMENT_CONFIG = {
    'model_path': './models/lam_model',  # Path to save/load model
    'batch_size': 1,  # Batch size for inference
    'max_concurrent_requests': 10,  # Maximum concurrent requests
    'timeout_seconds': 30,  # Request timeout in seconds
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'metrics_update_frequency': 60,  # Seconds between metric updates
    'alert_thresholds': {
        'error_rate': 0.01,  # Maximum acceptable error rate
        'latency_ms': 1000,  # Maximum acceptable latency
        'memory_usage_gb': 8,  # Maximum memory usage
    }
}

# Ethics Configuration
ETHICS_CONFIG = {
    'safety_checks': [
        'bias_detection',
        'content_filtering',
        'action_verification'
    ],
    'update_threshold': 50,  # Minimum examples before model update
    'compliance_check_frequency': 1000,  # Steps between compliance checks
}

# Wandb Configuration
WANDB_CONFIG = {
    'project': 'large-action-model',
    'entity': 'your-entity',
    'tags': ['lam', 'development'],
}

# Combine all configurations
LAM_CONFIG = {
    'model': MODEL_CONFIG,
    'training': TRAINING_CONFIG,
    'data': DATA_CONFIG,
    'environment': ENVIRONMENT_CONFIG,
    'memory': MEMORY_CONFIG,
    'deployment': DEPLOYMENT_CONFIG,
    'monitoring': MONITORING_CONFIG,
    'ethics': ETHICS_CONFIG,
    'wandb': WANDB_CONFIG,
}
