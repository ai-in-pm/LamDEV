import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from tqdm import tqdm
from .model import LAMModel, ActionBuffer
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class LAMTrainer:
    """Trainer class for the Large Action Model."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the LAM trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set default training parameters if not provided
        if 'training' not in self.config:
            self.config['training'] = {}
            
        # Set default values for training parameters
        training_defaults = {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'ppo_epochs': 4,
            'clip_epsilon': 0.2,
            'value_loss_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'buffer_capacity': 10000,
            'continuous_action_weight': 1.0
        }
        
        for key, value in training_defaults.items():
            if key not in self.config['training']:
                self.config['training'][key] = value
        
        # Initialize model and optimizer
        self.model = LAMModel(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate']
        )
        
        # Initialize action buffer for RL training
        self.action_buffer = ActionBuffer(self.config['training']['buffer_capacity'])

    def supervised_training_step(self, batch: Union[Dict[str, torch.Tensor], List[torch.Tensor]]) -> Dict[str, float]:
        """Perform a single supervised training step."""
        # Handle both list and dict batch formats
        if isinstance(batch, list) or isinstance(batch, tuple):
            input_ids = batch[0]
            attention_mask = batch[1] if len(batch) > 1 else None
            action_type = batch[2] if len(batch) > 2 else None
            continuous_action = batch[3] if len(batch) > 3 else None
        else:
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask')
            action_type = batch.get('action_type')
            continuous_action = batch.get('continuous_action')

        # Move tensors to device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if action_type is not None:
            action_type = action_type.to(self.device)
        if continuous_action is not None:
            continuous_action = continuous_action.to(self.device)

        # Forward pass
        outputs = self.model(input_ids, attention_mask=attention_mask)

        # Calculate losses
        total_loss = 0.0
        metrics = {}

        if action_type is not None and 'action_type_logits' in outputs:
            action_type_loss = F.cross_entropy(outputs['action_type_logits'], action_type)
            total_loss += action_type_loss
            metrics['action_type_loss'] = action_type_loss.item()

        if continuous_action is not None and 'action_mean' in outputs:
            continuous_action_loss = F.mse_loss(outputs['action_mean'], continuous_action)
            total_loss += self.config.get('training', {}).get('continuous_action_weight', 1.0) * continuous_action_loss
            metrics['continuous_action_loss'] = continuous_action_loss.item()

        metrics['total_loss'] = total_loss.item()
        metrics['loss'] = total_loss.item()  # For compatibility with test expectations

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.config.get('training', {}).get('max_grad_norm'):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
        self.optimizer.step()

        return metrics

    def rl_training_step(
        self,
        buffer_data: Dict[str, torch.Tensor],
        epsilon: float = 0.2
    ) -> Dict[str, float]:
        """Execute a reinforcement learning training step using PPO."""
        self.model.train()

        # Move data to device
        observations = {k: v.to(self.device) for k, v in buffer_data['observations'].items()}
        old_action_types = buffer_data['action_types'].to(self.device)
        old_continuous_actions = buffer_data['continuous_actions'].to(self.device)
        old_values = buffer_data['values'].to(self.device)
        old_action_type_log_probs = buffer_data['action_type_log_probs'].to(self.device)
        old_continuous_action_log_probs = buffer_data['continuous_action_log_probs'].to(self.device)
        rewards = buffer_data['rewards'].to(self.device)
        masks = buffer_data['masks'].to(self.device)

        # Compute advantages
        with torch.no_grad():
            advantages = self._compute_advantages(rewards, old_values, masks)
            returns = advantages + old_values

        # PPO iterations
        for _ in range(self.config['training']['ppo_epochs']):
            # Forward pass
            outputs = self.model(
                input_ids=observations['input_ids'],
                attention_mask=observations['attention_mask'],
                action_types=old_action_types,
                continuous_actions=old_continuous_actions
            )

            # Compute PPO losses
            ratio = torch.exp(outputs['action_type_log_probs'] - old_action_type_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            action_type_loss = -torch.min(surr1, surr2).mean()

            ratio = torch.exp(outputs['continuous_action_log_probs'] - old_continuous_action_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            continuous_action_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * ((outputs['values'] - returns) ** 2).mean()
            entropy_loss = -(outputs['action_type_entropy'] + outputs['continuous_action_entropy']).mean()

            # Compute total loss
            total_loss = (
                action_type_loss +
                continuous_action_loss +
                self.config['training']['value_loss_coef'] * value_loss +
                self.config['training']['entropy_coef'] * entropy_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
            self.optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'action_type_loss': action_type_loss.item(),
            'continuous_action_loss': continuous_action_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item()
        }

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> torch.Tensor:
        """Compute generalized advantage estimation (GAE)."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + gamma * next_value * masks[t] - values[t]
            advantages[t] = delta + gamma * lambda_ * masks[t] * last_advantage
            last_advantage = advantages[t]
            
        return advantages
        
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Checkpoint saved to {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")
        
    def train_supervised(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, num_epochs: int = 1) -> Dict[str, float]:
        """Train the model using supervised learning."""
        metrics = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            num_batches = 0

            for batch in train_dataloader:
                # Forward pass and compute loss
                batch_metrics = self.supervised_training_step(batch)
                train_loss += batch_metrics['loss']
                
                # Compute accuracy if possible
                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0]
                    action_type = batch[2] if len(batch) > 2 else None
                else:
                    input_ids = batch['input_ids']
                    action_type = batch.get('action_type')

                if action_type is not None:
                    with torch.no_grad():
                        outputs = self.model(input_ids.to(self.device))
                        pred_action = torch.argmax(outputs['action_type_logits'], dim=-1)
                        train_acc += (pred_action == action_type.to(self.device)).float().mean().item()

                num_batches += 1

            # Calculate training metrics
            if num_batches > 0:
                train_loss /= num_batches
                train_acc /= num_batches
                metrics['train_loss'].append(train_loss)
                metrics['train_accuracy'].append(train_acc)

            # Validation
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                metrics['val_loss'].append(val_metrics['loss'])
                metrics['val_accuracy'].append(val_metrics['accuracy'])

            # Log metrics if not in test mode
            if not self.config.get('test_mode', False):
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            'train/loss': train_loss,
                            'train/accuracy': train_acc,
                            'train/epoch': epoch,
                            'val/loss': val_metrics['loss'] if val_dataloader else None,
                            'val/accuracy': val_metrics['accuracy'] if val_dataloader else None
                        })
                except (ImportError, wandb.Error):
                    pass

        return metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                # Handle both list and dict batch formats
                if isinstance(batch, (list, tuple)):
                    input_ids = batch[0]
                    attention_mask = batch[1] if len(batch) > 1 else None
                    action_type = batch[2] if len(batch) > 2 else None
                    continuous_action = batch[3] if len(batch) > 3 else None
                else:
                    input_ids = batch['input_ids']
                    attention_mask = batch.get('attention_mask')
                    action_type = batch.get('action_type')
                    continuous_action = batch.get('continuous_action')

                # Move tensors to device
                input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                if action_type is not None:
                    action_type = action_type.to(self.device)
                if continuous_action is not None:
                    continuous_action = continuous_action.to(self.device)

                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)

                # Calculate losses
                total_batch_loss = 0.0

                if action_type is not None and 'action_type_logits' in outputs:
                    action_type_loss = F.cross_entropy(outputs['action_type_logits'], action_type)
                    total_batch_loss += action_type_loss
                    pred_action = torch.argmax(outputs['action_type_logits'], dim=-1)
                    total_acc += (pred_action == action_type).float().mean().item()

                if continuous_action is not None and 'action_mean' in outputs:
                    continuous_action_loss = F.mse_loss(outputs['action_mean'], continuous_action)
                    total_batch_loss += self.config.get('training', {}).get('continuous_action_weight', 1.0) * continuous_action_loss

                total_loss += total_batch_loss.item()
                num_batches += 1

        return {
            'loss': total_loss / num_batches if num_batches > 0 else float('inf'),
            'accuracy': total_acc / num_batches if num_batches > 0 else 0.0
        }
