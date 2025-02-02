import torch
import torch.nn as nn
from transformers import GPT2Model, AutoConfig
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class LAMModel(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model."""
        super().__init__()
        
        # Load base transformer model
        self.transformer = GPT2Model.from_pretrained('gpt2')
        hidden_size = self.transformer.config.n_embd
        
        # Get action space sizes from config
        self.action_type_size = len(config['environment']['action_space']['discrete_actions'])
        self.continuous_action_size = len(config['environment']['action_space']['continuous_actions'])
        
        # Action type head
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_type_size)
        )
        
        # Action parameters head (mean and log_std)
        self.continuous_action_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.continuous_action_size * 2)
        )
        
        # Value head for critic
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_value: bool = False,
        return_dict: bool = True
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """Forward pass of the model."""
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Get transformer outputs
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get the last hidden state
        last_hidden_state = transformer_outputs.last_hidden_state
        batch_size = last_hidden_state.size(0)
        
        # Average pool over sequence length
        pooled_output = last_hidden_state.mean(dim=1)

        # Action type prediction
        action_type_logits = self.action_type_head(pooled_output)

        # Action parameters prediction (mean and log_std)
        continuous_action = self.continuous_action_head(pooled_output)
        action_mean, action_log_std = torch.chunk(continuous_action, 2, dim=-1)
        action_std = torch.exp(action_log_std)

        if not return_dict:
            outputs = (action_type_logits, action_mean, action_std)
            if return_value:
                value = self.value_head(pooled_output)
                outputs = outputs + (value,)
            return outputs

        outputs = {
            'action_type_logits': action_type_logits,
            'action_mean': action_mean,
            'action_std': action_std,
            'action_log_std': action_log_std,
            'continuous_action': continuous_action
        }

        if return_value:
            value = self.value_head(pooled_output)
            outputs['value'] = value

        return outputs
        
    def sample_action(
        self,
        action_type_logits: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions from predicted distributions."""
        # Sample discrete action type
        action_type_probs = torch.softmax(action_type_logits, dim=-1)
        action_type = torch.multinomial(action_type_probs, 1)
        
        # Sample continuous actions
        continuous_action = action_mean + action_std * torch.randn_like(action_mean)
        
        return action_type, continuous_action
        
    def get_action_log_probs(
        self,
        action_type_logits: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        action_type: torch.Tensor,
        continuous_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate log probabilities of actions."""
        # Discrete action log probs
        action_type_log_probs = torch.log_softmax(action_type_logits, dim=-1)
        action_type_log_prob = action_type_log_probs.gather(1, action_type)
        
        # Continuous action log probs
        continuous_log_prob = -0.5 * (
            ((continuous_action - action_mean) / action_std) ** 2 +
            2 * torch.log(action_std) +
            torch.log(torch.tensor(2 * torch.pi))
        ).sum(dim=-1, keepdim=True)
        
        return action_type_log_prob, continuous_log_prob
        
class ActionBuffer:
    """Buffer for storing trajectory data for RL training."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.reset()
        
    def reset(self):
        """Reset the buffer."""
        self.observations = []
        self.action_types = []
        self.continuous_actions = []
        self.rewards = []
        self.values = []
        self.action_type_log_probs = []
        self.continuous_action_log_probs = []
        self.masks = []
        
    def add(
        self,
        observation: Dict[str, torch.Tensor],
        action_type: torch.Tensor,
        continuous_action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        action_type_log_prob: torch.Tensor,
        continuous_action_log_prob: torch.Tensor,
        mask: float
    ):
        """Add a transition to the buffer."""
        self.observations.append(observation)
        self.action_types.append(action_type)
        self.continuous_actions.append(continuous_action)
        self.rewards.append(reward)
        self.values.append(value)
        self.action_type_log_probs.append(action_type_log_prob)
        self.continuous_action_log_probs.append(continuous_action_log_prob)
        self.masks.append(mask)
        
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data from the buffer."""
        data = {
            'observations': self.observations,
            'action_types': torch.cat(self.action_types),
            'continuous_actions': torch.cat(self.continuous_actions),
            'rewards': torch.tensor(self.rewards),
            'values': torch.cat(self.values),
            'action_type_log_probs': torch.cat(self.action_type_log_probs),
            'continuous_action_log_probs': torch.cat(self.continuous_action_log_probs),
            'masks': torch.tensor(self.masks)
        }
        self.reset()
        return data
