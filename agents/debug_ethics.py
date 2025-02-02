import os
from typing import Dict, Any, List
import torch
import wandb
from dotenv import load_dotenv
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DebugEthicsOfficer:
    """Debug and ethics monitoring agent for the LAM system."""
    
    def __init__(self, config: Dict[str, Any], model: Any):
        """Initialize DebugEthicsOfficer."""
        self.config = config
        self.model = model
        self.safety_rules = self._load_safety_rules()
        
        # Initialize wandb only if not in test mode
        if not config.get('test_mode', False):
            try:
                wandb.init(project="large-action-model")
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
        
    def check_model_safety(self, action: Dict[str, Any]) -> bool:
        """Check if a proposed action is safe."""
        # Check action type
        if not isinstance(action.get('type'), str):
            return False
            
        # Check parameters
        params = action.get('parameters', {})
        if not isinstance(params, dict):
            return False
            
        # Check specific safety rules
        for rule in self.safety_rules:
            if not self._check_rule(action, rule):
                return False
                
        return True
        
    def monitor_compliance(self) -> Dict[str, Any]:
        """Monitor ethical compliance of the system."""
        compliance_status = {
            'compliant': True,
            'violations': [],
            'recommendations': []
        }
        
        # Check model parameters if model is a torch model
        if isinstance(self.model, torch.nn.Module):
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    compliance_status['compliant'] = False
                    compliance_status['violations'].append(f"NaN values detected in {name}")
                    compliance_status['recommendations'].append(f"Reset or reinitialize {name}")
        
        if not self.config.get('test_mode', False):
            wandb.log({"compliance_status": compliance_status})
            
        return compliance_status
        
    def _load_safety_rules(self) -> List[Dict[str, Any]]:
        """Load safety rules from configuration."""
        default_rules = [
            {
                'name': 'movement_bounds',
                'description': 'Movement must stay within safe bounds',
                'recommendation': 'Ensure movement parameters are within [-1, 1] range'
            },
            {
                'name': 'action_speed',
                'description': 'Action speed must not exceed safe limits',
                'recommendation': 'Reduce action speed to safe levels'
            }
        ]
        return self.config.get('safety_rules', default_rules)
        
    def _check_rule(self, action: Dict[str, Any], rule: Dict[str, Any]) -> bool:
        """Check if an action complies with a specific safety rule."""
        if rule['name'] == 'movement_bounds':
            params = action.get('parameters', {})
            return all(-1 <= params.get(k, 0) <= 1 for k in ['x', 'y'])
            
        if rule['name'] == 'action_speed':
            params = action.get('parameters', {})
            return params.get('speed', 1) <= 1.0
            
        return True

    def check_action_safety(self, action: Dict[str, Any]) -> bool:
        """Check if an action is safe to execute."""
        try:
            # Check action type
            if 'type' not in action:
                logger.warning("Action missing 'type' field")
                return False
                
            # Check parameters
            if 'parameters' not in action:
                logger.warning("Action missing 'parameters' field")
                return False
                
            # Apply safety rules based on action type
            if action['type'] == 'move':
                return self._check_move_safety(action['parameters'])
            elif action['type'] == 'grab':
                return self._check_grab_safety(action['parameters'])
            else:
                logger.warning(f"Unknown action type: {action['type']}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking action safety: {e}")
            return False

    def monitor_ethical_compliance(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Monitor ethical compliance of actions."""
        report = {
            'compliant': True,
            'violations': [],
            'recommendations': []
        }

        try:
            for action in actions:
                # Check action type
                if not self._is_valid_action_type(action.get('type')):
                    report['compliant'] = False
                    report['violations'].append(f"Invalid action type: {action.get('type')}")

                # Check parameters
                if not self._are_valid_parameters(action.get('parameters', {})):
                    report['compliant'] = False
                    report['violations'].append(f"Invalid parameters in action: {action}")

                # Check for potential harmful actions
                if self._is_potentially_harmful(action):
                    report['compliant'] = False
                    report['violations'].append(f"Potentially harmful action detected: {action}")

            # Only log to wandb if not in test mode and wandb is available
            if not self.config.get('test_mode', False):
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({'ethical_compliance': report['compliant']})
                except (ImportError, wandb.Error):
                    pass

        except Exception as e:
            report['compliant'] = False
            report['violations'].append(f"Error during compliance check: {str(e)}")

        return report

    def _is_valid_action_type(self, action_type: str) -> bool:
        """Check if action type is valid."""
        valid_types = {'move', 'grab', 'place', 'rotate', 'push', 'pull'}
        return action_type in valid_types

    def _are_valid_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Check if action parameters are valid."""
        required_params = {
            'move': {'x', 'y'},
            'grab': {'object'},
            'place': {'location'},
            'rotate': {'angle'},
            'push': {'direction', 'force'},
            'pull': {'direction', 'force'}
        }
        
        action_type = parameters.get('type')
        if action_type not in required_params:
            return False
            
        return all(param in parameters for param in required_params[action_type])

    def _is_potentially_harmful(self, action: Dict[str, Any]) -> bool:
        """Check if action is potentially harmful."""
        # Example checks - customize based on your needs
        if action.get('type') in {'push', 'pull'}:
            force = action.get('parameters', {}).get('force', 0)
            if force > self.config.get('safety', {}).get('max_force', 10):
                return True
        return False

    def get_error_log(self) -> List[Dict[str, Any]]:
        """Get the error log."""
        if not hasattr(self, 'error_log'):
            self.error_log = []
        return self.error_log
        
    def log_error(self, error_type: str, message: str, details: Dict[str, Any] = None) -> None:
        """Log an error."""
        if not hasattr(self, 'error_log'):
            self.error_log = []
            
        error_entry = {
            'type': error_type,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.error_log.append(error_entry)
        
        # Log to wandb if enabled
        if not self.config.get('test_mode', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'error_type': error_type,
                        'error_message': message
                    })
            except (ImportError, wandb.Error):
                pass

    def debug_model_failure(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Debug model failure and provide a report."""
        report = {
            'error_type': error_info.get('error', 'Unknown error'),
            'timestamp': datetime.now().isoformat(),
            'analysis': [],
            'recommendations': []
        }
        
        # Analyze error type
        error_str = str(error_info.get('error', ''))
        
        if 'TypeError' in error_str:
            report['analysis'].append('Type mismatch in function arguments or data processing')
            report['recommendations'].extend([
                'Check input data types and function signatures',
                'Verify data preprocessing steps',
                'Add type hints and validation'
            ])
        elif 'ValueError' in error_str:
            report['analysis'].append('Invalid value or operation')
            report['recommendations'].extend([
                'Validate input ranges and constraints',
                'Check for missing or invalid configuration values',
                'Add input validation checks'
            ])
        elif 'RuntimeError' in error_str:
            report['analysis'].append('Runtime execution failure')
            report['recommendations'].extend([
                'Check system resources and GPU memory',
                'Verify model architecture compatibility',
                'Review batch size and model parameters'
            ])
        else:
            report['analysis'].append('Unrecognized error type')
            report['recommendations'].extend([
                'Add detailed logging and error tracking',
                'Review recent code changes',
                'Add unit tests for edge cases'
            ])
        
        # Check ethical implications
        ethical_check = self.check_model_safety({
            'error_context': error_info,
            'model_state': 'failed'
        })
        report['ethical_implications'] = ethical_check
        
        # Log report
        if not self.config.get('test_mode', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'debug_report': report
                    })
            except (ImportError, wandb.Error):
                pass
        
        return report

    def _check_move_safety(self, parameters: Dict[str, Any]) -> bool:
        """Check safety of move action parameters."""
        if 'x' not in parameters or 'y' not in parameters:
            return False
        # Add more specific safety checks as needed
        return True
        
    def _check_grab_safety(self, parameters: Dict[str, Any]) -> bool:
        """Check safety of grab action parameters."""
        if 'object' not in parameters:
            return False
        # Add more specific safety checks as needed
        return True

if __name__ == "__main__":
    # Test debug ethics officer functionality
    officer = DebugEthicsOfficer({'test_mode': True}, None)
    # Add test cases here
