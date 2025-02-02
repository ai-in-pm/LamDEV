import os
import time
from typing import Dict, Any, List
import psutil
import wandb
from dotenv import load_dotenv
import torch
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize logger
logger = logging.getLogger(__name__)

class DeploymentMonitor:
    """Deployment monitoring agent for the LAM system."""
    
    def __init__(self, config: Dict[str, Any], model: Any):
        """Initialize DeploymentMonitor."""
        self.config = config
        self.model = model
        self.metrics_history = []
        
        # Initialize wandb only if not in test mode
        if not config.get('test_mode', False):
            try:
                wandb.init(project="large-action-model")
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
        
    def check_health(self) -> Dict[str, Any]:
        """Check the system's health status."""
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        status = 'healthy'
        if cpu_usage > 90 or memory_usage > 90:
            status = 'warning'
        if cpu_usage > 95 or memory_usage > 95:
            status = 'critical'
            
        health_status = {
            'status': status,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'timestamp': time.time()
        }
        
        if not self.config.get('test_mode', False):
            wandb.log({"health_status": health_status})
            
        return health_status
        
    def monitor_performance(self) -> Dict[str, float]:
        """Monitor model performance metrics."""
        metrics = {
            'throughput': 0.0,  # requests per second
            'latency': 0.0,     # average response time
            'error_rate': 0.0,  # percentage of failed requests
            'memory_usage': 0.0, # model memory usage
            'performance': 100.0  # overall performance score
        }
        
        # In a real implementation, these would be calculated from actual monitoring data
        # For testing, we're returning default values
        if not self.config.get('test_mode', False):
            wandb.log(metrics)
            
        return metrics
        
    def alert_if_needed(self, metrics: Dict[str, float]) -> List[str]:
        """Generate alerts if metrics exceed thresholds."""
        alerts = []
        
        if metrics.get('error_rate', 0) > 0.1:
            alerts.append("High error rate detected")
            
        if metrics.get('latency', 0) > 1.0:
            alerts.append("High latency detected")
            
        if metrics.get('memory_usage', 0) > 90:
            alerts.append("High memory usage detected")
            
        if metrics.get('performance', 100) < 80:
            alerts.append("Low performance detected")
            
        return alerts
        
    def log_deployment_event(self, event: Dict[str, Any]):
        """Log deployment-related events."""
        if not self.config.get('test_mode', False):
            wandb.log({"deployment_event": event})
        self.metrics_history.append(event)

    def check_system_health(self) -> Dict[str, Any]:
        """Check the health of the deployed system."""
        health_status = {
            'model_loaded': hasattr(self, 'model'),
            'gpu_available': torch.cuda.is_available(),
            'memory_usage': psutil.Process().memory_info().rss / (1024 * 1024),  # MB
            'cpu_usage': psutil.Process().cpu_percent(),
            'errors': []
        }
        
        # Check model state
        if health_status['model_loaded']:
            try:
                # Test model with dummy input
                dummy_input = torch.randint(0, 1000, (1, 128)).to(self.device)
                dummy_mask = torch.ones(1, 128).to(self.device)
                with torch.no_grad():
                    _ = self.model(input_ids=dummy_input, attention_mask=dummy_mask)
                health_status['model_responsive'] = True
            except Exception as e:
                health_status['model_responsive'] = False
                health_status['errors'].append(f"Model error: {str(e)}")
        
        # Check GPU memory if available
        if health_status['gpu_available']:
            try:
                health_status['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                health_status['gpu_memory_cached'] = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            except Exception as e:
                health_status['errors'].append(f"GPU error: {str(e)}")
        
        # Log health status
        if not self.config.get('test_mode', False):
            wandb.log({
                'system_health': health_status
            })
        
        return health_status

    def deploy_model(self, model_path: str) -> bool:
        """Deploy a model to production."""
        try:
            # Verify model exists
            if not os.path.exists(model_path):
                logger.error(f"Model not found at {model_path}")
                return False
                
            # Load model
            try:
                self.model.load_state_dict(torch.load(model_path))
                logger.info(f"Successfully loaded model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return False
                
            # Set model to eval mode
            self.model.eval()
            
            # Run system health check
            if not self.check_health():
                logger.error("System health check failed")
                return False
                
            # Monitor initial performance
            performance = self.monitor_performance()
            if performance['error_rate'] > self.config.get('deployment', {}).get('error_threshold', 0.1):
                logger.error("Initial performance check failed")
                return False
                
            logger.info("Model deployment successful")
            return True
            
        except Exception as e:
            logger.error(f"Error during model deployment: {e}")
            return False

    def raise_alert(self, alert_type: str, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Raise an alert for the monitoring system."""
        alert = {
            'type': alert_type,
            'timestamp': datetime.now().isoformat(),
            'data': alert_data,
            'severity': self._determine_severity(alert_type, alert_data)
        }

        # Log alert if not in test mode
        if not self.config.get('test_mode', False):
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({'alert': alert})
            except (ImportError, wandb.Error):
                pass

        return alert

    def _determine_severity(self, alert_type: str, alert_data: Dict[str, Any]) -> str:
        """Determine the severity of an alert."""
        severity_mapping = {
            'ethics_violation': 'critical',
            'model_error': 'high',
            'performance_degradation': 'medium',
            'resource_warning': 'low'
        }
        return severity_mapping.get(alert_type, 'medium')

if __name__ == "__main__":
    # Test deployment monitor functionality
    monitor = DeploymentMonitor({'test_mode': True}, None)
    # Add test cases here
