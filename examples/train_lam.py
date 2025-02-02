import os
import sys
import logging
from pathlib import Path

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from core.orchestrator import LAMOrchestrator
from config.lam_config import LAM_CONFIG
from core.utils import setup_logging
import wandb

def main():
    """Example script to train a Large Action Model."""
    # Set up logging
    setup_logging('logs')
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize wandb
        wandb.init(
            project=LAM_CONFIG['wandb']['project'],
            entity=LAM_CONFIG['wandb']['entity'],
            config=LAM_CONFIG,
            tags=LAM_CONFIG['wandb']['tags']
        )
        
        # Initialize orchestrator
        orchestrator = LAMOrchestrator()
        
        # Step 1: Initialize development environment
        logger.info("Initializing development environment...")
        success = orchestrator.initialize_development(LAM_CONFIG)
        
        if not success:
            logger.error("Failed to initialize development environment")
            return
            
        # Step 2: Training and evaluation
        logger.info("Starting training and evaluation...")
        training_config = {
            'task_data': [],  # Add your task data here
            'expert_data': [],  # Add your expert demonstrations here
            'environment_data': [],  # Add your environment data here
            'reward_data': []  # Add your reward data here
        }
        
        success = orchestrator.train_and_evaluate(training_config)
        
        if not success:
            logger.error("Training and evaluation failed")
            return
            
        # Step 3: Deploy and monitor
        logger.info("Deploying and setting up monitoring...")
        deployment_config = {
            'model_path': os.path.join('models', 'lam_model'),
            'learning_config': LAM_CONFIG['training']
        }
        
        success = orchestrator.deploy_and_monitor(deployment_config)
        
        if not success:
            logger.error("Deployment failed")
            return
            
        # Step 4: Monitor system status
        logger.info("Monitoring system status...")
        status = orchestrator.get_system_status()
        
        logger.info("System status:")
        logger.info(f"Deployment status: {status['deployment_status']}")
        logger.info(f"Ethical compliance: {status['ethical_compliance']}")
        logger.info(f"Performance metrics: {status['performance_metrics']}")
        
        # Log final metrics to wandb
        wandb.log({
            'final_status': status
        })
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
        
    finally:
        # Close wandb run
        wandb.finish()

if __name__ == "__main__":
    main()
