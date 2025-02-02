import os
import logging
import argparse
from dotenv import load_dotenv
from core.orchestrator import LAMOrchestrator
from config.lam_config import LAM_CONFIG

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Large Action Model (LAM) Development System')
    parser.add_argument('--mode', type=str, default='train',
                      choices=['train', 'evaluate', 'deploy'],
                      help='Operation mode: train, evaluate, or deploy')
    parser.add_argument('--config', type=str, default='config/lam_config.py',
                      help='Path to configuration file')
    return parser.parse_args()

def main():
    """Main entry point for the LAM system."""
    args = parse_arguments()
    
    try:
        # Initialize orchestrator
        orchestrator = LAMOrchestrator()
        
        if args.mode == 'train':
            logger.info("Starting LAM training process")
            
            # Initialize development environment
            orchestrator.initialize_development(LAM_CONFIG)
            
            # Execute training pipeline
            success = orchestrator.train_and_evaluate(LAM_CONFIG['training'])
            
            if success:
                logger.info("Training completed successfully")
            else:
                logger.error("Training failed")
                
        elif args.mode == 'evaluate':
            logger.info("Starting LAM evaluation")
            
            # Get system status and evaluation metrics
            status = orchestrator.get_system_status()
            logger.info(f"System status: {status}")
            
        elif args.mode == 'deploy':
            logger.info("Starting LAM deployment")
            
            # Deploy and monitor the system
            success = orchestrator.deploy_and_monitor(LAM_CONFIG['deployment'])
            
            if success:
                logger.info("Deployment completed successfully")
                
                # Start monitoring
                status = orchestrator.get_system_status()
                logger.info(f"Initial system status: {status}")
            else:
                logger.error("Deployment failed")
                
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
