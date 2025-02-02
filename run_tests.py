import os
import sys
import pytest
import logging
from core.utils import setup_logging

def main():
    """Run all tests for the LAM system."""
    # Set up logging
    setup_logging('logs')
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Running LAM system tests...")
        
        # Run tests with pytest
        result = pytest.main([
            'tests',
            '-v',
            '--tb=short',
            '--cov=core',
            '--cov=agents',
            '--cov-report=term-missing',
            '--cov-report=html:coverage_report'
        ])
        
        if result == 0:
            logger.info("All tests passed successfully!")
        else:
            logger.error("Some tests failed. Check the output above for details.")
            
    except Exception as e:
        logger.error(f"An error occurred while running tests: {str(e)}")
        raise

if __name__ == "__main__":
    main()
