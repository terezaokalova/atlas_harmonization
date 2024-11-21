import logging
import os

def setup_logging(config):
    """Set up logging configuration"""
    # Create results directory if it doesn't exist
    os.makedirs(config['paths']['results'], exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config['paths']['results'], 'pipeline.log')),
            logging.StreamHandler()
        ]
    )