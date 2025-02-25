import logging
import os
from pathlib import Path

def setup_logging(config):
    """Set up logging configuration"""
    # Create results directory if it doesn't exist
    results_dir = Path(config['paths']['results'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(results_dir, 'pipeline.log')),
            logging.StreamHandler()
        ],
        force=True
    )

def validate_paths(config):
    """Validate existence of required paths"""
    required_paths = ['base_data', 'results']
    for path_key in required_paths:
        path = Path(config['paths'][path_key])
        if not path.exists() and path_key != 'results':
            raise FileNotFoundError(f"Required path {path_key} ({path}) does not exist")
        elif path_key == 'results':
            path.mkdir(parents=True, exist_ok=True)

# import logging
# import os

# def setup_logging(config):
#     """Set up logging configuration"""
#     # Create results directory if it doesn't exist
#     os.makedirs(config['paths']['results'], exist_ok=True)
    
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(os.path.join(config['paths']['results'], 'pipeline.log')),
#             logging.StreamHandler()
#         ]
#     )