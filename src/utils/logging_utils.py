import logging
import json
from datetime import datetime
from pathlib import Path
import sys

def setup_logger(log_dir: str = None) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    # Create logger
    logger = logging.getLogger('manipulation_uncertainty')
    logger.setLevel(logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if log_dir:
        # Only log to file if log_dir is provided
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"experiment_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Set console handler to WARNING level to reduce spam
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(detailed_formatter)
        console_handler.setLevel(logging.WARNING)
        logger.addHandler(console_handler)
    
    return logger

def log_dict(logger: logging.Logger, data: dict, prefix: str = ""):
    """Log dictionary in a readable format, removing sensitive or redundant info"""
    # Remove sensitive or redundant information
    clean_data = data.copy()
    if 'image_url' in clean_data:
        del clean_data['image_url']
    if 'request' in clean_data:
        clean_data['request'] = {k: v for k, v in clean_data['request'].items() 
                               if k not in ['image_url', 'base64_image']}
    
    logger.info(f"{prefix}{json.dumps(clean_data, indent=2)}") 