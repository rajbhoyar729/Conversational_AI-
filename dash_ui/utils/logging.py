import logging
import sys
from typing import Optional

def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Configure application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger instance
    """
    log_level = getattr(logging, level.upper(), logging.INFO) if level else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Return logger for the application
    return logging.getLogger("dash_ui")
