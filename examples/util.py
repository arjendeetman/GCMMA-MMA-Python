"""
GCMMA-MMA-Python

This file is part of GCMMA-MMA-Python. GCMMA-MMA-Python is licensed under the terms of GNU 
General Public License as published by the Free Software Foundation. For more information and 
the LICENSE file, see <https://github.com/arjendeetman/GCMMA-MMA-Python>. 
"""

# Loading modules
import logging

# Setup logger
def setup_logger(logfile):
    
    """
    Sets up a logger with both console and file handlers.

    Args:
        logfile (str): The path to the log file where logs will be written.

    Returns:
        logging.Logger: Configured logger instance.
    """
        
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    
    # Create file handler and set level to debug
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    
    # Add formatter to ch and fh
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # Add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    
    # Open logfile and reset
    with open(logfile, 'w'): pass
    
    # Return logger
    return logger