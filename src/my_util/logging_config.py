import logging
import os
import sys

def setup_logging(log_filename: str, logger_name: str = "") -> logging.Logger:
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', '%Y-%m-%d %H:%M:%S')

    for handler in [
        logging.FileHandler(log_filename, mode='a', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger