import os
import logging

def setup_logging(name: str = "souli") -> logging.Logger:
    level = os.getenv("SOULI_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
