import os
import time
import logging
import functools
from typing import Callable

def setup_logging(name: str = "souli") -> logging.Logger:
    level = os.getenv("SOULI_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def timed(label: str = None):
    """
    Decorator that logs how long a function takes.

    Usage:
        @timed("stt.transcribe_bytes")
        def transcribe_bytes(self, ...):
            ...

    Output in logs:
        [TIMER] stt.transcribe_bytes → 1243ms
    """
    def decorator(fn: Callable) -> Callable:
        _label = label or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                logging.getLogger("souli.timer").info(
                    "[TIMER] %-45s → %dms", _label, elapsed_ms
                )

        return wrapper
    return decorator