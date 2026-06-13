import time
import random
import logging
import functools
from typing import Any, Callable, Tuple, Type

logger = logging.getLogger("ResilienceUtilities")

def retry_with_backoff(
    retries: int = 3, 
    initial_delay: float = 1.0, 
    backoff_factor: float = 2.0, 
    exceptions: Tuple[Type[BaseException], ...] = (Exception,)
):
    """
    Decorator for adding exponential backoff with full random jitter to API requests.
    Prevents simultaneous client retries from hammering downstream model nodes.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == retries:
                        logger.error(f"Execution limit reached for {func.__name__}. Propagating error.")
                        raise e
                    
                    # Compute exponential variance with random jitter
                    jittered_delay = delay * random.uniform(0.5, 1.5)
                    logger.warning(
                        f"Transient error in {func.__name__} (Attempt {attempt}/{retries}): {str(e)}. "
                        f"Retrying in {jittered_delay:.2f} seconds..."
                    )
                    time.sleep(jittered_delay)
                    delay *= backoff_factor
        return wrapper
    return decorator