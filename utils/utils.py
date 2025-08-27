import time
import logging
import asyncio
from functools import wraps
import httpx
import requests

logger = logging.getLogger(__name__)

def retry_with_backoff(retries=3, backoff_in_seconds=1, catch_exceptions=(requests.exceptions.RequestException,)):
    """
    A decorator to retry a synchronous function with exponential backoff.
    """
    def rwb(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            _retries, _backoff = retries, backoff_in_seconds
            while _retries > 1:
                try:
                    return f(*args, **kwargs)
                except catch_exceptions as e:
                    logger.warning(f"Function {f.__name__} failed with {e}. Retrying in {_backoff} seconds...")
                    time.sleep(_backoff)
                    _retries -= 1
                    _backoff *= 2
            return f(*args, **kwargs)
        return wrapper
    return rwb

def async_retry_with_backoff(retries=3, backoff_in_seconds=1, catch_exceptions=(httpx.RequestError,)):
    """
    A decorator to retry an asynchronous function with exponential backoff.
    """
    def rwb(f):
        @wraps(f)
        async def wrapper(*args, **kwargs):
            _retries, _backoff = retries, backoff_in_seconds
            while _retries > 1:
                try:
                    return await f(*args, **kwargs)
                except catch_exceptions as e:
                    logger.warning(f"Function {f.__name__} failed with {e}. Retrying in {_backoff} seconds...")
                    await asyncio.sleep(_backoff)
                    _retries -= 1
                    _backoff *= 2
            return await f(*args, **kwargs)
        return wrapper
    return rwb

