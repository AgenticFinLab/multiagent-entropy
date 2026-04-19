import logging

try:
    import aiohttp
    import backoff
    ASYNC_TOOLS_AVAILABLE = True
except ImportError:
    ASYNC_TOOLS_AVAILABLE = False
    backoff = None
    aiohttp = None

logger = logging.getLogger(__name__)


def is_429(exception) -> bool:
    is429 = (
        isinstance(exception, aiohttp.ClientResponseError)
        and exception.status == 429
    ) if ASYNC_TOOLS_AVAILABLE else False
    if is429:
        logger.warning(f"429 rate limit error: {exception}")
    return is429


def is_retriable_http_error(exception) -> bool:
    if ASYNC_TOOLS_AVAILABLE and isinstance(exception, aiohttp.ClientResponseError):
        return exception.status in (429, 403)
    error_str = str(exception)
    return "429" in error_str or "403" in error_str


def retry_on_429(func):
    if backoff is None:
        return func

    @backoff.on_exception(
        backoff.expo,
        aiohttp.ClientResponseError,
        max_tries=8,
        base=2,
        factor=3,
        jitter=backoff.full_jitter,
        giveup=lambda e: not is_429(e),
    )
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


def retry_on_retriable(func):
    if backoff is None:
        return func

    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientResponseError, Exception),
        max_tries=5,
        base=2,
        factor=2,
        jitter=backoff.full_jitter,
        giveup=lambda e: not is_retriable_http_error(e),
    )
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper
