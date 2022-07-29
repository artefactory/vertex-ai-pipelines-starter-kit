import logging
import time
from functools import wraps
from typing import Any, Callable

import pandas as pd  # type: ignore

log = logging.getLogger()


def timeit(method: Callable) -> Callable:
    @wraps(method)
    def timed(*args, **kw) -> Any:
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        log.info("%r: time %2.2fs", method.__name__, (te - ts))
        return result

    return timed


def shapeit(method: Callable) -> Callable:
    @wraps(method)
    def wrapped(*args, **kw) -> Any:
        result = method(*args, **kw)
        if isinstance(result, pd.DataFrame):
            log.info("%r: shape %s", method.__name__, result.shape)
        return result

    return wrapped
