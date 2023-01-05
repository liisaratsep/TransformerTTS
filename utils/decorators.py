import logging
from time import time

logger = logging.getLogger(__name__)


def ignore_exception(f):
    def apply_func(*args, **kwargs):
        # noinspection PyBroadException
        try:
            result = f(*args, **kwargs)
            return result
        except Exception:
            logger.exception(f'Caught exception in {f}')
            return None

    return apply_func


def time_it(f):
    def apply_func(*args, **kwargs):
        t_start = time()
        result = f(*args, **kwargs)
        t_end = time()
        dur = round(t_end - t_start, ndigits=2)
        return result, dur

    return apply_func
