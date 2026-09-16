"""Optional, additive wall-clock measurements for query stages."""

import time
from contextlib import contextmanager


@contextmanager
def stage_time(timings: dict[str, float] | None, name: str):
    if timings is None:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - start
