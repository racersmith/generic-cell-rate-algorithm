from collections import namedtuple

from contextlib import AbstractContextManager
from multiprocessing import Lock

import pytest
from typing import List, Any, Callable

from generic_cell_rate_algorithm import throttle
from generic_cell_rate_algorithm.throttle import RateLimit


def _at_least_1d(x: Any) -> List[Any]:
    """ Ensure that x is a list """
    if not isinstance(x, (list, set, tuple)):
        return [x]
    elif isinstance(x, (set, tuple)):
        return list(x)
    return x



class TimeWarp(throttle.Time):
    def __init__(self):
        self.clock = 0.0

    def time(self) -> float:
        return self.clock

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("sleep length must be non-negative")
        self.clock += seconds


@pytest.fixture
def time():
    return TimeWarp()


class MockEndpoint:
    def __init__(self, time: TimeWarp, rate_limit: RateLimit | List[RateLimit], fixed_period=False):
        self._requests = list()
        self.time = time
        self.rate_limit = _at_least_1d(rate_limit)
        self.fixed_period = fixed_period

    def __call__(self, *args, **kwargs):
        now = self.time.time()
        self._requests.append(now)
        if self.fixed_period:
            self.verify_fixed_period(now, self.rate_limit)
        else:
            self.verify_rolling_period(self.rate_limit)

    @staticmethod
    def _update_usage(now: float, last_timestamp: float, rate_limit: RateLimit):
        if now // rate_limit.period > last_timestamp // rate_limit.period:
            # The current time is in a different period than the last timestamp.
            rate_limit.usage = 0

        rate_limit.usage += 1

    def verify_fixed_period(self, now, rate_limits):
        last_timestamp = list([self._requests[-1]] + self._requests)[-2]
        for rate_limit in rate_limits:
            self._update_usage(now, last_timestamp, rate_limit)
            if rate_limit.usage > rate_limit.count:
                current_index = now//rate_limit.period
                remaining = (current_index + 1) * rate_limit.period - now
                raise ResourceWarning(
                    f"Exceeded requests {rate_limit.usage} of {rate_limit.count} over fixed {rate_limit.period} {remaining}"
                )

    def verify_rolling_period(self, rate_limits):
        for rate_limit in rate_limits:
            start_of_period = self._requests[-1] - rate_limit.period
            within_period = list(filter(lambda time_stamp: time_stamp > start_of_period, self._requests))
            n = len(within_period)
            if n > rate_limit.count:
                raise ResourceWarning(f"Exceeded requests {n} of {rate_limit.count} over rolling {rate_limit.period}")


class PriorityFunction:
    def __init__(self, gcra: throttle.GCRA, fn: Callable):
        self.gcra = gcra
        self.fn = fn

    def __call__(self, level: int=0, allowed_wait: float=float('inf')):
        @self.gcra.throttle(level=level, allowed_wait=allowed_wait)
        def wrapper(*args, **kwargs):
            return self.fn(*args, **kwargs)
        return wrapper


throttle_state_row = namedtuple("throttle_state_row", ["id", "level", "tat", "allocation"])


class ThrottleStateInterface(throttle.ThrottleStateIO):
    def __init__(self, throttle_state_list: List[throttle_state_row]):
        self._db = {state.id: state for state in throttle_state_list}

    def read(self) -> List[throttle.ThrottleState]:
        result = list()
        for row in self._db.values():
            throttle_state = throttle.ThrottleState(row.tat, row.level, row.allocation, id=row.id)
            result.append(throttle_state)

        return result

    def write(self, previous: throttle.ThrottleState, new: throttle.ThrottleState):
        self._db[previous.id] = throttle_state_row(**new.__dict__)

    def transaction_context(self) -> AbstractContextManager:
        return Lock()


@pytest.fixture
def throttle_state_io():
    states = [throttle_state_row(id=123, level=0, tat=0, allocation=1)]
    return ThrottleStateInterface(states)


@pytest.fixture
def priority_throttle_state_io():
    states = [
        throttle_state_row(id=234, level=0, tat=0, allocation=2),
        throttle_state_row(id=345, level=1, tat=0, allocation=1),
    ]
    return ThrottleStateInterface(states)


class RateLimitInterface(throttle.RateLimitIO):
    def __init__(self, rate_limits: throttle.RateLimit | List[throttle.RateLimit]):
        self.rate_limits = rate_limits

    def read(self) -> throttle.RateLimit | List[throttle.RateLimit]:
        return self.rate_limits


@pytest.fixture
def single_rate_limit_io():
    rate_limits = throttle.RateLimit(count=600, period=900)
    return RateLimitInterface(rate_limits)


@pytest.fixture
def multi_rate_limit_io():
    rate_limits = [
        throttle.RateLimit(count=600, period=900),
        throttle.RateLimit(count=6000, period=86400),
    ]
    return RateLimitInterface(rate_limits)


class System:
    def __init__(self, rate_limit_io, throttle_state_io, fixed_period: bool):
        self.time = TimeWarp()
        self.gcra = throttle.GCRA(
            rate_io=rate_limit_io,
            throttle_io=throttle_state_io,
            time=self.time
        )

        self.fixed_period = fixed_period
        self.endpoint = MockEndpoint(self.time, self.rate_limit, fixed_period=fixed_period)

    @property
    def rate_limit(self):
        return self.gcra.rate_io.read()

    def rate_limited_call(self, level: int=0, allowed_wait: float=float('inf')):
        @self.gcra.throttle(level=level, allowed_wait=allowed_wait)
        def _fn():
            return self.endpoint()

        return _fn()



# @pytest.fixture
# def gcra_simple(single_rate_limit_io, throttle_io, time):
#     return throttle.GCRA(single_rate_limit_io, throttle_io, time)
#
#
# @pytest.fixture
# def gcra_priority(single_rate_limit_io, priority_throttle_state_io, time):
#     return throttle.GCRA(single_rate_limit_io, priority_throttle_state_io, time)
#
#
# def setup(gcra: throttle.GCRA):
#     endpoint = MockEndpoint(gcra, fixed_period=False)
#     fn = PriorityFunction(gcra, endpoint)
