from collections import namedtuple

from contextlib import AbstractContextManager
from multiprocessing import Lock

import pytest
from typing import List, Any

import numpy as np

from generic_cell_rate_algorithm import throttle
from generic_cell_rate_algorithm.throttle import RateLimit


def _at_least_1d(x: Any) -> List[Any]:
    """Ensure that x is a list"""
    if not isinstance(x, (list, set, tuple)):
        return [x]
    elif isinstance(x, (set, tuple)):
        return list(x)
    return x


class TimeWarp(throttle.Time):
    """Time without all the waiting."""

    def __init__(self):
        self.clock = 0.0

    def time(self) -> float:
        """return the current clock time"""
        return self.clock

    def sleep(self, seconds: float) -> None:
        """Simulate sleeping by moving the clock forward"""
        if seconds < 0:
            raise ValueError("sleep length must be non-negative")
        self.clock += seconds


@pytest.fixture
def time():
    return TimeWarp()


class MockEndpoint:
    """Create a mock endpoint that enforces one or more rate limit policies"""

    def __init__(
        self,
        time: TimeWarp,
        rate_limit: RateLimit | List[RateLimit],
        fixed_period=False,
    ):
        self.log = list()
        self.time = time
        self.rate_limit = _at_least_1d(rate_limit)
        self.fixed_period = fixed_period
        self.period_log = {rate_limit: dict() for rate_limit in self.rate_limit}

    def __call__(self, *args, **kwargs):
        now = self.time.time()
        self.log.append(now)
        if self.fixed_period:
            self.verify_fixed_period(now, self.rate_limit)
        else:
            self.verify_rolling_period(self.rate_limit)

    def verify_fixed_period(self, now, rate_limits):
        """Verify that if the rate limit usage is reset at fixed intervals that we don't exceed the allowed count"""

        for rate_limit in rate_limits:
            period = int(now // rate_limit.period)
            count = self.period_log[rate_limit].get(period, 0)
            count += 1
            self.period_log[rate_limit][period] = count
            rate_limit.usage = count

            if rate_limit.usage > rate_limit.count:
                remaining = (period + 1) * rate_limit.period - now
                raise ResourceWarning(
                    f"Exceeded requests {rate_limit.usage} of {rate_limit.count} "
                    f"over fixed {rate_limit.period}s with {remaining}s remaining"
                )

    def verify_rolling_period(self, rate_limits):
        """Verify that in the last period we have not exceeded the count for each rate limit"""
        for rate_limit in rate_limits:
            log = np.array(self.log)
            start_of_period = log[-1] - rate_limit.period
            in_period = log > start_of_period
            n_in_period = np.sum(in_period)
            if n_in_period > rate_limit.count:
                raise ResourceWarning(
                    f"Exceeded requests {n_in_period} of {rate_limit.count} over rolling {rate_limit.period}"
                    f", {log[in_period][0]} to {log[in_period][-1]}={log[in_period][-1] - log[in_period][0]}"
                )


throttle_state_row = namedtuple(
    "throttle_state_row", ["id", "level", "tat", "allocation"]
)


class ThrottleStateInterface(throttle.ThrottleStateIO):
    """Create a interface to a mock DB for throttle states"""

    def __init__(self, throttle_state: throttle_state_row | List[throttle_state_row]):
        if isinstance(throttle_state, throttle_state_row):
            throttle_state = [throttle_state]

        self._db = {state.id: state for state in throttle_state}

    def read(self) -> List[throttle.ThrottleState]:
        """Read from our mock DB and provide ThrottleStates"""
        result = list()
        for row in self._db.values():
            throttle_state = throttle.ThrottleState(
                row.tat, row.level, row.allocation, id=row.id
            )
            result.append(throttle_state)

        return result

    def write(self, previous: throttle.ThrottleState, new: throttle.ThrottleState):
        """Write the new throttle state into the DB"""
        self._db[previous.id] = throttle_state_row(**new.__dict__)

    def transaction_context(self) -> AbstractContextManager:
        """Provide a transaction context for this throttle state"""
        return Lock()


class RateLimitInterface(throttle.RateLimitIO):
    """Mock an interface to a DB that holds information on our Rate Limits"""

    def __init__(self, rate_limits: throttle.RateLimit | List[throttle.RateLimit]):
        self.rate_limits = rate_limits

    def read(self) -> throttle.RateLimit | List[throttle.RateLimit]:
        return self.rate_limits

    def get_max_count(self):
        if isinstance(self.rate_limits, throttle.RateLimit):
            return self.rate_limits.count
        else:
            return max(self.rate_limits, key=lambda rate_limit: rate_limit.count).count
