from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext, AbstractContextManager
from typing import Any, List, Protocol, Optional
from dataclasses import dataclass

import math

import time as _time

import logging
logger = logging.getLogger(__name__)


# How many decimal places does time.time() return?
TIME_RESOLUTION = 1e7


class Time(Protocol): # pragma: nocover
    def time(self) -> float:
        ...

    def sleep(self, seconds: float) -> None:
        ...


class RateLimit:
    def __init__(self, count: int, period: float, usage: int | None=None):
        self._count = count
        self._period = period
        self.usage = usage

    @property
    def count(self) -> int:
        return self._count

    @property
    def period(self) -> float:
        return self._period

    @property
    def count_remaining(self) -> int:
        """ Negative numbers are possible. """
        if self.usage is None:
            return self.count
        else:
            return self.count - self.usage

    @property
    def inverse(self) -> float:
        """ Account for time resolution and round up and don't allow infinite time, period time should be the max """
        count = max(1, self.count_remaining)
        return math.ceil(TIME_RESOLUTION * self.period/count) / TIME_RESOLUTION

    @property
    def rate(self) -> float:
        """ Requests per 'second' """
        return self.count_remaining/self.period


class RateLimitIO(ABC):  # pragma: nocover
    """ Overall Rate limit """
    @abstractmethod
    def read(self) -> RateLimit | List[RateLimit]:
        ...


@dataclass(frozen=True)
class ThrottleState:
    tat: float  # Theoretical arrival time
    level: int | None = None  # priority level
    allocation: float | None = None  # allocation given to this level
    id: Any = None  # for use by end user in write updates

    def __lt__(self, other):
        # allow for comparisons of tat times.
        return self.tat < other.tat

    def new(self, tat: float):
        # duplicate the throttle state and update the tat.
        data = dict(self.__dict__)
        data['tat'] = tat
        return ThrottleState(**data)


class ThrottleStateIO(ABC):  # pragma: nocover
    """ one or more TAT with priority level and allocation """
    @abstractmethod
    def read(self) -> ThrottleState | List[ThrottleState]:
        """ Give a list of all available TATs """
        ...

    @abstractmethod
    def write(self, previous: ThrottleState, new: ThrottleState):
        """ Update the specified TAT it is only necessary to update ThrottleState.tat"""
        ...

    @staticmethod
    def transaction_context() -> AbstractContextManager:
        """ Transaction context manager for the read/write cycle"""
        return nullcontext()


class ExcessiveWaitTime(Exception):
    def __init__(self, current_wait: float, allowed_wait: float):
        self.current_wait = current_wait
        self.allowed_wait = allowed_wait
        msg = (f"Throttle wait time of {self.current_wait:.2f} seconds exceeds "
               f"allowed time of {self.allowed_wait:.2f} seconds")
        super().__init__(msg)


def enforce_single_throttle_state(throttle_state: ThrottleState | list[ThrottleState]):
    if isinstance(throttle_state, ThrottleState):
        return throttle_state

    elif isinstance(throttle_state, list) and len(throttle_state) == 1:
        return throttle_state[0]

    else:
        raise ValueError(f"Expected ThrottleStateIO.read() to return a ThrottleState instance or [ThrottleState]")


def enforce_single_rate_limit(rate_limit: RateLimit | List[RateLimit]):
    if isinstance(rate_limit, RateLimit):
        return rate_limit

    elif isinstance(rate_limit, list) and len(rate_limit) == 1:
        return rate_limit[0]

    else:
        raise ValueError(f"Expected RateLimitIO.read() to return a RateLimit instance or [RateLimit]")


def filter_throttle_states(throttle_state_list: List[ThrottleState], level: int) -> ThrottleState:
    """ Get the next available TAT that we have access to with the given priority level that has allocation """
    try:
        filter_fn = lambda state: state.level <= level and state.allocation > 0
        return min(filter(filter_fn, throttle_state_list))
    except ValueError as e:
        raise LookupError(f"No throttle states found for level: {level}") from e


def sum_allocations(throttle_state_list: List[ThrottleState]) -> float:
    """ Get the total allocation over all tats for normalizing allocations """
    return sum(throttle_stat.allocation for throttle_stat in throttle_state_list)


def normalize_rate_limit(rate_limit: RateLimit, throttle_state: ThrottleState, total_allocation: float) -> RateLimit:
    """ Calculate the rate limit adjusted to the allocation of the selected throttle_state. """

    if throttle_state.allocation == 0:
        raise ValueError(f"throttle_stata has an allocation of zero")

    count = int(max(1.0, rate_limit.count * throttle_state.allocation/total_allocation))

    return RateLimit(count=count, period=rate_limit.period, usage=rate_limit.usage)


def calculate_next_tat(now: float, throttle_state: ThrottleState, rate_limit: RateLimit) -> float:
    """ Get the Theoretical Arrival Time for the next no-wait call. """
    allowed_execution_time = max(throttle_state.tat, now)
    separation_from_tat = allowed_execution_time - now
    max_interval = rate_limit.period - rate_limit.inverse
    wait_time = separation_from_tat - max_interval
    expected_execution_time = now + wait_time
    return max(allowed_execution_time, expected_execution_time) + rate_limit.inverse


class GCRA:
    def __init__(self, rate_io: RateLimitIO, throttle_io: ThrottleStateIO, time: Time = _time):
        self.rate_io = rate_io
        self.throttle_io = throttle_io
        self.time = time

    def get_throttle_state(self) -> ThrottleState:
        throttle_state = self.throttle_io.read()
        return enforce_single_throttle_state(throttle_state)

    def get_rate_limit(self):
        rate_limit = self.rate_io.read()
        return enforce_single_rate_limit(rate_limit)

    @contextmanager
    def throttle(self, allowed_wait: float=float('inf')):
        rate_limit = self.get_rate_limit()

        # allow for an IO transaction context manager for the read/write to TatIO
        with self.throttle_io.transaction_context() as _:
            # Request the throttle states from the user defined reader.  This starts the R/W transaction
            throttle_state = self.get_throttle_state()

            now = self.time.time()
            wait_time = throttle_state.tat - now

            if wait_time > allowed_wait:
                raise ExcessiveWaitTime(wait_time, allowed_wait)

            next_tat = calculate_next_tat(now, throttle_state, rate_limit)
            next_throttle_state = throttle_state.new(next_tat)

            # write the update, ending our transaction
            self.throttle_io.write(throttle_state, next_throttle_state)

        # Sleep until our start time
        self.time.sleep(max(0.0, wait_time))

        # Allow execution of context/wrapped code block
        yield



class GcraPriority(GCRA):
    """ GCRA with multiple throttle states that allow for priority restricted bandwidth
    Throttle states are given levels and when throttling, you can provide the allowed level.

    The TAT with the soonest execution time is used that is at or below the requested level.
    Multiple TATs can be given the same level to allow for a manner of burst capacity that still respects the rate limit.
    Each TAT is given an allocation of the rate limit.  Allocations are automatically normalized to support
    all non-zero values.

    All that said, throttle states must be provided with level and allocation information.
    """
    def get_throttle_state(self) -> ThrottleState:
        ...

    @contextmanager
    def throttle(self, allowed_wait: float=float('inf'), level: int=0):
        ...


class GcraMultiRate(GCRA):
    """ GCRA that respects multiple rate limits simultaneously.
    This allows for specific burst and sustained rate limits for instance.

    To take advantage of burst rate limits, it is required for RateLimitIO.read() to read the rate limit
    with current usage.  Designed for when the API returns usage information that can be utilized.
    Otherwise, this can not provide any benefit, use a simple GCRA with the rate limit set to the
    lowest allowed call rate.
    """

    def get_rate_limit(self):
        """ Get the rate limit that is going to fail first, which is just the rate limit with the
        lowest remaining count
        """
        rate_limits = self.rate_io.read()

        if isinstance(rate_limits, RateLimit):
            logger.warning(f"RateLimitIO.read() only provided a single rate limit. "
                           f"Don't use Multi-Rate, use basic GCRA instead.")
            return rate_limits

        if len(rate_limits) == 0:
            raise ValueError("RateLimitIO.read() didn't return any rate limits.")

        if any([rate_limit.usage is None for rate_limit in rate_limits]):
            raise ValueError("You must provide 'usage' information for rate limits in multi-rate.")

        return min(rate_limits, key=lambda rate_limit: rate_limit.count_remaining)

    def get_throttle_state(self):
        throttle_state = self.throttle_io.read()
        return enforce_single_throttle_state(throttle_state)



class GcraMultiRatePriority(GCRA):
    def get_rate_limit(self) -> RateLimit:
        """ Get the current rate limit """
        response = self.rate_io.read()

        if isinstance(response, RateLimit):
            return response

        elif isinstance(response, list) and all(isinstance(r, RateLimit) for r in response):
            """ Find the RateLimit with the fewest remaining counts """
            return min(response, key=lambda rl: rl.count_remaining)

        raise ValueError('Unknown response from RateLimitIO.read().  Expected RateLimit or [RateLimit, ...]')

    def get_throttle_state(self, level: int) -> (ThrottleState, float):
        """ Get the list of throttle states filter them and get the total allocation """
        throttle_state_list = self.throttle_io.read()
        if isinstance(throttle_state_list, RateLimit):
            throttle_state_list = [throttle_state_list]


        if not isinstance(throttle_state_list, list):
            raise ValueError(f"Unexpected result from ThrottleStateIO.read().  "
                             f"Expected [ThrottleState, ...] not {throttle_state_list}")


        throttle_state = filter_throttle_states(throttle_state_list, level=level)
        total_allocation = sum_allocations(throttle_state_list)
        return throttle_state, total_allocation

    @contextmanager
    def throttle(self, allowed_wait: float = float('inf'), level: int = 0) -> Any:
        # read rate limit
        # considered a relatively static resource that does not require an in_transaction
        rate_limit = self.get_rate_limit()

        # allow for an IO transaction context manager for the read/write to TatIO
        with self.throttle_io.transaction_context() as _:
            # read tats, starting our transaction
            throttle_state, total_allocation = self.throttle_io.read(level=level)

            now = self.time.time()
            wait_time = throttle_state.tat - now

            if wait_time > allowed_wait:
                # Wait time is too long
                raise ExcessiveWaitTime(wait_time, allowed_wait)

            level_rate_limit = normalize_rate_limit(rate_limit, throttle_state, total_allocation)

            next_tat = calculate_next_tat(now, throttle_state, level_rate_limit)
            next_throttle_state = throttle_state.new(next_tat)

            # write the update, ending our transaction
            self.throttle_io.write(throttle_state, next_throttle_state)

        # Sleep until our start time
        self.time.sleep(max(0.0, wait_time))

        # Allow execution of context code block
        yield
