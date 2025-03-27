from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext, AbstractContextManager
from typing import Any, List, Protocol, Optional
from dataclasses import dataclass

import math

import time as _time


# How many decimal places does time.time() return?
TIME_RESOLUTION = 1e7


class Time(Protocol): # pragma: nocover
    def time(self) -> float:
        ...

    def sleep(self, seconds: float) -> None:
        ...


class RateLimit:
    def __init__(self, count: int, period: float, usage: int=0):
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
    level: int = 0  # priority level
    allocation: float = 0  # allocation given to this level
    id: Any = None  # for use by end user in write updates

    def __lt__(self, other):
        return self.tat < other.tat

    def _new(self, tat: float):
        data = dict(self.__dict__)
        data['tat'] = tat
        return ThrottleState(**data)


class ThrottleStateIO(ABC):  # pragma: nocover
    """ one or more TAT with priority level and allocation """
    @abstractmethod
    def read(self) -> List[ThrottleState]:
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


class GCRA:
    def __init__(self, rate_io: RateLimitIO, throttle_io: ThrottleStateIO, time: Time = _time):

        self.rate_io = rate_io
        self.throttle_io = throttle_io
        self.time = time

    def _get_rate_limit(self) -> RateLimit:
        """ Get the current rate limit """
        response = self.rate_io.read()

        if isinstance(response, RateLimit):
            return response

        elif isinstance(response, list) and all(isinstance(r, RateLimit) for r in response):
            """ Find the RateLimit with the fewest remaining counts """
            return min(response, key=lambda rl: rl.count_remaining)

        raise ValueError('Unknown response from RateLimitIO.read().  Expected RateLimit or [RateLimit, ...]')

    @staticmethod
    def _filter_throttle_states(throttle_state_list: List[ThrottleState], level: int) -> ThrottleState:
        """ Get the next available TAT that we have access to with the given level """
        return min(filter(lambda throttle_state: throttle_state.level <= level, throttle_state_list))

    @staticmethod
    def _sum_allocations(throttle_stat_list) -> float:
        """ Get the total allocation over all tats for normalization rate limit"""
        return sum(throttle_stat.allocation for throttle_stat in throttle_stat_list)

    @staticmethod
    def _calculate_rate_limit(global_rate_limit: RateLimit, throttle_state: ThrottleState, total_allocation: float) -> RateLimit:
        # We are assuming that there are not Tats with 0 allocation.
        # If that were the case their next TAT would be float('inf')... which is pointless.
        count = int(max(1.0, global_rate_limit.count * throttle_state.allocation/total_allocation))

        return RateLimit(count=count, period=global_rate_limit.period, usage=global_rate_limit.usage)

    @staticmethod
    def _calculate_next_tat(now: float, throttle_state: ThrottleState, rate_limit: RateLimit) -> float:
        max_tat = max(throttle_state.tat, now)
        separation = max_tat - now
        max_interval = rate_limit.period - rate_limit.inverse
        wait_time = separation - max_interval
        expected_execution_time = now + wait_time
        return max(max_tat, expected_execution_time) + rate_limit.inverse

    @contextmanager
    def throttle(self, level: int = 0, allowed_wait: float = float('inf')) -> Any:
        # read rate limit
        # considered a relatively static resource that does not require an in_transaction
        rate_limit = self._get_rate_limit()

        # allow for an IO transaction context manager for the read/write to TatIO
        with self.throttle_io.transaction_context() as _:
            # read tats, starting our transaction
            throttle_state_list = self.throttle_io.read()

            # select the next TAT accessible for our level
            throttle_state = self._filter_throttle_states(throttle_state_list, level)

            now = self.time.time()
            wait_time = throttle_state.tat - now

            if wait_time > allowed_wait:
                # Wait time is too long
                raise ExcessiveWaitTime(wait_time, allowed_wait)

            total_allocation = self._sum_allocations(throttle_state_list)
            level_rate_limit = self._calculate_rate_limit(rate_limit, throttle_state, total_allocation)

            next_tat = self._calculate_next_tat(now, throttle_state, level_rate_limit)
            next_throttle_state = throttle_state._new(next_tat)

            # write the update, ending our transaction
            self.throttle_io.write(throttle_state, next_throttle_state)

        # Sleep until our start time
        self.time.sleep(max(0.0, wait_time))

        # Allow execution of context code block
        yield
