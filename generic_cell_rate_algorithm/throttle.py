from typing import Callable, Any, List, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from contextlib import contextmanager
import datetime
import time
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimit:
    """ Define the rate limit in counts over a period """
    count: int
    period: float

    def __post_init__(self):
        if self.count <= 0:
            raise ValueError("count must be positive")

        if self.period <= 0:
            raise ValueError("period must be positive")

    @property
    def inverse(self) -> float:
        return self.period/self.count

    def export(self) -> dict:
        return {'count': self.count, 'period': self.period}


def create_two_point_rate_limit(short_count: int, short_period: datetime.timedelta,
                                long_count: int, long_period: datetime.timedelta) -> RateLimit:
    """ Create a rate limit for GCRA that tries to achieve the requested short and long term limits.
    Assume that any allowed burst can be achieved in zero time (conservative)
    Then determine what burst is allowed followed by the remaining rate to achieve the long count at long period
    such that the combination of the burst and this reduced rate achieves the short count over the short period.
    """
    if short_count > long_count:
        raise ValueError(f"short_count should be less than long_count: {short_count=} < {long_count=}")

    if short_period > long_period:
        raise ValueError(f"short_period should be less than long_period: {short_period=} < {long_period=}")

    short_seconds = short_period.total_seconds()
    long_seconds = long_period.total_seconds()

    target_rate = (long_count - short_count) / (long_seconds - short_seconds)
    target_seconds = short_count / target_rate
    target_count = max(1, int(short_count - target_rate * short_seconds))

    return RateLimit(count=target_count, period=datetime.timedelta(seconds=target_seconds).total_seconds())


@dataclass
class TAT:
    """ Theoretical Arrival Time """
    tat: float

    def __lt__(self, other):
        return self.tat < other.tat

    def export(self):
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}


@dataclass
class TaTUpdate:
    new_tat: float
    wait_time: float


@dataclass
class State:
    now: float
    tat: TAT
    rate_limit: RateLimit


class RateLimitIO(ABC):
    @abstractmethod
    def read_rate_limit(self) -> RateLimit:
        """ Read the rate limit from your store """
        ...

    @abstractmethod
    def write_rate_limit(self, rate_limit: RateLimit):
        """ write the rate limit to your store """
        ...


class ThrottleDataHandler(RateLimitIO):  # pragma: nocover
    @staticmethod
    def now() -> float:
        """ Override this with your own timestamp method if you please. """
        return time.time()

    @abstractmethod
    def read_tat(self) -> TAT:
        """ Read the throttle state from your store """
        ...

    @abstractmethod
    def write_tat(self, new_tat: TAT):
        """ Write the throttle state to your store """
        ...

    def _get_state(self) -> State:
        """ Don't override this method. """
        state = State(
            now=self.now(),
            tat=self.read_tat(),
            rate_limit=self.read_rate_limit()
        )
        return state


class ThrottleDataHandlerProtocol(Protocol):
    def now(self) -> float:
        ...

    def write_tat(self, new_tat: TAT):
        ...

    def _get_state(self) -> State:
        ...


@dataclass
class PriorityTAT(TAT):
    """ Throttle state for priority throttling
    level: the larger the number the higher the priority
    allocation:  How much of the rate limit can this level consume.
    """

    level: int
    allocation: float


def get_priority_rate_limit(tat_to_use: PriorityTAT, tat_list: List[PriorityTAT], rate_limit: RateLimit) -> RateLimit:
    # sort so we can fudge the largest number when round takes small values to zero
    allocation_sorted_tat_list = sorted(tat_list, key=lambda tat: tat.allocation)

    # Normalize the allocation values so they are fractional
    total_allocation = sum(tat.allocation for tat in allocation_sorted_tat_list)
    fractions = [tat.allocation/total_allocation for tat in allocation_sorted_tat_list]

    # Generate the counts based on the fractions maintaining positive counts
    counts = [max(1, round(rate_limit.count * fraction)) for fraction in fractions]
    counts[-1] = rate_limit.count - sum(counts[:-1])  # fudge the largest value

    # Find the index of the requested
    index = allocation_sorted_tat_list.index(tat_to_use)
    return RateLimit(int(counts[index]), rate_limit.period)


class PriorityDataHandler(RateLimitIO):
    @abstractmethod
    def read_tat_list(self) -> List[PriorityTAT]:
        """ Read the list of priority TATs """
        ...

    @abstractmethod
    def write_tat(self, tat: PriorityTAT, tat_update: TaTUpdate):
        ...

    def _get_state(self, level: int=0) -> State:
        # request the list of states from the user implemented function
        tat_list = self.read_tat_list()

        # Get the earliest TAT available for this priority level
        tat_in_use = min(filter(lambda tat: tat.level <= level, tat_list))

        rate_limit = self.read_rate_limit()
        priority_rate_limit = get_priority_rate_limit(tat_in_use, tat_list, rate_limit)
        return State(
            now=self.now(),
            tat=tat_in_use,
            rate_limit=priority_rate_limit,
        )



def update_tat(now: float, tat: float, rate_limit: RateLimit) -> TaTUpdate:
    """ Standalone function to be used by TAT Store in its update method.
    This allows us to place the anvil transaction in the same place as the gcra creation.
    The simplification to this module is large at the expense of a small increase in setup work in the modules where it
    is utilized.

    Args:
        now (float): The current time.
        tat (float): The current TAT value in the same units as now.
        rate_limit (RateLimit): The requested rate limit.

    Returns:
        TaTUpdate: The new TAT value and the required wait time for the call.
        the new time must be set by the TATStore!!!

    Examples:

        class TATStore(TatStoreProtocol):
            def __init__(self, rate_limit: RateLimit, now: Callable):
                self.tat = 0

                self.rate_limit = rate_limit
                self.now = now

            def _get(self):
                return self.tat

            def _set(self, tat):
                self.tat = tat

            def _now(self):
                return self.now()

            def update(self):
                tat_update = throttle.update_tat(self._now(), self._get(), self.rate_limit)
                self._set(tat_update.new_tat)
                return tat_update.wait_time
    """

    tat = max(tat, now)
    separation = tat - now
    max_interval = rate_limit.period_seconds - rate_limit.inverse
    wait_time = separation - max_interval
    expected_execution_time = now + wait_time
    new_tat = max(tat, expected_execution_time) + rate_limit.inverse
    return TaTUpdate(new_tat=new_tat, wait_time=wait_time)


class ExcessiveWaitTime(Exception):
    def __init__(self, current_wait: float, allowed_wait: float):
        self.current_wait = current_wait
        self.allowed_wait = allowed_wait
        msg = (f"Throttle wait time of {self.current_wait:.2f} seconds exceeds "
               f"allowed time of {self.allowed_wait:.2f} seconds")
        super().__init__(msg)


class GCRA:
    """ Generic Cell Rate Algorithm
    This allows for some amount of burst while maintaining the overall rate limits.

    Examples:
        gcra = GCRA(tat_store, time.sleep)

        @gcra.throttle
        def example_api_call(id):
            result = requests.get('somewhere.com', somthing)
            return result
    """
    def __init__(self, tat_store: ThrottleStoreProtocol, sleep_fn: Callable) -> None:
        self.tat_store = tat_store
        self.sleep_fn = sleep_fn

    @contextmanager
    def throttle(self, level: int = 0, allowed_wait: float = float('inf')) -> Any:
        wait_time = self.tat_store.update(level=level)

        if wait_time <= 0:
            # There is no current wait required for the call.
            pass

        elif wait_time > allowed_wait:
            # The wait time has exceeded the allowed time, raising an error
            raise ExcessiveWaitTime(wait_time, allowed_wait)

        else:
            # Call is throttled
            logger.info(f'Throttling call.  Wait time is {datetime.timedelta(seconds=wait_time)}')
            self.sleep_fn(wait_time)

        yield
