from contextlib import AbstractContextManager
from typing import List
from collections import namedtuple

from multiprocessing import Lock

from generic_cell_rate_algorithm import throttle
from generic_cell_rate_algorithm.util import calculate_two_point_rate

import pytest

class TimeWarp(throttle.Time):
    def __init__(self):
        self.clock = 0.0

    def time(self) -> float:
        return self.clock

    def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("negative sleep time not allowed")
        self.clock += seconds


class RateLimitInterface(throttle.RateLimitIO):
    def __init__(self):

        rate = calculate_two_point_rate(short_count=600, short_period=15*60, long_count=6000, long_period=24*60*60)

        self.rate_limit = throttle.RateLimit(rate.count, rate.period)

    def read(self) -> throttle.RateLimit:
        return self.rate_limit


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



def test_rate_limit_inverse():
    rate_limit = throttle.RateLimit(count=12, period=23)
    assert abs(rate_limit.inverse - 23/12) < 1/throttle.TIME_RESOLUTION



def test_throttle_state():
    a = throttle.ThrottleState(level=0, tat=123)
    b = throttle.ThrottleState(level=1, tat=111)
    assert b < a



class TestThrottleStateInterface:
    def test_read(self):
        throttle_states = [
            throttle_state_row(1234, 0, 0, 1),
            throttle_state_row(2345, 1, 0, 1),
            throttle_state_row(3456, 2, 0, 1),
        ]
        throttle_state_io = ThrottleStateInterface(throttle_states)
        result = throttle_state_io.read()
        assert len(result) == len(throttle_states)
        for state in result:
            assert state.__dict__ == throttle_state_io._db[state.id]._asdict()


    def test_write(self):
        throttle_states = [
            throttle_state_row(1234, 0, 0, 1),
            throttle_state_row(2345, 1, 0, 1),
            throttle_state_row(3456, 2, 0, 1),
        ]
        throttle_state_io = ThrottleStateInterface(throttle_states)
        result = throttle_state_io.read()
        state = result[1]
        new_state = state._new(99)
        throttle_state_io.write(state, new_state)
        assert throttle_state_io._db[state.id]._asdict() == new_state.__dict__


class TestGcraMethods:
    def __init__(self):
        rate_limit_io = RateLimitInterface()

        throttle_states = [throttle_state_row(0, 0, 0, 1)]
        throttle_state_io = ThrottleStateInterface(throttle_states)

        time = TimeWarp()

        self.gcra = throttle.GCRA(rate_limit_io, throttle_state_io, time)

    def test_filter(self):
        states = [
            throttle_state_row(level=0, tat=1, allocation=1, id=0),
            throttle_state_row(level=1, tat=3, allocation=1, id=1),
            throttle_state_row(level=2, tat=2, allocation=1, id=2),
            throttle_state_row(level=3, tat=0, allocation=1, id=3)

        ]

        assert self.gcra._filter_throttle_states(states, 0) == states[0]
        assert self.gcra._filter_throttle_states(states, 1) == states[0]
        assert self.gcra._filter_throttle_states(states, 2) == states[2]
        assert self.gcra._filter_throttle_states(states, 3) == states[3]

    def test_sum(self):
        states = [
            throttle_state_row(level=0, tat=1, allocation=1, id=0),
        ]
        assert self.gcra._sum_allocations(states) == 1

        states = [
            throttle_state_row(level=0, tat=1, allocation=1, id=0),
            throttle_state_row(level=1, tat=3, allocation=10, id=1),
            throttle_state_row(level=2, tat=2, allocation=3, id=2),
            throttle_state_row(level=3, tat=0, allocation=1, id=3)

        ]
        assert self.gcra._sum_allocations(states) == 15

    def test_calculate_rate_limit(self):
        state = throttle_state_row(level=3, tat=0, allocation=100, id=3)
        rate_limit = throttle.RateLimit(count=333, period=10)

        assert self.gcra._calculate_rate_limit(rate_limit, state, total_allocation=1) == 333
        assert self.gcra._calculate_rate_limit(rate_limit, state, total_allocation=10) == 333
        assert self.gcra._calculate_rate_limit(rate_limit, state, total_allocation=200) == 166
        assert self.gcra._calculate_rate_limit(rate_limit, state, total_allocation=300) == 111
        assert self.gcra._calculate_rate_limit(rate_limit, state, total_allocation=1e9) == 1

    def test_calculate_next_tat(self):
        rate_limit = throttle.RateLimit(count=1, period=10)

        now = 10
        state = throttle.ThrottleState(level=3, tat=now)
        assert self.gcra._calculate_next_tat(now, state, rate_limit) == now + rate_limit.inverse

        state = throttle.ThrottleState(level=3, tat=now + 3)
        assert self.gcra._calculate_next_tat(now, state, rate_limit) == now + 3 + rate_limit.inverse

        state = throttle.ThrottleState(level=3, tat=now - 3)
        assert self.gcra._calculate_next_tat(now, state, rate_limit) == now - 3 + rate_limit.inverse


def test_simple_single_level():
    rate_limit_io = RateLimitInterface()

    throttle_states = [throttle_state_row(level=0, tat=10, allocation=1, id=0)]
    throttle_state_io = ThrottleStateInterface(throttle_states)

    time = TimeWarp()

    gcra = throttle.GCRA(rate_limit_io, throttle_state_io, time)

    class ThrottleFn:
        def __init__(self, time: throttle.Time):
            self.time = time
            self.log = list()

        @gcra.throttle()
        def __call__(self, *args, **kwargs):
            self.log.append(self.time.time())
            return None

    fn = ThrottleFn(time)

    for _ in range(543):
        fn()

    calc_limit = gcra._calculate_rate_limit(rate_limit_io.rate_limit, throttle_state_io.read()[0], 1)
    assert calc_limit.count == rate_limit_io.rate_limit.count
    assert calc_limit.period == rate_limit_io.rate_limit.period

    dt = fn.log[-1]
    count = len(fn.log) - 1

    rate_count = rate_limit_io.rate_limit.count
    rate_period = rate_limit_io.rate_limit.period
    limit = rate_count / rate_period
    rate = count/dt
    assert fn.log[-1] - fn.log[-2] >= rate_period/rate_count
    assert rate/limit <= 1, f"{rate/limit}, count={count}/{rate_count}, dt={dt}/{rate_period}"
    assert rate <= limit, f"throttled at {rate:.3f}/second with limit at {limit:.3f}/second"
    assert rate >= 0.99*limit, f"throttled at {rate:.3f}/second with limit at {limit:.3f}/second"



def test_multi_level_high_priority():
    rate_limit_io = RateLimitInterface()

    throttle_states = [throttle_state_row(i, i, 0, 10-i) for i in range(10)]
    throttle_state_io = ThrottleStateInterface(throttle_states)

    time = TimeWarp()

    gcra = throttle.GCRA(rate_limit_io, throttle_state_io, time)

    class ThrottleFn:
        def __init__(self, time: throttle.Time):
            self.time = time
            self.log = list()

        @gcra.throttle(level=len(throttle_states))
        def __call__(self, *args, **kwargs):
            self.log.append(self.time.time())
            return None

    fn = ThrottleFn(time)

    for _ in range(1000):
        fn()

    dt = time.time()
    count = len(fn.log) - 1
    rate_limit = rate_limit_io.read()
    rate = count/dt
    limit = rate_limit.count/rate_limit.period
    assert rate <= limit, f"throttled at {rate:.3f}/second with limit at {limit:.3f}/second"
    assert rate >= 0.99*limit, f"throttled at {rate:.3f}/second with limit at {limit:.3f}/second"

    for state in throttle_state_io._db.values():
        print(state)


def test_priority_levels():
    rate_limit_io = RateLimitInterface()

    throttle_states = [
        throttle_state_row(level=0, tat=0, allocation=1, id=0),
        throttle_state_row(level=1, tat=0, allocation=1, id=1),
        throttle_state_row(level=2, tat=0, allocation=1, id=2),
        throttle_state_row(level=3, tat=0, allocation=1, id=3),
    ]
    throttle_state_io = ThrottleStateInterface(throttle_states)

    time = TimeWarp()

    gcra = throttle.GCRA(rate_limit_io, throttle_state_io, time)

    class ThrottleFn:
        def __init__(self, time: throttle.Time):
            self.time = time
            self.log = list()

        def __call__(self, *args, **kwargs):
            @gcra.throttle(level=kwargs['level'])
            def _fn(*args, **kwargs):
                self.log.append(self.time.time())
                return None
            return _fn(*args, **kwargs)

    fn = ThrottleFn(time)

    fn(level=0)
    print(throttle_state_io._db.values())
    assert throttle_state_io._db[0].tat > 0
    assert throttle_state_io._db[1].tat == 0
    assert throttle_state_io._db[2].tat == 0
    assert throttle_state_io._db[3].tat == 0

    time.sleep(1)
    fn(level=1)
    print(throttle_state_io._db.values())
    assert throttle_state_io._db[1].tat == throttle_state_io._db[0].tat + 1 > 0

    # for _ in range(1000):
    #     fn()

    # dt = time.time()
    # count = len(fn.log) - 1
    # rate_limit = rate_limit_io.read()
    # rate = count/dt
    # limit = rate_limit.count/rate_limit.period
    # assert rate <= limit, f"throttled at {rate:.3f}/second with limit at {limit:.3f}/second"
    # assert rate >= 0.99*limit, f"throttled at {rate:.3f}/second with limit at {limit:.3f}/second"
    #
    # for state in throttle_state_io._db.values():
    #     print(state)


def test_excessive_wait():
    rate_limit_io = RateLimitInterface()

    throttle_states = [throttle_state_row(level=0, tat=1e6, id=0, allocation=1)]
    throttle_state_io = ThrottleStateInterface(throttle_states)

    time = TimeWarp()

    gcra = throttle.GCRA(rate_limit_io, throttle_state_io, time)

    class ThrottleFn:
        def __init__(self, time: throttle.Time):
            self.time = time
            self.log = list()

        @gcra.throttle(allowed_wait=60)
        def __call__(self, *args, **kwargs):
            self.log.append(self.time.time())
            return None

    fn = ThrottleFn(time)

    with pytest.raises(throttle.ExcessiveWaitTime):
        fn()
