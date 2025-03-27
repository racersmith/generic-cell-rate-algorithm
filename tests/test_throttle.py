from generic_cell_rate_algorithm import throttle

import pytest

from tests.conftest import ThrottleStateInterface, RateLimitInterface, throttle_state_row

from tests.conftest import System




class TestRateLimit:
    def test_read_write(self):
        rate_limit = throttle.RateLimit(count=10, period=12)

        with pytest.raises(AttributeError):
            rate_limit.count = 9

        with pytest.raises(AttributeError):
            rate_limit.period = 99

        try:
            rate_limit.usage = 1
            assert rate_limit.usage == 1

            rate_limit.usage += 1
            assert rate_limit.usage == 2
        except AttributeError:
            pytest.fail("usage should be writable")

    def test_rate(self):
        count = 10
        period = 12
        rate_limit = throttle.RateLimit(count=count, period=period)
        assert rate_limit.rate == count/period

    def test_inverse(self):
        count = 10
        period = 12
        rate_limit = throttle.RateLimit(count=count, period=period)
        assert rate_limit.inverse == period / count

        rate_limit = throttle.RateLimit(count=12, period=23)
        assert abs(rate_limit.inverse - 23/12) < 1/throttle.TIME_RESOLUTION

    def test_count_remaining(self):
        count = 10
        period = 12
        rate_limit = throttle.RateLimit(count=count, period=period)
        assert rate_limit.count_remaining == count
        usage = 5
        rate_limit.usage = 5
        assert rate_limit.count_remaining == count - usage


class TestThrottleState:
    def test_compare(self):
        a = throttle.ThrottleState(level=0, tat=123)
        b = throttle.ThrottleState(level=1, tat=111)
        assert b < a

    def test_new(self):
        b = throttle.ThrottleState(level=1, tat=111)
        b_copy = b._new(tat=123)
        for name, value in b.__dict__.items():
            if name == 'tat':
                assert b.tat == 111
                assert b_copy.tat == 123
            else:
                assert b.__getattribute__(name) == b_copy.__getattribute__(name)

    def test_sort(self):
        states = [
            throttle.ThrottleState(level=0, tat=101),
            throttle.ThrottleState(level=1, tat=100),
            throttle.ThrottleState(level=2, tat=102),
            throttle.ThrottleState(level=3, tat=103),
        ]
        states.sort()
        assert states[0].tat == 100
        assert states[1].tat == 101
        assert states[2].tat == 102
        assert states[3].tat == 103


class TestGcraMethods:
    def __init__(self, time):
        rate_limit_io = RateLimitInterface()

        throttle_states = [throttle_state_row(0, 0, 0, 1)]
        throttle_state_io = ThrottleStateInterface(throttle_states)

        self.gcra = throttle.GCRA(rate_limit_io, throttle_state_io, time)

    def test_filter(self):
        states = [
            throttle.ThrottleState(level=0, tat=1, allocation=1, id=0),
            throttle.ThrottleState(level=1, tat=3, allocation=1, id=1),
            throttle.ThrottleState(level=2, tat=2, allocation=1, id=2),
            throttle.ThrottleState(level=3, tat=0, allocation=1, id=3)

        ]

        assert self.gcra._filter_throttle_states(states, level=0) == states[0]
        assert self.gcra._filter_throttle_states(states, level=1) == states[0]
        assert self.gcra._filter_throttle_states(states, level=2) == states[2]
        assert self.gcra._filter_throttle_states(states, level=3) == states[3]

    def test_sum(self):
        states = [
            throttle_state_row(level=0, tat=1, allocation=1, id=0),
        ]
        assert self.gcra._sum_allocations(states) == 1

        states = [
            throttle.ThrottleState(level=0, tat=1, allocation=1, id=0),
            throttle.ThrottleState(level=1, tat=3, allocation=10, id=1),
            throttle.ThrottleState(level=2, tat=2, allocation=3, id=2),
            throttle.ThrottleState(level=3, tat=0, allocation=1, id=3)

        ]
        assert self.gcra._sum_allocations(states) == 15

    def test_calculate_rate_limit(self):
        state = throttle.ThrottleState(level=3, tat=0, allocation=100, id=3)
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


def test_simple_single_level(time, single_rate_limit_io, throttle_state_io):
    rate_limit_io = RateLimitInterface()

    throttle_states = [throttle_state_row(level=0, tat=10, allocation=1, id=0)]
    throttle_state_io = ThrottleStateInterface(throttle_states)

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

    for _ in range(1000):
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
    assert rate >= 0.95*limit, f"throttled at {rate:.3f}/second with limit at {limit:.3f}/second"



def test_multi_level_high_priority(time):
    rate_limit_io = RateLimitInterface()

    throttle_states = [throttle_state_row(i, i, 0, 10-i) for i in range(10)]
    throttle_state_io = ThrottleStateInterface(throttle_states)

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
    assert rate >= 0.95*limit, f"throttled at {rate:.3f}/second with limit at {limit:.3f}/second"


def test_priority_levels(time):
    rate_limit_io = RateLimitInterface()

    throttle_states = [
        throttle_state_row(level=0, tat=0, allocation=1, id=0),
        throttle_state_row(level=1, tat=0, allocation=1, id=1),
        throttle_state_row(level=2, tat=0, allocation=1, id=2),
        throttle_state_row(level=3, tat=0, allocation=1, id=3),
    ]
    throttle_state_io = ThrottleStateInterface(throttle_states)

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
    assert throttle_state_io._db[0].tat > 0
    assert throttle_state_io._db[1].tat == 0
    assert throttle_state_io._db[2].tat == 0
    assert throttle_state_io._db[3].tat == 0

    time.sleep(1)
    fn(level=1)
    assert throttle_state_io._db[1].tat == throttle_state_io._db[0].tat + 1 > 0

    for _ in range(1000):
        fn()

    dt = time.time()
    count = len(fn.log) - 1
    rate_limit = rate_limit_io.read()
    rate = count/dt
    limit = rate_limit.count/rate_limit.period
    assert rate <= limit, f"throttled at {rate:.3f}/second with limit at {limit:.3f}/second"
    assert rate >= 0.95*limit, f"throttled at {rate:.3f}/second with limit at {limit:.3f}/second"


def test_excessive_wait(time):
    rate_limit_io = RateLimitInterface()

    throttle_states = [throttle_state_row(level=0, tat=1e6, id=0, allocation=1)]
    throttle_state_io = ThrottleStateInterface(throttle_states)

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

class TestGCRA:
    def test_simple(self, single_rate_limit_io, throttle_state_io):
        system = System(single_rate_limit_io, throttle_state_io, fixed_period=False)

        n = 6500
        try:
            for _ in range(n):
                system.rate_limited_call(level=0)

            assert len(system.endpoint._requests) == n
        except ResourceWarning as e:
            pytest.fail(f"Limiting was not successful: {e}")

    def test_single_rate_priority(self, single_rate_limit_io, priority_throttle_state_io):
        system = System(single_rate_limit_io, priority_throttle_state_io, fixed_period=False)

        n = 6500
        try:
            for _ in range(n):
                system.rate_limited_call(level=0)

            assert len(system.endpoint._requests) == n
        except ResourceWarning as e:
            pytest.fail(f"Limiting was not successful: {e}")

        states = priority_throttle_state_io.read()
        next_state = min(states)
        assert next_state.level == 1, f"Throttle should have only used the level=0 priority tat"
        assert next_state.tat == 0, f"Throttle should have only used the level=0 priority tat"

        try:
            for _ in range(n):
                system.rate_limited_call(level=1)

            assert len(system.endpoint._requests) == 2*n
        except ResourceWarning as e:
            pytest.fail(f"Limiting was not successful: {e}")

    def test_multi_rate_no_priority(self, multi_rate_limit_io, throttle_state_io):
        system = System(multi_rate_limit_io, throttle_state_io, fixed_period=True)

        n = 6005
        try:
            for _ in range(n):
                system.rate_limited_call(level=0)

            assert len(system.endpoint._requests) == n
        except ResourceWarning as e:
            rl = system.gcra.rate_io.read()
            pytest.fail(f"Limiting was not successful: {e}, {system.time.time()}:{[r.__dict__ for r in rl]}")
