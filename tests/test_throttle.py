from generic_cell_rate_algorithm import throttle

import pytest

from tests.conftest import ThrottleStateInterface, RateLimitInterface, throttle_state_row, MockEndpoint


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
        b_copy = b.new(tat=123)
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
    # def __init__(self, time):
    #     rate_limit_io = RateLimitInterface()
    #
    #     throttle_states = [throttle_state_row(0, 0, 0, 1)]
    #     throttle_state_io = ThrottleStateInterface(throttle_states)
    #
    #     self.gcra = throttle.GCRA(rate_limit_io, throttle_state_io, time)

    def test_filter_throttle_states(self):
        states = [
            throttle.ThrottleState(level=0, tat=1, allocation=0, id=0),
            throttle.ThrottleState(level=1, tat=3, allocation=1, id=1),
            throttle.ThrottleState(level=2, tat=2, allocation=1, id=2),
            throttle.ThrottleState(level=3, tat=0, allocation=1, id=3),
            throttle.ThrottleState(level=4, tat=0, allocation=0, id=3)
        ]

        assert throttle.filter_throttle_states(states, level=1) == states[1]
        assert throttle.filter_throttle_states(states, level=2) == states[2]
        assert throttle.filter_throttle_states(states, level=3) == states[3]
        assert throttle.filter_throttle_states(states, level=4) == states[3]

        with pytest.raises(LookupError):
            throttle.filter_throttle_states(states, level=0)

    def test_sum_allocations(self):
        states = [
            throttle_state_row(level=0, tat=1, allocation=1, id=0),
        ]
        assert throttle.sum_allocations(states) == 1

        states = [
            throttle.ThrottleState(level=0, tat=1, allocation=1, id=0),
            throttle.ThrottleState(level=1, tat=3, allocation=10, id=1),
            throttle.ThrottleState(level=2, tat=2, allocation=3, id=2),
            throttle.ThrottleState(level=3, tat=0, allocation=1, id=3)

        ]
        assert throttle.sum_allocations(states) == 15

    def test_normalize_rate_limit(self):
        state = throttle.ThrottleState(level=3, tat=0, allocation=100, id=3)
        rate_limit = throttle.RateLimit(count=333, period=10)

        assert throttle.normalize_rate_limit(rate_limit, state, total_allocation=100).count == 333
        assert throttle.normalize_rate_limit(rate_limit, state, total_allocation=150).count == 222
        assert throttle.normalize_rate_limit(rate_limit, state, total_allocation=200).count == 166
        assert throttle.normalize_rate_limit(rate_limit, state, total_allocation=300).count == 111
        assert throttle.normalize_rate_limit(rate_limit, state, total_allocation=1e9).count == 1

        with pytest.raises(ValueError):
            throttle.normalize_rate_limit(
                rate_limit,
                throttle.ThrottleState(level=3, tat=0, allocation=0, id=3),
                total_allocation=1e9)

    def test_calculate_next_tat(self):
        rate_limit = throttle.RateLimit(count=1, period=10)

        now = 10
        state = throttle.ThrottleState(tat=now)
        assert throttle.calculate_next_tat(now, state, rate_limit) == now + rate_limit.inverse

        state = throttle.ThrottleState(tat=now + 3)
        assert throttle.calculate_next_tat(now, state, rate_limit) == now + 3 + rate_limit.inverse

        state = throttle.ThrottleState(tat=now - 3)
        assert throttle.calculate_next_tat(now, state, rate_limit) == now + rate_limit.inverse


class TestGCRA:
    def test_rolling_period(self,  time):
        # mock data
        rate_limit = throttle.RateLimit(count=600, period=900)
        throttle_state = throttle_state_row(id=0, level=None, tat=0, allocation=None)

        # Build interface to mock db
        rate_io = RateLimitInterface(rate_limit)
        throttle_io = ThrottleStateInterface(throttle_state)

        # Initialize our GCRA throttler
        gcra = throttle.GCRA(
            rate_io=rate_io,
            throttle_io=throttle_io,
            time=time
        )

        # Mock our test endpoint which enforces the rate limit policy provided
        endpoint = MockEndpoint(time, rate_limit, fixed_period=False)

        # Mock our throttled api call function
        @gcra.throttle()
        def api_request():
            endpoint()
            return None

        # Run the test
        try:
            for _ in range(2*rate_io.get_max_count() + 5):
                api_request()
        except ResourceWarning:
            pytest.fail(f"Did not correctly limit function")

        average_rate = (len(endpoint.log)-1)/(endpoint.log[-1]-endpoint.log[0])
        assert average_rate <= rate_limit.rate, f"Throttled rate too high."
        assert average_rate > 0.95 * rate_limit.rate, f"Throttled rate too low."

        current_wait_time = gcra._get_throttle_state().tat - time.time()
        assert current_wait_time > 0

        @gcra.throttle(allowed_wait=0.9*current_wait_time)
        def impatient_request():
            return None

        with pytest.raises(throttle.ExcessiveWaitTime):
            impatient_request()


class TestGcraMultiRate:
    def test_normal(self, time):
        # mock data
        burst_rate = throttle.RateLimit(count=600, period=900, usage=0)
        sustained_rate = throttle.RateLimit(count=6000, period=86400, usage=0)
        rate_limit = [burst_rate, sustained_rate]
        throttle_state = throttle_state_row(id=0, level=None, tat=0, allocation=None)

        # Build interface to mock db
        rate_io = RateLimitInterface(rate_limit)
        throttle_io = ThrottleStateInterface(throttle_state)

        # Initialize our GCRA throttler
        gcra = throttle.GcraMultiRate(
            rate_io=rate_io,
            throttle_io=throttle_io,
            time=time
        )

        # Mock our test endpoint which enforces the rate limit policy provided
        endpoint = MockEndpoint(time, rate_limit, fixed_period=True)

        # Mock our throttled api call function
        @gcra.throttle()
        def api_request():
            endpoint()
            return None

        # Run the test
        try:
            for _ in range(2*rate_io.get_max_count() + 5):
                api_request()
        except ResourceWarning:
            pytest.fail(f"Did not correctly limit function")

        for limit, period_log in endpoint.period_log.items():
            for i, count in period_log.items():
                assert count < limit.count, f"Exceeded limit of {limit.count}/{limit.period} in period: {i} with {count} calls"
                assert count > 0, f"We didn't make any calls in period: {i}"

        # check that we are getting burst rates greater than sustained rates.
        burst_period_log = endpoint.period_log[burst_rate]
        count = burst_period_log[0]
        dt = endpoint.log[count] - endpoint.log[0]
        assert count/dt > sustained_rate.rate, f"Throttle not allowing burst rates."

        current_wait_time = gcra._get_throttle_state().tat - time.time()
        assert current_wait_time > 0

        @gcra.throttle(allowed_wait=0.9*current_wait_time)
        def impatient_request():
            return None

        with pytest.raises(throttle.ExcessiveWaitTime):
            impatient_request()

    # def test_single_rate_priority(self, single_rate_limit_io, priority_throttle_state_io):
    #     system = System(single_rate_limit_io, priority_throttle_state_io, fixed_period=False)
    #
    #     n = system.rate_limit.count + 25
    #
    #     try:
    #         for _ in range(n):
    #             system.rate_limited_call(level=0)
    #
    #         assert len(system.endpoint.log) == n
    #     except ResourceWarning as e:
    #         pytest.fail(f"Limiting was not successful: {e}")
    #
    #     states = priority_throttle_state_io.read()
    #     next_state = min(states)
    #     assert next_state.level == 1, f"Throttle should have only used the level=0 priority tat"
    #     assert next_state.tat == 0, f"Throttle should have only used the level=0 priority tat"
    #
    #     try:
    #         for _ in range(n):
    #             system.rate_limited_call(level=1)
    #
    #         assert len(system.endpoint.log) == 2 * n
    #     except ResourceWarning as e:
    #         pytest.fail(f"Limiting was not successful: {e}")
    #
    # def test_multi_rate_no_priority(self, multi_rate_limit_io, throttle_state_io):
    #     system = System(multi_rate_limit_io, throttle_state_io, fixed_period=True)
    #
    #     count = max(system.rate_limit, key=lambda rl: rl.count).count
    #     n = count + 25
    #     try:
    #         for _ in range(n):
    #             system.rate_limited_call(level=0)
    #
    #         assert len(system.endpoint.log) == n
    #     except ResourceWarning as e:
    #         rl = system.gcra.rate_io.read()
    #         pytest.fail(f"Limiting was not successful: {e}, {system.time.time()}:{[r.__dict__ for r in rl]}")
    #
    # def test_excessive_wait(self, single_rate_limit_io, throttle_state_io):
    #     system = System(single_rate_limit_io, throttle_state_io, fixed_period=True)
    #
    #     n = system.rate_limit.count + 25
    #     initial_wait = 2 * system.rate_limit.inverse
    #
    #     with pytest.raises(throttle.ExcessiveWaitTime):
    #         for _ in range(n):
    #             system.rate_limited_call(level=0, allowed_wait=initial_wait)
    #
    #     assert system.rate_limit.usage > 1, f"Expected to have some usage, {system.rate_limit.usage}"
