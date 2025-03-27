import pytest

from generic_cell_rate_algorithm import throttle
from generic_cell_rate_algorithm.throttle import RateLimit

from .conftest import ThrottleStateInterface, throttle_state_row, MockEndpoint


def test_time_warp(time):
    assert time.time() == 0
    time.sleep(1)
    assert time.time() == 1
    time.sleep(2)
    assert time.time() == 3
    assert time.time() == 3


class TestMockEndpoint:
    def test_verify_rolling_period(self, time):
        rate_limit = RateLimit(count=10, period=12)
        endpoint = MockEndpoint(time, rate_limit)

        def fn():
            endpoint()
            time.sleep(rate_limit.inverse)

        for _ in range(rate_limit.count):
            fn()

        assert len(endpoint._requests) == rate_limit.count
        assert endpoint.verify_rolling_period([rate_limit]) is None
        assert endpoint.verify_rolling_period([rate_limit, rate_limit]) is None

        restricted_rate = RateLimit(count=9, period=12)
        with pytest.raises(ResourceWarning):
            endpoint.verify_rolling_period([restricted_rate])

        with pytest.raises(ResourceWarning):
            endpoint.verify_rolling_period([rate_limit, restricted_rate])

    def test_auto_rolling_valid(self, time):
        rate_limit = RateLimit(count=10, period=12)
        endpoint = MockEndpoint(time, rate_limit)

        def fn():
            endpoint()
            time.sleep(rate_limit.inverse)

        try:
            for _ in range(rate_limit.count):
                fn()
        except ResourceWarning as e:
            pytest.fail(f"Too many requests for the period: {e}", e)


    def test_auto_rolling_invalid(self, time):
        rate_limit = RateLimit(count=10, period=12)
        restricted_rate = RateLimit(
            count=rate_limit.count - 1, # Restrict the allowed count by 1.
            period=rate_limit.period
        )

        endpoint = MockEndpoint(time, [rate_limit, restricted_rate])

        def fn():
            endpoint()
            time.sleep(rate_limit.inverse)

        with pytest.raises(ResourceWarning):
            for _ in range(rate_limit.count):
                fn()

    def test_update_usage(self, time):
        rate_limit = RateLimit(count=10, period=12)

        endpoint = MockEndpoint(time=time, rate_limit=[rate_limit], fixed_period=True)
        def fn():
            endpoint()
            return None

        assert rate_limit.usage == 0, f"{rate_limit}"
        fn()
        assert len(endpoint._requests) == 1, f"{endpoint._requests}"
        assert rate_limit.usage == 1, f"{rate_limit}"

        fn()
        fn()
        fn()
        assert len(endpoint._requests) == 4, f"{endpoint._requests}"
        assert rate_limit.usage == 4, f"{rate_limit}"

        time.sleep(2*rate_limit.period)
        fn()
        print(endpoint._requests)
        print(rate_limit.usage)
        assert endpoint._requests[-1] == 2*rate_limit.period
        assert len(endpoint._requests) == 5, f"{endpoint._requests}"
        assert rate_limit.usage == 1, f"{rate_limit}"

    def test_auto_fixed_valid(self, time):
        rate_limit = RateLimit(count=10, period=12)

        endpoint = MockEndpoint(time=time, rate_limit=[rate_limit], fixed_period=True)

        def fn():
            endpoint
            time.sleep(2*rate_limit.inverse)

        try:
            for _ in range(2*rate_limit.count):
                fn()

        except ResourceWarning as e:
            pytest.fail(f"Too many requests for the period: {e}, {endpoint._requests[-1]}", e)

    def test_auto_fixed_invalid(self, time):
        rate_limit = RateLimit(count=10, period=10)
        restricted_rate = RateLimit(
            count=rate_limit.count - 1, # Restrict the allowed count by 1.
            period=rate_limit.period
        )

        endpoint = MockEndpoint(time=time, rate_limit=[rate_limit, restricted_rate], fixed_period=True)

        def fn():
            endpoint()
            time.sleep(1)

        with pytest.raises(ResourceWarning):
            for _ in range(rate_limit.count):
                fn()
            assert False, f"{restricted_rate.usage}: {endpoint._requests}"


# class TestPriorityFunction:
#     def test_levels(self, time):
#         rate_limit = throttle.RateLimit(count=10, period=12)
#         gcra = throttle.GCRA()


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


