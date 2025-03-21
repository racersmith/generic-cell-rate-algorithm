import pytest

from typing import Callable, List
import datetime
import numpy as np

from generic_cell_rate_algorithm import throttle


class TimeWarp:
    def __init__(self):
        self.clock = 0.0

    def now(self) -> float:
        return self.clock

    def sleep(self, seconds) -> None:
        self.clock += seconds


class TATStore(throttle.TatStoreProtocol):
    def __init__(self, rate_limit: throttle.RateLimit, now: Callable):
        self.tat = 0

        self.rate_limit = rate_limit
        self.now = now

    def _get(self):
        return self.tat

    def _set(self, tat):
        self.tat = tat

    def _now(self):
        return self.now()

    def update(self, *args, **kwargs):
        tat_update = throttle.update_tat(self._now(), self._get(), self.rate_limit)
        self._set(tat_update.new_tat)
        return tat_update.wait_time


class PriorityTATStore(throttle.TatStoreProtocol):
    def __init__(self, rate_limits: List[throttle.RateLimit], now: Callable):
        self.tats = [[0.0, rate_limit, 0] for rate_limit in rate_limits]
        self.now = now

    def _get(self, level:int=0):
        tats = self.tats[:level+1]
        i = np.argmin([tat[0] for tat in tats]).ravel()[0]

        return i, self.tats[i][0], self.tats[i][1]

    def _set(self, i, new_tat):
        self.tats[i][0] = new_tat
        self.tats[i][2] += 1

    def _now(self):
        return self.now()

    def update(self, level: int=0):
        i, tat, ratelimit = self._get(level)
        tat_update = throttle.update_tat(self._now(), tat, ratelimit)
        self._set(i, tat_update.new_tat)
        return tat_update.wait_time


class TestPriorityTATStore:
    def test_levels(self):
        rate_limits = [
            throttle.RateLimit(10, 60),
            throttle.RateLimit(5, 100)
        ]
        timewarp = TimeWarp()
        store = PriorityTATStore(rate_limits=rate_limits, now=timewarp.now)

        store.tats[0][0] = 60

        i, tat, ratelimit = store._get(level=0)
        assert i == 0
        assert tat == 60
        assert ratelimit == rate_limits[0]
        store._set(i, 75)
        assert store.tats[i][2] == 1
        i, tat, ratelimit = store._get(level=0)
        assert tat == 75

        i, tat, ratelimit = store._get(level=1)
        assert i == 1
        assert tat == 0
        assert ratelimit == rate_limits[1]
        store._set(i, 90)
        assert store.tats[i][2] == 1
        i, tat, ratelimit = store._get(level=1)
        assert tat == 75


class TestTwoPoint:
    n_s, n_l = 10, 100
    p_s, p_l = datetime.timedelta(seconds=10), datetime.timedelta(seconds=1000)

    def test_error(self):
        with pytest.raises(ValueError):
            throttle.create_two_point_rate_limit(self.n_l, self.p_s, self.n_s, self.p_l)

        with pytest.raises(ValueError):
            throttle.create_two_point_rate_limit(self.n_s, self.p_l, self.n_l, self.p_s)

    def test_normal(self):
        rate_limit = throttle.create_two_point_rate_limit(self.n_s, self.p_s, self.n_l, self.p_l)
        assert rate_limit.count > 0
        assert rate_limit.period_seconds > 0

    def test_short(self):
        # force a condition where no burst is allowed
        rate_limit = throttle.create_two_point_rate_limit(self.n_s, self.p_s, 1000 * self.n_l, self.p_l)
        assert rate_limit.count == 1
        assert rate_limit.period_seconds > 0


class TestThrottle:
    def test_simple(self):
        n = 10
        period = 60

        rate_limit = throttle.RateLimit(n, datetime.timedelta(seconds=period).total_seconds())

        time_warp = TimeWarp()
        tat_store = TATStore(rate_limit=rate_limit, now=time_warp.now)
        gcra = throttle.GCRA(tat_store=tat_store, sleep_fn=time_warp.sleep)

        @gcra.throttle()
        def fn(i):
            i += 1
            return i

        count = 0
        start_time = time_warp.now()
        for _ in range(n):
            count = fn(count)
        end_time = time_warp.now()
        assert end_time - start_time == 0, f"Burst should not be limited. wait={end_time - start_time}"

        # start_time = time_warp.now()
        for i in range(1000*n):
            count = fn(count)

        end_time = time_warp.now()
        rate = count/(end_time - start_time)
        expected_rate = 1/rate_limit.inverse
        rate_limit_error = (rate - expected_rate)/expected_rate
        assert rate_limit_error < 0.01, f"Try and be conservative on the error"

    def test_two_point(self):
        n_s = 300
        period_s = datetime.timedelta(minutes=15)
        n_l = 3000
        period_l = datetime.timedelta(hours=24)

        rate_limit = throttle.create_two_point_rate_limit(n_s, period_s, n_l, period_l)

        time_warp = TimeWarp()
        tat_store = TATStore(rate_limit=rate_limit, now=time_warp.now)
        gcra = throttle.GCRA(tat_store=tat_store, sleep_fn=time_warp.sleep)

        @gcra.throttle()
        def fn(i):
            i += 1
            return i

        count = 0

        # Check the burst limit in the short period
        while time_warp.now() < period_s.total_seconds():
            count = fn(count)

        assert count <= n_s, f"We want throttle to be conservative, {count} <= {n_s}"
        assert (n_s - count) / n_s <= 0.1, f"Good enough for me is within 1% of the target count"
        assert time_warp.now() >= period_s.total_seconds()

        # Check the combined burst and throttled calls adhere to the long period
        while time_warp.now() < period_l.total_seconds():
            count = fn(count)
        assert count <= n_l, f"We want throttle to be conservative, {count} <= {n_l}"
        assert (n_l - count)/n_l <= 0.1, f"Good enough for me is within 1% of the target count"
        assert time_warp.now() >= period_l.total_seconds()

    def test_priority_default(self):
        n_s = 300
        period_s = datetime.timedelta(minutes=15)
        n_l = 3000
        period_l = datetime.timedelta(hours=24)

        rate_limit = throttle.create_two_point_rate_limit(n_s, period_s, n_l, period_l)

        time_warp = TimeWarp()
        tat_store = PriorityTATStore(rate_limits=[rate_limit], now=time_warp.now)
        gcra = throttle.GCRA(tat_store=tat_store, sleep_fn=time_warp.sleep)

        @gcra.throttle()
        def fn(i):
            i += 1
            return i

        count = 0

        # Check the burst limit in the short period
        while time_warp.now() < period_s.total_seconds():
            count = fn(count)

        assert count <= n_s, f"We want throttle to be conservative, {count} <= {n_s}"
        assert (n_s - count) / n_s <= 0.1, f"Good enough for me is within 1% of the target count"
        assert time_warp.now() >= period_s.total_seconds()

        # Check the combined burst and throttled calls adhere to the long period
        while time_warp.now() < period_l.total_seconds():
            count = fn(count)
        assert count <= n_l, f"We want throttle to be conservative, {count} <= {n_l}"
        assert (n_l - count)/n_l <= 0.1, f"Good enough for me is within 1% of the target count"
        assert time_warp.now() >= period_l.total_seconds()

    def test_priority(self):
        rate_limits = [
            throttle.RateLimit(1, datetime.timedelta(minutes=15).total_seconds()),
            throttle.RateLimit(1, datetime.timedelta(minutes=15).total_seconds()),
        ]

        time_warp = TimeWarp()
        tat_store = PriorityTATStore(rate_limits=rate_limits, now=time_warp.now)
        gcra = throttle.GCRA(tat_store=tat_store, sleep_fn=time_warp.sleep)

        @gcra.throttle(level=0)
        def fn(i):
            i += 1
            return i

        count = 0

        def fn2(i):
            i += 1
            return i

        n_fn = 0
        n_fn2 = 0

        fn_sleep = list()
        fn2_sleep = list()

        for i_calls in range(1000):
            t0 = time_warp.now()
            n_fn = fn(n_fn)
            t1 = time_warp.now()
            fn_sleep.append(t1 - t0)
            with gcra.throttle(level=1):
                t0 = time_warp.now()
                n_fn2 = fn2(n_fn2)
                t1 = time_warp.now()
                fn2_sleep.append(t1 - t0)

        assert tat_store.tats[0][0] > 0
        assert tat_store.tats[1][0] > 0
        assert np.mean(fn2_sleep) < np.mean(fn_sleep)

        level_0_rate = (tat_store.tats[0][2] - 1)/time_warp.now()
        assert level_0_rate <= tat_store.tats[0][1].count/tat_store.tats[0][1].period_seconds
        level_1_rate = (tat_store.tats[1][2] - 1)/time_warp.now()
        assert level_1_rate <= tat_store.tats[1][1].count / tat_store.tats[1][1].period_seconds


    def test_wrapper(self):
        time_warp = TimeWarp()
        rate_limit = throttle.RateLimit(1, datetime.timedelta(minutes=15).total_seconds())
        tat_store = TATStore(rate_limit=rate_limit, now=time_warp.now)
        gcra = throttle.GCRA(tat_store=tat_store, sleep_fn=time_warp.sleep)

        @gcra.throttle(level=0)
        def fn(i):
            i += 1
            return i

        def fn2(i):
            i += 1
            return i

        a = 0
        a = fn(a)

        with gcra.throttle():
            b = 0
            b = fn2(b)

    def test_timeout(self):
        time_warp = TimeWarp()
        rate_limit = throttle.RateLimit(2, 60)
        tat_store = TATStore(rate_limit=rate_limit, now=time_warp.now)

        allowed_wait = 15  # seconds
        gcra = throttle.GCRA(tat_store=tat_store, sleep_fn=time_warp.sleep)

        def fn(i):
            i += 1
            return i

        a = 0
        # we should get two unrestricted calls
        for i in range(2):
            with gcra.throttle(allowed_wait=allowed_wait):
                a = fn(a)

        # now we should have a wait time that exceeds our allowed.
        with pytest.raises(throttle.ExcessiveWaitTime):
            with gcra.throttle(allowed_wait=allowed_wait):
                a = fn(a)
