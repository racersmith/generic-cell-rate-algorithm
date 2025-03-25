from generic_cell_rate_algorithm import util

def test_two_point():
    short_rate = util.GenericRate(count=10, period=60)
    long_rate = util.GenericRate(count=123, period=987)

    rate = util.calculate_two_point_rate(
        short_count=short_rate.count,
        short_period=short_rate.period,
        long_count=long_rate.count,
        long_period=long_rate.period,
    )

    assert rate.count/rate.period <= short_rate.count/short_rate.period
    assert rate.count/rate.period <= long_rate.count/long_rate.period
