from dataclasses import dataclass



@dataclass(frozen=True)
class GenericRate:  #pragma: nocover
    count: int
    period: float


def calculate_two_point_rate(short_count: int, short_period: float, long_count: int, long_period: float) -> GenericRate:
    """ Create a rate limit for GCRA that tries to achieve the requested short and long term limits.
    Assume that any allowed burst can be achieved in zero time (conservative)
    Then determine what burst is allowed followed by the remaining rate to achieve the long count at long period
    such that the combination of the burst and this reduced rate achieves the short count over the short period.
    """
    if short_count > long_count:
        raise ValueError(f"short_count should be less than long_count: {short_count=} < {long_count=}")

    if short_period > long_period:
        raise ValueError(f"short_period should be less than long_period: {short_period=} < {long_period=}")

    target_rate = (long_count - short_count) / (long_period - short_period)
    target_seconds = short_count / target_rate
    target_count = max(1, int(short_count - target_rate * short_period))

    return GenericRate(count=target_count, period=target_seconds)
