# Throttling using Generic Cell Rate Algorithm
Designed for limiting calls to APIs that have imposed rate limits.

[Wiki page on GCRA](https://en.wikipedia.org/wiki/Generic_cell_rate_algorithm)
    
The GCRA is an interesting approach to rate limiting in that it's storage is
minimal and requires only the storage of a single value in the simplest terms, the
Theoretical Arrival Time.

The Theoretical Arrival Time or TAT is the time at which the next call could be made 
that would not be blocked due to rate limits.

# Something new
## Priority levels
You can provide multiple levels of rate limiting and request a priority level during throttling.
Higher priority levels have access to levels at or below the priority level.  The system will
greedily select the earliest TAT from the store for use.  Each priority level can be given an allocation
of the total rate limit which is set globally.  

## Smooth Burst to Sustained Throttling
With the addition of one additional storage attribute, it is possible to create a rate limiting system
that will smoothly transition between a defined burst limit (count, period) and a sustained limit.
