# Throttling using Generic Cell Rate Algorithm
Designed for limiting calls to APIs that have imposed rate limits.

[Wiki page on GCRA](https://en.wikipedia.org/wiki/Generic_cell_rate_algorithm)
    
The GCRA is an interesting approach to rate limiting in that it's storage is
minimal and requires only the storage of a single value in the simplest terms, the
Theoretical Arrival Time.

The Theoretical Arrival Time or TAT is the time at which the next call could be made 
that would not be blocked due to rate limits.