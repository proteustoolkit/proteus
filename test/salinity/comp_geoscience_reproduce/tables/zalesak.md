## Zalesak's disk after one revolution

| run | solute mass change | u min at T | u max at T | in [0, 1], all frames | L1 error |
|---|---|---|---|---|---|
| ref 4, low order | -2.6e-02 | -2.7e-12 | 0.560530 | yes | 9.6324e-02 |
| ref 4, FCT | -2.6e-04 | -5.1e-19 | 0.816087 | yes | 5.3465e-02 |
| ref 5, low order | -5.9e-03 | -1.1e-20 | 0.665860 | yes | 7.8379e-02 |
| ref 5, FCT | -7.6e-06 | -7.3e-19 | 0.938197 | yes | 3.8874e-02 |

- solute mass change: sum(ML u) at T / at t = 0 - 1, including what leaves through the weakly imposed u = 0 boundary
- L1 error against the exact solution (the initial condition)
- ref 4: current build, serial; ref 5: serial (the 64-rank ref 5 run predates the MPI fix of TADR's limiter)
