## Coupled manufactured solution (T3): conservation and bounds

| scheme | N | solute MB error | fluid mass drift | c min | c max | in [0.2, 0.8] |
|---|---|---|---|---|---|---|
| low order | 10 | 3.1e-18 | 0.0e+00 | 0.2000 | 0.7972 | yes |
| low order | 20 | 1.8e-18 | 0.0e+00 | 0.2000 | 0.7972 | yes |
| low order | 40 | 3.8e-19 | 0.0e+00 | 0.2000 | 0.7972 | yes |
| low order | 80 | 4.4e-20 | 0.0e+00 | 0.2000 | 0.7972 | yes |
| low order | 160 | 5.0e-20 | 0.0e+00 | 0.2000 | 0.7972 | yes |
| low order | 320 | 8.2e-20 | 0.0e+00 | 0.2000 | 0.7972 | yes |
| FCT | 10 | 5.4e-18 | 5.0e-17 | 0.2000 | 0.7972 | yes |
| FCT | 20 | 1.4e-18 | 2.2e-16 | 0.2000 | 0.7972 | yes |
| FCT | 40 | 8.5e-20 | 4.4e-16 | 0.2000 | 0.7972 | yes |
| FCT | 80 | 7.4e-19 | 9.9e-16 | 0.2000 | 0.7972 | yes |
| FCT | 160 | 2.0e-19 | 2.0e-15 | 0.2000 | 0.7972 | yes |
| FCT | 320 | 7.1e-20 | 4.1e-15 | 0.2000 | 0.7972 | yes |
| T1 (c = c0), FCT | 20 | 3.0e-19 | 1.9e-16 | 0.5000 | 0.5000 | max|c-c0| = 0.0e+00 |

- solute MB error: time-integrated solute created by the coupling / solute mass at T
- fluid mass drift: max over steps of |sum ML (M - m_flow)| / sum ML m_flow
- T1: constant concentration c0 = 0.5; the last column is max over the run of |c - c0|
