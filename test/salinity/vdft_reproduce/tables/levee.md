## Levee, density and non-density: conservation and bounds

| run | steps | solute MB error | fluid mass drift | interior fluid residual | u min | u max | in [0, 0.7] |
|---|---|---|---|---|---|---|---|
| density | 2906 | 7.4e-20 | 0.0e+00 | 8.0e-07 | -2.3e-183 | 0.700000 | yes |
| non-density | 985 | 5.0e-20 | 0.0e+00 | 8.8e-07 | -9.4e-180 | 0.700000 | yes |

- solute MB error: time-integrated solute created by the coupling / solute mass at T
- fluid mass drift: max over steps of |sum ML (M - m_flow)| / sum ML m_flow
- interior fluid residual: max |B_interior|/dt (Newton residual); -- = not logged in this run
