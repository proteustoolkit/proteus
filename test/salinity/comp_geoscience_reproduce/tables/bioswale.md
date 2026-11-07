## Bioswale, bio2d set-up (T = 0.5 d): conservation and bounds

| run | steps | solute MB error | fluid mass drift | interior fluid residual | u min | u max | in [0, 1] |
|---|---|---|---|---|---|---|---|
| FCT | 430 | 1.1e-20 | 2.2e-11 | -- | 0.0e+00 | 1.000000 | yes |

- solute MB error: time-integrated solute created by the coupling / solute mass at T
- fluid mass drift: max over steps of |sum ML (M - m_flow)| / sum ML m_flow
- interior fluid residual: max |B_interior|/dt (Newton residual); -- = not logged in this run
