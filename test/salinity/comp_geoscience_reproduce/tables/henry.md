## Henry problem: conservation, bounds and error against the semi-analytical solution

| run | steps | solute MB error | fluid mass drift | interior fluid residual | u min | u max | in [0, 1] | L2 error | Linf error |
|---|---|---|---|---|---|---|---|---|---|
| FCT | 3094 | 4.9e-18 | 0.0e+00 | 4.7e-18 | 0.0e+00 | 0.998243 | yes | 2.761e-02 | 3.929e-01 |
| low order | 2921 | 2.7e-18 | 0.0e+00 | -- | 0.0e+00 | 0.998118 | yes | 2.785e-02 | 4.661e-01 |

- solute MB error: time-integrated solute created by the coupling / solute mass at T
- fluid mass drift: max over steps of |sum ML (M - m_flow)| / sum ML m_flow
- interior fluid residual: max |B_interior|/dt (Newton residual); -- = not logged in this run
- L2, Linf: nodal error against the semi-analytical solution (Zidane et al. 2012)
