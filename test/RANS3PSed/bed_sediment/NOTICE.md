# bed_sediment: origin

Copied on 2026-11-08 from erdc/air-water-vv, commit f93ff99 (2022-08-11), directory `2d/sediment/bed_sediment`:
the case modules `tank.py`, `tank_so.py` and the p/n modules of the nine models its `pnList` names with
`sedimentDynamics=True` and `useRANS=0` (vos, vof, ls, redist, ls_consrv, threep_navier_stokes_sed,
twp_navier_stokes, pressureincrement, pressure, plus pressureInitial). The turbulence modules (kappa, dissipation,
appended only for `useRANS > 0`), `run.sh`, the plotting scripts and the test series were not copied.

air-water-vv is MIT licensed, Copyright (c) 2014 ERDC Computational Mechanics. Its authors: see that repository's
history.

One change, so that it runs on proteus 2.0: `RDLS3P` (removed from proteus in f59334fc, #963) is `RDLS` in
`redist_p.py` and `redist_n.py`. Everything else is as copied, including the integer `rho_s = 2600`, which proteus's
PresInc used to refuse (fixed in the same commit). `test_bed_sediment.py` says what is checked.
