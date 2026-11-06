# tranport_n_base.py is Elder_problem/stab_5/tranport_n.py, which the original
# run dir loaded as ../tranport_n.py
exec(open('tranport_n_base.py').read())
l_atol_res = 1.0e-10
# NO dt cap: stepController stays ../tranport_n.py's HeuristicNL_dt_controller.

# ELDER_DT [d]: fixed time step for both models (time-refinement / splitting
# study); unset keeps the HeuristicNL_dt_controller above.
import os as _os
if _os.environ.get("ELDER_DT"):
    from proteus.StepControl import FixedStep as _FixedStep
    _ELDER_DT = float(_os.environ["ELDER_DT"])
    class FixedDT_controller(_FixedStep):
        """FixedStep that restores dt = ELDER_DT before every step (plain
        FixedStep would keep the clipped 1e-8 d startup step)."""
        def choose_dt_model(self):
            self.dt_model = _ELDER_DT
            _FixedStep.choose_dt_model(self)
    stepController = FixedDT_controller
