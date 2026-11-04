# tranport_n_base.py is Elder_problem/stab_5/tranport_n.py, which the original
# run dir loaded as ../tranport_n.py
exec(open('tranport_n_base.py').read())
l_atol_res = 1.0e-10
# NO dt cap: stepController stays ../tranport_n.py's HeuristicNL_dt_controller.
