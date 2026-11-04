# flow_n_base.py is Elder_problem/stab_5/flow_n.py, which the original run dir
# loaded as ../flow_n.py
exec(open('flow_n_base.py').read())
l_atol_res = 1.0e-10
# ../flow_n.py sets no prefix, so the -richards_* SuperLU options in
# petsc.options.superlu never reached Richards: it ran default GMRES, which
# cannot solve this closed saturated box (beta=0, two weakly pinned corners)
# once the nodal-rho coupling asks Newton for a real correction.
linear_solver_options_prefix = "richards_"
# NO dt cap: stepController stays ../flow_n.py's HeuristicNL_dt_controller
# (dt grows on the Newton iteration count only), as in ../run_fullspeed.
