from proteus import Context

ct=Context.Options([
    ("USE_SUPG_NS",0,"Use SUPG stabilization"),
    ("ARTIFICIAL_VISCOSITY_NS",2,"0: no art visc., 1: shock capturing, 2: entropy-viscosity"),
    ("nd",2,"Physical dimensioln"),
    ("T",0.01,"Final time of the automated test (test_hotstart_3p runs further)")
],mutable=True)