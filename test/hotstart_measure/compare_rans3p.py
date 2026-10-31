"""Compare two run_rans3p.py archives: per field (u, v, p, pInc, velocity), relative L2 and Linf of the
difference at steps FROM, FROM+1, FROM+5 and the last.

    python compare_rans3p.py A.h5 B.h5 NAME FROM_STEP
"""
import sys
import h5py
import numpy as np

a, b, from_step = h5py.File(sys.argv[1], "r"), h5py.File(sys.argv[2], "r"), int(sys.argv[4])
fields = ["u", "v", "p", "pInc", "velocity"]
steps = sorted(int(k[3:]) for k in a if k.startswith("u_t") and k[3:].isdigit())
last = {}
print("%-6s %-9s %12s %12s" % ("step", "field", "rel_L2", "Linf"))
for s in steps:
    if s < from_step:
        continue
    for f in fields:
        k = "%s_t%d" % (f, s)
        if k not in a or k not in b:
            continue
        x, y = np.array(a[k]).ravel(), np.array(b[k]).ravel()
        d = x - y
        l2 = np.linalg.norm(d) / max(np.linalg.norm(x), 1e-300)
        last[f] = (s, l2, np.abs(d).max())
        if s in (from_step, from_step + 1, from_step + 5, steps[-1]):
            print("%-6d %-9s %12.3e %12.3e" % (s, f, l2, last[f][2]))
print("FINAL", sys.argv[3], " ".join("%s:L2=%.2e,Linf=%.2e" % (f, v[1], v[2]) for f, v in last.items()))
