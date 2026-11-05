"""Every call from proteus/NumericalFlux.py into cnumericalFlux passes the sparse-diffusion pattern and the boundary
flags where the Cython signature takes them.

The wrappers take plain ndarrays, so an argument in the wrong place is not caught: Diffusion_LDG passed (rowptr,
colind, isDiffusiveFluxBoundary) for (isDiffusiveFluxBoundary, rowptr, colind), so the LDG exterior flux Jacobian read
the face flags past the end of rowptr and dropped a V.n (an LDG P1 solve was off by 1.7e2 and differed run to run);
three Hamilton-Jacobi diffusion fluxes passed rowptr as colind (the same numbers only for a diagonal tensor); and
DarcyFCFF's residual passed the (0,1) block's rowptr for the (1,0) and (1,1) blocks. This reads the source, so it needs
no build."""
import os
import re

import pytest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "proteus")


def _signatures():
    pyx = open(os.path.join(ROOT, "cnumericalFlux.pyx")).read()
    sigs = {}
    for m in re.finditer(r"^def (\w+)\((.*?)\):", pyx, re.S | re.M):
        sigs[m.group(1)] = [a.split()[-1].split("=")[0] for a in m.group(2).replace("\n", " ").split(",") if a.strip()]
    return sigs


def _calls():
    src = open(os.path.join(ROOT, "NumericalFlux.py")).read()
    src = "\n".join(re.sub(r"#.*", "", line) for line in src.split("\n"))
    for m in re.finditer(r"cnumericalFlux\.(\w+)\(", src):
        i = j = m.end()
        depth = 1
        while depth:
            depth += (src[j] in "([{") - (src[j] in ")]}")
            j += 1
        args, depth, cur = [], 0, ""
        for c in src[i:j - 1]:
            depth += (c in "([{") - (c in ")]}")
            if c == "," and depth == 0:
                args.append(cur.strip())
                cur = ""
            else:
                cur += c
        if cur.strip():
            args.append(cur.strip())
        yield m.group(1), src[:m.start()].count("\n") + 1, args


def _block(name):
    """The sdInfo block a parameter like rowptr_mw names, as the pair of component indices ((1, 0) for mw)."""
    suffix = name.split("_")[1] if "_" in name else None
    return None if suffix is None else tuple("wmn".index(c) if c != "n" else 1 for c in suffix)


def test_sparse_diffusion_and_flag_arguments_are_in_place():
    sigs = _signatures()
    wrong = []
    for name, line, args in _calls():
        sig = sigs.get(name)
        if sig is None or len(sig) != len(args):
            continue  # wrappers with defaulted trailing arguments
        for param, arg in zip(sig, args):
            p = param.lower()
            kind = "rowptr" if p.startswith("rowptr") else "colind" if p.startswith("colind") else None
            if kind:
                index = "[0]" if kind == "rowptr" else "[1]"
                if "sdInfo" not in arg or not arg.endswith(index):
                    wrong.append((line, name, param, arg))
                    continue
                block = _block(p)
                if block is not None and p.endswith(("ww", "nn", "mm", "wm", "mw")):
                    found = re.search(r"sdInfo\[\((\d),(\d)\)\]", arg.replace(" ", ""))
                    expected = {"ww": (0, 0), "nn": (1, 1), "mm": (1, 1), "wm": (0, 1), "mw": (1, 0)}[p[-2:]]
                    if found and tuple(int(x) for x in found.groups()) != expected:
                        wrong.append((line, name, param, arg))
            elif "sdInfo" in arg:
                wrong.append((line, name, param, arg))
            if ("isdiffusivefluxboundary" in p) != ("isDiffusiveFluxBoundary" in arg):
                wrong.append((line, name, param, arg))
    assert not wrong, "\n".join("NumericalFlux.py:%d %s: %s <- %s" % w for w in wrong)
