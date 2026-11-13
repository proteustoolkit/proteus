"""Every proteus.mprans module imports in a fresh interpreter, on its own.

A module that does ``import proteus`` and then names proteus.StepControl,
proteus.SubgridError, ... as attributes works only if something else imported
those submodules first. A single pytest process (or ``from proteus.iproteus
import *``) always had, so it never showed; one pytest process per file did
(``module 'proteus' has no attribute 'StepControl'`` from MCorr, on the 1.9.1
HPC module tests). Each module gets its own subprocess so no earlier import
can mask a missing one.
"""
import os
import re
import subprocess
import sys

import pytest

import proteus

_mprans_dir = os.path.join(os.path.dirname(proteus.__file__), "mprans")
_root = os.path.dirname(os.path.dirname(os.path.abspath(proteus.__file__)))

MODULES = sorted(f[:-3] for f in os.listdir(_mprans_dir)
                 if f.endswith(".py") and f != "__init__.py")


@pytest.mark.parametrize("module", MODULES)
def test_mprans_module_imports_alone(module):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_root] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env.setdefault("FI_PROVIDER", "tcp")
    r = subprocess.run(
        [sys.executable, "-c", "import proteus.mprans.%s" % module],
        capture_output=True, text=True, env=env, timeout=300)
    if r.returncode == 0:
        return
    last = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
    m = re.match(r"ModuleNotFoundError: No module named '([^']+)'", last)
    if m and not m.group(1).startswith("proteus"):
        # an optional third-party dependency (e.g. pychrono), not ours
        pytest.skip("optional dependency missing: " + m.group(1))
    pytest.fail("import proteus.mprans.%s in a fresh interpreter failed: %s\n%s"
                % (module, last, r.stderr[-2000:]))
