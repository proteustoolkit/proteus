"""The coverage ratchet (.github/coverage/ratchet.py) checks parts of the package, not only modules.

A floor whose key ends in "/" is an aggregate (every module under it, counted together), a key
``<module>::<Class>`` is one class of a module. Both are checked like a module's floor: below it
(by more than the tolerance) fails the run, a class or a package that cannot be found is reported
as no longer measured. Built here on a small Cobertura report and a two-class module, so the
branches that the real report exercises only partly (a drop in an aggregate, a class that moved)
are pinned.
"""
import importlib.util
import json
import os
import textwrap

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATCHET = os.path.join(REPO_ROOT, ".github", "coverage", "ratchet.py")
if not os.path.exists(RATCHET):      # an installed package's tests run without the CI directory
    pytest.skip("the checkout has no .github/coverage/ratchet.py", allow_module_level=True)

spec = importlib.util.spec_from_file_location("ratchet_under_test", RATCHET)
ratchet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ratchet)

# Steps.py: class Fast on lines 3-203 (200 statements), class Slow on lines 205-405 (200 statements). The units are
# this large because a small one is allowed to sit two statements under its floor, which is a large share of it.
STEPS = "import os\n\nclass Fast:\n" + "    pass\n" * 200 + "\nclass Slow:\n" + "    pass\n" * 200


def report(tmp_path, kernels_hits):
    """A combined report: proteus/Steps.py (Fast fully covered, Slow not) and two files under proteus/kernels/."""
    src = tmp_path / "proteus"
    (src / "kernels").mkdir(parents=True)
    (src / "Steps.py").write_text(STEPS)
    lines = lambda numbers, hits: "".join('<line number="%d" hits="%d"/>' % (n, hits(n)) for n in numbers)
    steps = lines(range(4, 204), lambda n: 1) + lines(range(206, 406), lambda n: 0)
    k1 = lines(range(1, 201), lambda n: kernels_hits if n <= 100 else 1)     # 200 of 200, or 100 of 200
    k2 = lines(range(1, 201), lambda n: 1 if n <= 100 else 0)                # 100 of 200
    xml = ('<coverage><sources><source>%s</source></sources><packages><package><classes>'
           '<class filename="Steps.py"><lines>%s</lines></class>'
           '<class filename="kernels/A.py"><lines>%s</lines></class>'
           '<class filename="kernels/B.py"><lines>%s</lines></class>'
           '</classes></package></packages></coverage>' % (src, steps, k1, k2))
    path = tmp_path / "coverage.xml"
    path.write_text(xml)
    return str(path)


def run(tmp_path, floors, kernels_hits=1, extra=()):
    floors_path = tmp_path / "floors.json"
    floors_path.write_text(json.dumps(floors))
    return ratchet.main([report(tmp_path, kernels_hits), str(floors_path), *extra]), floors_path


def test_an_aggregate_and_a_class_floor_hold_at_the_measured_values(tmp_path, capsys):
    # kernels: 300 of 400 statements = 75.0%; Fast 200 of 200; Slow 0 of 200
    rc, _ = run(tmp_path, {"proteus/kernels/": 75.0, "proteus/Steps.py::Fast": 100.0, "proteus/Steps.py::Slow": 0.0,
                           "proteus/Steps.py": 50.0, "proteus/kernels/A.py": 100.0, "proteus/kernels/B.py": 50.0})
    assert rc == 0, capsys.readouterr().out


def test_an_aggregate_below_its_floor_fails_though_every_module_holds_its_own(tmp_path, capsys):
    # A.py drops from 200 to 100 covered statements (50%, its floor): the aggregate falls from 75% to 50%
    rc, _ = run(tmp_path, {"proteus/kernels/": 70.0, "proteus/kernels/A.py": 50.0, "proteus/kernels/B.py": 50.0,
                           "proteus/Steps.py": 50.0}, kernels_hits=0)
    out = capsys.readouterr()
    assert rc == 1
    assert "`proteus/kernels/`" in out.out


def test_a_class_below_its_floor_fails_and_its_neighbour_is_not_counted(tmp_path, capsys):
    # Slow is at 0%; a floor of 60 on it fails, while the module as a whole (200 of 400 = 50%) holds its own floor
    rc, _ = run(tmp_path, {"proteus/Steps.py": 50.0, "proteus/Steps.py::Slow": 60.0, "proteus/Steps.py::Fast": 100.0})
    out = capsys.readouterr()
    assert rc == 1
    assert "proteus/Steps.py::Slow" in out.out and "proteus/Steps.py::Fast" not in out.out.split("Below the recorded floor")[1].split("###")[0]


def test_a_class_that_is_gone_is_reported_not_failed(tmp_path, capsys):
    rc, _ = run(tmp_path, {"proteus/Steps.py": 50.0, "proteus/Steps.py::Renamed": 100.0})
    assert rc == 0
    assert "Floors for modules no longer measured" in capsys.readouterr().out


def test_write_floors_raises_a_part_only_when_it_is_a_point_above_and_adds_no_new_one(tmp_path):
    rc, floors_path = run(tmp_path, {"proteus/kernels/": 70.0, "proteus/Steps.py::Fast": 99.5,
                                     "proteus/Steps.py": 50.0}, extra=("--write-floors",))
    assert rc == 0
    written = json.loads(floors_path.read_text())
    assert written["proteus/kernels/"] == 75.0           # 5 points above: raised
    assert written["proteus/Steps.py::Fast"] == 99.5     # half a point above: kept
    assert "proteus/Steps.py::Slow" not in written       # a part has a floor only when someone writes one
