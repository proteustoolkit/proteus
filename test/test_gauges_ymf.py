"""ymf ``gauges:`` read into Proteus gauges (FORCE_HISTORY_GAUGES_PLAN step 4): point, line, line_integral and
boundary_integral definitions become PointGauges, LineGauges, LineIntegralGauges and BoundaryIntegralGauges, and a
run with them writes what the same gauges built by hand write."""
import os
import shutil
import subprocess
import sys

import numpy
import pytest

from proteus.Gauges import (BoundaryIntegralGauges, Gauges, attach_ymf_gauges, gauges_from_ymf)

DAMBREAK = [
    {"name": "pressure", "kind": "point", "fields": ["p"],
     "points": [[3.22, 0.16], [3.22, 0.584], [3.22, 0.12]], "file": "ymf_pressure.csv"},
    {"name": "heights", "kind": "line", "fields": ["φ"],
     "lines": [[[0.0, 0.0, 0.0], [0.0, 0.0, -0.01]], [[0.0, 0.0], [3.22, 0.0]]], "file": "ymf_height2.csv"},
    {"kind": "line_integral", "fields": ["φ"], "lines": [[[1.0, 0.0], [1.0, 1.8]]], "sample_rate": 0.01},
    {"kind": "boundary_integral", "tags": ["y-"], "quantities": ["wetted_area", "force", "moment"],
     "moment_point": [0.1, 0.2], "file": "ymf_forces.csv"},
]
TAGS = {"y-": 1, "x+": 2, "y+": 3, "x-": 4}


def test_every_kind():
    point, line, integral, boundary = gauges_from_ymf({"model": {"gauges": DAMBREAK}}, fields={"φ": "phi"},
                                                      tags=TAGS)
    assert isinstance(point, Gauges) and point.fields == ["p"] and point.fileName == "ymf_pressure.csv"
    assert list(point.points) == [(3.22, 0.16, 0.0), (3.22, 0.584, 0.0), (3.22, 0.12, 0.0)]
    assert line.fields == ["phi"] and line.isLineGauge and len(line.lines) == 2
    assert line.lines[1] == ("phi", ((0.0, 0.0, 0.0), (3.22, 0.0, 0.0)))
    assert integral.isLineIntegralGauge and integral.sampleRate == 0.01
    assert integral.fileName == "line_integral_gauge_2.csv"
    assert isinstance(boundary, BoundaryIntegralGauges)
    assert dict(boundary.tags) == {"y-": 1} and boundary.quantities == ("wetted_area", "force", "moment")
    assert list(boundary.moment_points["y-"]) == [0.1, 0.2, 0.0]


def test_strictyaml_data():
    strictyaml = pytest.importorskip("strictyaml")
    doc = strictyaml.dirty_load("""model:
  gauges:
    - kind: boundary_integral
      tags: [x+]
      quantities: [force]
      sample_rate: 0.5
    - name: p
      kind: point
      fields: [p]
      points: [[1.0, 2.0]]
""", allow_flow_style=True)
    boundary, point = gauges_from_ymf(doc, tags=TAGS)
    assert dict(boundary.tags) == {"x+": 2} and boundary.sampleRate == 0.5 and boundary.fileName == "boundary_integral_gauge_0.csv"
    assert list(point.points) == [(1.0, 2.0, 0.0)] and point.fileName == "p.csv"


@pytest.mark.parametrize("gauge, message", [
    ({"kind": "point", "fields": ["p"]}, "needs points"),
    ({"kind": "line", "fields": ["p"], "points": [[0.0, 0.0]], "lines": [[[0, 0], [1, 1]]]}, "takes no points"),
    ({"kind": "boundary_integral", "tags": ["y-"]}, "needs tags and quantities"),
    ({"kind": "boundary_integral", "tags": ["y-"], "quantities": ["force"], "fields": ["p"]}, "takes no fields"),
    ({"kind": "boundary_integral", "tags": ["lid"], "quantities": ["force"]}, "no boundary flag for the region 'lid'"),
    ({"kind": "line_integral", "fields": ["p"], "lines": [[[0, 0], [1, 1], [2, 2]]]}, "two ends, not 3 points"),
    ({"kind": "point", "fields": ["p"], "points": [[0.0, 0.0]], "tags": ["y-"]}, "takes no tags"),
    ({"kind": "surface", "fields": ["p"]}, "unknown kind"),
    ({"kind": "boundary_integral", "tags": ["y-"], "quantities": ["lift"]}, "unknown boundary integral quantity"),
])
def test_what_a_gauge_needs_is_checked(gauge, message):
    with pytest.raises(ValueError, match=message):
        gauges_from_ymf([gauge], tags=TAGS)


class _P(object):
    def __init__(self, module, names):
        self.coefficients = type("Coefficients", (object,), {"__module__": module})()
        self.coefficients.variableNames = names


class _N(object):
    auxiliaryVariables = []


def test_each_gauge_goes_to_the_model_with_its_fields():
    pList = [_P("proteus.mprans.RANS3PF", ["u", "v"]), _P("proteus.mprans.Pres", ["p"]),
             _P("proteus.mprans.CLSVOF", ["phi"])]
    nList = [_N(), _N(), _N()]
    gauges = attach_ymf_gauges(DAMBREAK, pList, nList, fields={"φ": "phi"}, tags=TAGS)
    assert nList[0].auxiliaryVariables == [gauges[3]]
    assert nList[1].auxiliaryVariables == [gauges[0]]
    assert nList[2].auxiliaryVariables == gauges[1:3]
    with pytest.raises(ValueError, match="no model has a model with the variables \\['T'\\]"):
        attach_ymf_gauges([{"kind": "point", "fields": ["T"], "points": [[0, 0]]}], pList, nList)


def _mpiexec():
    bindir = os.path.dirname(sys.executable)
    for name in ("mpiexec", "mpirun"):
        if os.path.exists(os.path.join(bindir, name)):
            return os.path.join(bindir, name)
    return shutil.which("mpiexec")


YAML = """gauges:
  - name: pressure
    kind: point
    fields: [p]
    points: [[3.22, 0.16], [3.22, 0.584], [3.22, 0.12]]
    file: ymf_pressure.csv
  - name: heights
    kind: line
    fields: [φ]
    lines: [[[0.0, 0.0, 0.0], [0.0, 0.0, -0.01]], [[0.0, 0.0], [3.22, 0.0]]]
    file: ymf_height2.csv
  - kind: boundary_integral
    tags: [y-]
    quantities: [wetted_area, force_pressure, force_viscous, force, moment]
    moment_point: [0.1, 0.2]
    file: ymf_forces.csv
"""


@pytest.mark.parametrize("ranks", [1, 2])
def test_a_dam_break_with_ymf_gauges_writes_the_decks_gauge_files(tmp_path, ranks):
    """test/TwoPhaseFlow/damBreak.py with its own pressure.csv and height2.csv gauges and a BoundaryIntegralGauges on
    flag 1 (moment about (0.1, 0.2)), plus the same three described in ymf: the files agree row for row."""
    pytest.importorskip("strictyaml")
    if ranks > 1 and _mpiexec() is None:
        pytest.skip("mpiexec not available")
    (tmp_path / "gauges.yaml").write_text(YAML, encoding="utf-8")
    env = dict(os.environ, OMP_NUM_THREADS="1", DECK="damBreak.py", OPTS="he=0.1", FT="0.05", BIGAUGES="1",
               YMFGAUGES=str(tmp_path / "gauges.yaml"), YMFFIELDS="φ=phi")
    env.setdefault("FI_PROVIDER", "tcp")
    env["PETSC_OPTIONS"] = (env.get("PETSC_OPTIONS", "") + " -mat_partitioning_type current").strip()
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + [env.get("PYTHONPATH", "")]).strip(os.pathsep)
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twophaseflow_partition_mpi.py")
    argv = [sys.executable, helper] if ranks == 1 else [_mpiexec(), "-n", str(ranks), sys.executable, helper]
    out = subprocess.run(argv, capture_output=True, text=True, timeout=1800, env=env, cwd=str(tmp_path))
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    for deck, ymf in (("pressure.csv", "ymf_pressure.csv"), ("height2.csv", "ymf_height2.csv")):
        assert (tmp_path / ymf).read_text() == (tmp_path / deck).read_text(), ymf
    rows = (tmp_path / "boundary_integrals.csv").read_text().splitlines()
    ymf_rows = (tmp_path / "ymf_forces.csv").read_text().splitlines()
    assert ymf_rows[0] == rows[0] and len(ymf_rows) == len(rows) > 1
    assert [r.replace(",y-,", ",1,", 1) for r in ymf_rows[1:]] == rows[1:]
