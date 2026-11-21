"""Two small guards on the files the mesh generators read and write.

* a triangle that exits 0 without leaving its ``.1.*`` files raises with the missing file's name, instead of an
  unchecked ``mv`` failing silently and a reader failing later;
* ``writePoly`` of the PLC and PSLG domains writes plain integers when the indices are numpy integers (under
  numpy 2 ``repr(np.int32(96))`` is ``np.int32(96)``, which tetgen and triangle cannot read).
"""
import os
import stat

import numpy as np
import pytest

from proteus import Domain, MeshTools


def _square(tmp_path, dtype=int):
    name = str(tmp_path / "sq")
    d = Domain.PlanarStraightLineGraphDomain(
        vertices=[[0., 0.], [1., 0.], [1., 1.], [0., 1.]],
        segments=[[dtype(a), dtype(b)] for a, b in ((0, 1), (1, 2), (2, 3), (3, 0))], name=name)
    d.polyfile = name
    return d, name


def test_triangle_that_leaves_no_files_raises_naming_the_file(tmp_path, monkeypatch):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "triangle").write_text("#!/bin/sh\nexit 0\n")
    (fake / "triangle").chmod(0o755)
    monkeypatch.setenv("PATH", str(fake) + os.pathsep + os.environ["PATH"])
    d, name = _square(tmp_path)
    d.writePoly(name)
    with pytest.raises(RuntimeError, match=r"did not leave .*sq\.1\.ele"):
        MeshTools.writeMeshFiles(d, MeshTools.getMeshOptions(d))


def test_writePoly_numpy_integers_are_plain(tmp_path):
    d, name = _square(tmp_path, np.int32)
    d.writePoly(name)
    assert "np." not in (tmp_path / "sq.poly").read_text()
    v = [[0., 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    facets = [[[np.int32(i) for i in f]] for f in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5),
                                                    (2, 3, 7, 6), (3, 0, 4, 7))]
    cube = Domain.PiecewiseLinearComplexDomain(vertices=v, facets=facets, name=str(tmp_path / "cube"))
    cube.writePoly(str(tmp_path / "cube"))
    assert "np." not in (tmp_path / "cube.poly").read_text()
