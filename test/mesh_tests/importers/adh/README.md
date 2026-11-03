# ADH mesh fixture

`angle_bt.3dm` is a 2D AdH/SMS mesh (`MESH2D`, `MESHNAME`, 392 `E3T` triangles, 250 `ND`
nodes, one nodestring `NS` and a `BEGPARAMDEF` block), copied unchanged from the public
repository erdc/AdhModel:

https://github.com/erdc/AdhModel/blob/2c5d49dd4cca484a6c46ded6e1f6dec25db4722c/tests/test_files/angle_bt.3dm

Copyright (c) 2019, US Army Engineer Research and Development Center. Redistributed under the
BSD 3-Clause License of that repository; its text is in `LICENSE.AdhModel` beside this file.

It is used by `test/mesh_tests/test_importers.py` to test proteus's ADH and 2DM readers on a
file written by the tools ADH users use, not by proteus.
