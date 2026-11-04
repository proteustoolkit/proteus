# ADH mesh fixture

`angle_bt.3dm` is a 2D AdH/SMS mesh (`MESH2D`, `MESHNAME`, 392 `E3T` triangles, 250 `ND`
nodes, one nodestring `NS` and a `BEGPARAMDEF` block), copied unchanged from the public
repository erdc/AdhModel:

https://github.com/erdc/AdhModel/blob/2c5d49dd4cca484a6c46ded6e1f6dec25db4722c/tests/test_files/angle_bt.3dm

Copyright (c) 2019, US Army Engineer Research and Development Center. Redistributed under the
BSD 3-Clause License of that repository; its text is in `LICENSE.AdhModel` beside this file.

It is used by `test/mesh_tests/test_importers.py` to test proteus's ADH and 2DM readers on a
file written by the tools ADH users use, not by proteus.

## Opt-in: the CHL test cases listed on the XMS Wiki

The XMS Wiki page "SMS:AdH" (section "Case Studies / Sample Problems") lists seven ADH simulations
"provided by CHL to test ADH and the SMS interface capabilities" (2007). Their terms of use are not
stated, so they are not in this repository. To run `test_importers.py`'s opt-in ADH tests, download and
unzip any of them into one directory and set `PROTEUS_ADH_TEST_FILES` to it:

| case | URL | mesh |
|---|---|---|
| Bayou_Sorrel | http://sms.aquaveo.com/ADH-bayousorrel.zip | 2D, 7040 nodes, 13325 `E3T`, two materials (checked by SHA-256) |
| Pool5 | http://sms.aquaveo.com/ADH-pool5.zip | 2D (not yet opened) |
| nb dis | http://sms.aquaveo.com/ADH-nb_dis.zip | straight flume (not yet opened) |
| op bt | http://sms.aquaveo.com/ADH-op_bt.zip | flume with a boat file (not yet opened) |
| db lde | http://sms.aquaveo.com/ADH-db_lde.zip | flume (not yet opened) |
| db ldh | http://sms.aquaveo.com/ADH-db_ldh.zip | flume (not yet opened) |
| db lid | http://sms.aquaveo.com/ADH-db_lid.zip | flume (not yet opened) |

```bash
PROTEUS_ADH_TEST_FILES=$HOME/adh-cases pytest test/mesh_tests/test_importers.py -k adh
```
