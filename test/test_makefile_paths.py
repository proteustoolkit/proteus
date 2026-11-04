"""Every test path the Makefile names exists (it pointed at proteus/tests/ after the tests moved to test/)."""
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_makefile_names_only_test_paths_that_exist():
    text = open(os.path.join(ROOT, "Makefile")).read()
    paths = sorted(set(re.findall(r"(?<![\w/])((?:proteus/)?tests?/[\w/.*-]+)", text)))
    assert paths, "found no test paths in the Makefile"
    missing = [p for p in paths if not glob.glob(os.path.join(ROOT, p))]
    assert not missing, "Makefile names paths that do not exist: %s" % missing
