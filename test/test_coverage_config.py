"""The coverage configuration measures the proteus package, and every process a test starts.

Coverage is tracked for the code that ships (``proteus/``), not for the test or scripts
directories, so that code nothing runs is visible. Most of the suite's work happens in
processes the tests start -- ``parun``, the helpers run under ``mpiexec`` -- so the
configuration has to reach those too, or the report would call the archive's rank code
dead. These tests pin the configuration and check, with a small script, that a
subprocess measures itself.
"""
import os
import subprocess
import sys
import textwrap

import pytest

tomllib = pytest.importorskip("tomllib")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYPROJECT = os.path.join(REPO_ROOT, "pyproject.toml")


def coverage_config():
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)["tool"]["coverage"]


def test_the_package_is_measured_and_the_tests_are_not():
    run = coverage_config()["run"]
    assert run["source"] == ["proteus"]
    assert any(o.endswith("proteus/tests/*") for o in run["omit"])
    assert run["branch"] is True


def test_every_process_a_test_starts_is_measured():
    run = coverage_config()["run"]
    assert run["parallel"] is True
    assert "subprocess" in run["patch"], "only the pytest workers would be measured"


def test_the_installed_package_is_reported_as_the_checkout():
    paths = coverage_config()["paths"]["source"]
    assert paths[0] == "proteus/"
    assert any("site-packages/proteus" in p for p in paths[1:])


def test_a_subprocess_measures_itself(tmp_path):
    """With patch = subprocess, a child interpreter's lines reach the report."""
    coverage = pytest.importorskip("coverage")
    if tuple(int(x) for x in coverage.__version__.split(".")[:2]) < (7, 10):
        pytest.skip("coverage %s predates patch = subprocess" % coverage.__version__)
    package = tmp_path / "measured"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "child.py").write_text(textwrap.dedent("""
        def in_the_child():
            return "ran in the child"
        if __name__ == "__main__":
            print(in_the_child())
        """))
    parent = tmp_path / "parent.py"
    parent.write_text(textwrap.dedent("""
        import subprocess, sys
        print(subprocess.run([sys.executable, "-m", "measured.child"],
                             capture_output=True, text=True, check=True).stdout.strip())
        """))
    rc = tmp_path / ".coveragerc"
    rc.write_text(textwrap.dedent("""
        [run]
        source = measured
        parallel = true
        patch = subprocess
        """))
    # a clean environment for the nested run: under pytest-cov (the CI coverage leg) this process carries COV_CORE_*
    # and COVERAGE_* settings that would send the nested run's data to the outer measurement and leave nothing to report
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COV_CORE", "COVERAGE"))}
    env.update(PYTHONPATH=str(tmp_path), COVERAGE_RCFILE=str(rc))
    ran = subprocess.run([sys.executable, "-m", "coverage", "run", str(parent)],
                         cwd=str(tmp_path), capture_output=True, text=True, env=env, timeout=600)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "ran in the child" in ran.stdout
    subprocess.run([sys.executable, "-m", "coverage", "combine"], cwd=str(tmp_path),
                   capture_output=True, text=True, env=env, check=True, timeout=600)
    report = subprocess.run([sys.executable, "-m", "coverage", "report", "-m"],
                            cwd=str(tmp_path), capture_output=True, text=True, env=env,
                            check=True, timeout=600).stdout
    (line,) = [l for l in report.splitlines() if "child.py" in l]
    # the function body ran in the child, so it is not missing
    assert "100%" in line, report
