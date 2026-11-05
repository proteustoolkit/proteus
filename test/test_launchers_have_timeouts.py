"""Every subprocess a test starts is given a timeout.

A launch with no timeout that hangs runs until the CI job is killed, and the job log says nothing
about where it was stuck. With one, retrying_run (test/conftest.py) scales it for the machine,
sets PROTEUS_TEST_DUMP_AFTER so a helper using mpi_failfast dumps every rank's stack first, and
on expiry ends the launch's whole process tree.

This checks the files pytest runs: every test_*.py and conftest.py under test/. (test/multilevel
has a stricter check of its own, over its helpers too.) ``subprocess.Popen`` takes no timeout,
so it is not checked: its ``wait``/``communicate`` is where one goes. A ``**kwargs`` that may
carry the timeout does not count; pass it by name.
"""
import ast
import os
import warnings

LAUNCHERS = ('run', 'call', 'check_call', 'check_output')
TEST_ROOT = os.path.dirname(os.path.abspath(__file__))


def launches_without_timeout(path):
    with open(path) as f, warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)      # old docstrings' escapes are not this test's business
        tree = ast.parse(f.read(), path)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in LAUNCHERS and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'subprocess'
                and not any(k.arg == 'timeout' for k in node.keywords)):
            yield '%s:%d' % (os.path.relpath(path, TEST_ROOT), node.lineno)


def files_pytest_runs():
    for root, dirs, files in os.walk(TEST_ROOT):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d != '__pycache__')
        for name in sorted(files):
            if name == 'conftest.py' or (name.startswith('test_') and name.endswith('.py')):
                yield os.path.join(root, name)


def test_every_launch_from_a_test_has_a_timeout():
    missing = []
    for path in files_pytest_runs():
        try:
            missing += launches_without_timeout(path)
        except SyntaxError:         # a Python 2 relic pytest cannot collect either
            pass
    assert not missing, 'subprocess launches with no timeout: ' + ', '.join(missing)


def test_the_check_sees_a_launch_without_one(tmp_path):
    sample = tmp_path / 'test_sample.py'
    sample.write_text('import subprocess\n'
                      'subprocess.run(["true"], timeout=10)\n'
                      'subprocess.check_call("true", shell=True)\n'
                      'subprocess.Popen(["true"]).wait(timeout=10)\n')
    assert [where.rsplit(':', 1)[1] for where in launches_without_timeout(str(sample))] == ['3']
