"""Every subprocess a test in this directory starts is given a timeout.

A launch with no timeout that deadlocks -- the 3-rank hexahedral MGM cycle did once, on
2026-10-31 -- runs until the CI job is killed, and the job log says nothing about where it
was stuck. With a timeout, retrying_run (test/conftest.py) scales it for the machine and sets
PROTEUS_TEST_DUMP_AFTER, so the helper's mpi_failfast dumps every rank's stack first.
"""
import ast
import os

LAUNCHERS = ('run', 'call', 'check_call', 'check_output', 'Popen')


def _launches_without_timeout(path):
    with open(path) as f:
        tree = ast.parse(f.read(), path)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in LAUNCHERS and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'subprocess'
                and not any(k.arg == 'timeout' for k in node.keywords)):
            yield '%s:%d' % (os.path.basename(path), node.lineno)


def test_every_launch_has_a_timeout():
    here = os.path.dirname(os.path.abspath(__file__))
    missing = [where for name in sorted(os.listdir(here)) if name.endswith('.py')
               for where in _launches_without_timeout(os.path.join(here, name))]
    assert not missing, 'subprocess launches with no timeout: ' + ', '.join(missing)
