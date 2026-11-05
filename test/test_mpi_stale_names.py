"""The sweep of stale MPICH singleton shm names and libomp files (test/mpi_stale_names.py, macOS)."""
import importlib.util
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='POSIX shm names outlive processes on macOS only')


@pytest.fixture
def stale_names():
    spec = importlib.util.spec_from_file_location(
        '_mpi_stale_names_under_test', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mpi_stale_names.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plant(pid):
    import _posixshmem
    name = '/mpich_shm_%x_0' % pid
    fd = _posixshmem.shm_open(name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    os.close(fd)
    return name


def _exists(name):
    import _posixshmem
    try:
        os.close(_posixshmem.shm_open(name, os.O_RDONLY, 0))
        return True
    except FileNotFoundError:
        return False


def test_a_dead_pids_name_is_removed_and_a_running_ones_kept(stale_names):
    import _posixshmem
    dead = subprocess.Popen(['/usr/bin/true'])
    dead.wait()
    running = subprocess.Popen(['/bin/sleep', '60'])
    names = []
    try:
        names = [_plant(dead.pid), _plant(running.pid)]
        removed, _, _ = stale_names.sweep()
        assert removed >= 1
        assert not _exists(names[0])
        assert _exists(names[1])
    finally:
        running.kill()
        running.wait()
        for name in names:
            try:
                _posixshmem.shm_unlink(name)
            except OSError:
                pass


def test_the_sweep_reports_what_it_did(stale_names):
    assert stale_names.report((3, 2, 1)) == ('removed 3 stale MPICH shm names and 2 libomp files of this user; '
                                             '1 stale names belong to other users')
