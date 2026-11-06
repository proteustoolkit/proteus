"""Remove this user's stale MPICH singleton shared-memory names and libomp registration files (macOS).

    python -I test/mpi_stale_names.py

An MPICH (5.x) process started without mpiexec, a singleton, names its shared memory after its pid:
``/mpich_shm_<hex pid>_0`` and ``/mpich_vci_<hex pid>_0`` (posix_init.c; under mpiexec the name is a hash of
hydra's kvs name, which carries a random number, so launched ranks do not collide). MPI_Init opens the name
with O_EXCL and, if it exists, attaches to it. A process that is killed or aborts never unlinks it, and on
macOS a POSIX shm name lives until reboot. When the pid comes round again, a stale name of the same user is
attached to; another user's gives EACCES and MPI_Init aborts::

    MPIDI_POSIX_comm_bootstrap(288).......: Out of memory

libomp leaves ``/tmp/__KMP_REGISTERED_LIB_<pid>`` (mode 0600) the same way. Each user can remove only their
own names, so every user who runs MPI on a shared Mac has to: the CI runners do it before each job
(sdist.yml), and test/conftest.py does it at the start and end of every pytest session, which covers the
people and agents who run the suite there. macOS cannot list POSIX shm, so every pid's two names are probed
(about a second). Names of pids still alive are kept, so a running job is never touched.
"""
import errno
import glob
import os
import sys

PID_LIMIT = 100000          # macOS pids run 1..99998
PREFIXES = ('mpich_shm', 'mpich_vci')


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def sweep():
    """Unlink this user's stale names; return (shm removed, libomp files removed, other users' stale shm names)."""
    import _posixshmem
    me = os.getuid()
    removed = foreign = 0
    for pid in range(1, PID_LIMIT):
        for prefix in PREFIXES:
            name = '/%s_%x_0' % (prefix, pid)
            try:
                fd = _posixshmem.shm_open(name, os.O_RDONLY, 0)
            except OSError as e:
                foreign += e.errno == errno.EACCES
                continue
            try:
                uid = os.fstat(fd).st_uid
            finally:
                os.close(fd)
            if uid == me and not _alive(pid):
                try:
                    _posixshmem.shm_unlink(name)
                    removed += 1
                except OSError:
                    pass
    kmp = 0
    for path in glob.glob('/tmp/__KMP_REGISTERED_LIB_*'):
        try:
            pid = int(path.rsplit('_', 1)[1])
            if os.lstat(path).st_uid == me and not _alive(pid):
                os.unlink(path)
                kmp += 1
        except (OSError, ValueError):
            pass
    return removed, kmp, foreign


def report(counts):
    removed, kmp, foreign = counts
    return ('removed %d stale MPICH shm names and %d libomp files of this user; '
            '%d stale names belong to other users' % (removed, kmp, foreign))


if __name__ == '__main__':
    if sys.platform != 'darwin':
        sys.exit(0)
    print(report(sweep()))
