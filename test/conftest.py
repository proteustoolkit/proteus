"""Test-suite wide fixtures."""
import importlib.util
import os
import subprocess
import sys
import time

import pytest

_TEST_ROOT = os.path.dirname(os.path.abspath(__file__)) + os.sep


def pytest_configure(config):
    """Under pytest-xdist, run each worker in a directory of its own.

    Tests write proteus.log, <so.name>.h5, matrix dumps and other files under
    fixed names relative to the working directory. Workers sharing one
    directory read back each other's files, or ones another worker has just
    deleted. Run as CI does (python -m pytest -n auto from an output
    directory), each worker gets <that directory>/<worker id>.

    A test must not read what another test wrote: tests land on any worker, in
    any order. One that needs a mesh or an earlier run makes it itself.
    """
    import signal
    import threading
    _register_exit_without_finalize()
    config.addinivalue_line('markers', 'mumps(reason): needs a PETSc built with MUMPS '
                            '(skipped without it, except under GitHub Actions)')
    if (threading.current_thread() is threading.main_thread()
            and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL):
        signal.signal(signal.SIGTERM, _end_launches_on_sigterm)
    worker = getattr(config, "workerinput", {}).get("workerid")
    if worker:
        path = os.path.join(os.getcwd(), worker)
        os.makedirs(path, exist_ok=True)
        os.chdir(path)

# Model modules a test imports at collection can call Comm.get(), which starts
# PETSc with Comm.argv, by default all of sys.argv: pytest's own options would
# reach PETSc. Start it with none instead, as importing proteus.iproteus did.
# (This sets a list; it does not start PETSc. A test that wants options of
# its own still sets Comm.argv before anything starts it.)
from proteus import Comm, runtime  # noqa: E402
Comm.argv = runtime.petsc_argv()


# Optional dependencies. Chrono and PUMI are compiled into extensions only when the build found
# them (PROTEUS_SKIP_CHRONO / PROTEUS_SKIP_PUMI, or no install to find), so a build without one
# has no such extension and every test that imports it fails at collection or at its first line.
# That says nothing about the code under test, so those tests are not collected, and the report
# header names what is missing and why. A capability is probed by the extension itself, not by
# the environment variable, so the answer is what the installed proteus can actually do.
#
# Tests that need one live under the paths listed here; a new one goes in this table.
OPTIONAL_DEPENDENCIES = {
    'chrono': dict(
        extension='proteus.mbd.CouplingFSI',
        paths=('AddedMass', 'FSI', 'test_mbd_chrono.py')),
    'pumi': dict(
        extension='proteus.MeshAdaptPUMI.MeshAdapt',
        paths=('MeshAdaptPUMI', 'test_gauges_pumi_mpi.py')),
}


def _has_extension(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_MISSING = {dep: spec for dep, spec in OPTIONAL_DEPENDENCIES.items()
            if not _has_extension(spec['extension'])}


def pytest_ignore_collect(collection_path, config):
    rel = os.path.relpath(str(collection_path), _TEST_ROOT[:-1])
    top = rel.split(os.sep)[0]
    for spec in _MISSING.values():
        if top in spec['paths']:
            return True
    return None


# MUMPS. Every CI leg's PETSc (conda-forge) has it; a PETSc built without it (the fleet's
# pip-built dev_pip prefix, a hand-built PETSc) cannot run a test that names MUMPS or that needs
# the direct solve it stands for: the singular and saddle-point systems whose fallback,
# `linear_solver: petsc` with PETSc's default GMRES+ILU(0), fails or diverges (a known gap:
# proteus has no scalable no-MUMPS solver for them yet). Such a test carries
# @pytest.mark.mumps(reason) and is skipped there, with the reason in the summary, instead of
# failing 22 times a night. Under GitHub Actions nothing is skipped: a CI environment that lost
# MUMPS must fail, not go quietly green. The probe runs once, in a subprocess (importing petsc4py
# here would start PETSc before a test could give it its own options), and only when a marked test
# is about to run.
_MUMPS = []


def petsc_has_mumps():
    """Whether the PETSc that petsc4py loads was built with MUMPS (probed once per process)."""
    if not _MUMPS:
        probe = ("import petsc4py; petsc4py.init([]); from petsc4py import PETSc; "
                 "print(int(PETSc.Sys.hasExternalPackage('mumps')))")
        try:
            out = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True,
                                 timeout=120).stdout.strip()
            _MUMPS.append(out.endswith('1'))
        except (OSError, subprocess.SubprocessError):
            _MUMPS.append(True)     # cannot tell: run the test and let it say
    return _MUMPS[0]


def pytest_runtest_setup(item):
    marker = item.get_closest_marker('mumps')
    if marker is None or os.environ.get('GITHUB_ACTIONS') == 'true':
        return
    if not petsc_has_mumps():
        why = marker.args[0] if marker.args else marker.kwargs.get('reason', 'it solves with MUMPS')
        pytest.skip('this PETSc was built without MUMPS, and ' + why)


def pytest_report_header(config):
    lines = []
    if _MISSING:
        lines.append('optional dependencies not built, their tests are not collected: '
                     + ', '.join('%s (no %s)' % (dep, spec['extension']) for dep, spec in sorted(_MISSING.items())))
    if _STALE_MPI_NAMES:
        lines.append('stale MPI names (macOS): ' + _STALE_MPI_NAMES[0])
    return lines or None


# On macOS, the stale MPICH singleton shm names and libomp files of this user are removed when a
# pytest session starts and when it ends (mpi_stale_names.py: a launch the suite kills leaves its
# names behind, and on a shared Mac they abort another user's MPI_Init when the pid comes round).
# Only names of pids that are not running are touched, so a concurrent session is safe. Once per
# session, in the controller (not in each xdist worker); about a second.
# PROTEUS_TEST_KEEP_STALE_MPI_NAMES=1 turns it off.
_STALE_MPI_NAMES = []


def _sweep_stale_mpi_names(config):
    if (sys.platform != 'darwin' or hasattr(config, 'workerinput')
            or os.environ.get('PROTEUS_TEST_KEEP_STALE_MPI_NAMES')):
        return None
    try:
        spec = importlib.util.spec_from_file_location('_proteus_mpi_stale_names',
                                                      os.path.join(_TEST_ROOT, 'mpi_stale_names.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.report(module.sweep())
    except Exception as exc:        # never fail a session over housekeeping
        return 'sweep failed: %r' % (exc,)


def pytest_sessionstart(session):
    message = _sweep_stale_mpi_names(session.config)
    if message:
        _STALE_MPI_NAMES[:] = [message]


_SESSION = []


def pytest_sessionfinish(session, exitstatus):
    _sweep_stale_mpi_names(session.config)
    _SESSION[:] = [session]


# SuperLU_DIST's finalize defect (conda-forge superlu_dist 9.2.1, PETSc 3.25.4, Open MPI 5.0.10): a process
# that factored a matrix with SuperLU_DIST frees its grid communicator from inside the attribute-delete callback
# of the communicator PetscFinalize is freeing (PetscFinalize -> MPI_Comm_free -> Petsc_Superlu_dist_keyval_DeleteFn
# -> superlu_gridexit -> MPI_Comm_free; valgrind: invalid reads in ompi_comm_free). The heap is damaged and glibc
# aborts the process at exit ("malloc(): unsorted double linked list corrupted", exit 134), or, in a worker, the
# worker; test_stokesDC_TH_2D.py::test_02_FullRun alone: about 1 run in 8. About twenty test files use
# superlu_dist. The result of the tests is already decided when this happens, so a pytest process (the main one,
# and each xdist worker, which exits through the same path) ends the way pytest decided, without running PETSc's
# finalize: it flushes its streams and calls os._exit with the session's exit status. What this hides: any other
# defect in PETSc/MPI finalization, PETSc's finalize-time reports (-log_view, unused options, leaked objects) and
# atexit handlers registered before this one. It does not touch the MPI jobs the tests launch under mpiexec.
# PROTEUS_TEST_FINALIZE=1 runs the normal finalize (to see the defect, or any other one, again).
def _exit_without_finalize():
    if os.environ.get('PROTEUS_TEST_FINALIZE') or not _SESSION:
        return
    status = _SESSION[0].exitstatus
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(int(status))


def _register_exit_without_finalize():
    import atexit
    atexit.register(_exit_without_finalize)


# mpiexec itself sometimes fails to start on a CI runner (the macOS Intel runner with Open MPI 5,
# mostly). Either PMIx cannot find a network interface for a moment ("ptl_tool: problems getting
# address for index 0 (kernel index -1)", exit status 213), or a rank's MPI_Init cannot reach the
# others ("It looks like MPI_INIT failed for some reason ... PML add procs failed --> Returned
# "Not found" (-13)", exit status 14). Every launch in that second fails, whatever it was testing,
# and no rank has run anything of the job when it happens, so launching it again is safe; any
# other failure is left alone.
#
# MPICH (5.x, macOS): a process started without mpiexec, a singleton, names its shared memory
# /mpich_shm_<hex pid>_0. A killed MPI process leaves that name behind until reboot, and when its
# pid comes round again under another user (a shared self-hosted Mac: the runners' user, people,
# agents), MPI_Init cannot open it and aborts, exit status 15, at
# "MPIDI_POSIX_comm_bootstrap(288).......: Out of memory" (mpi_stale_names.py has the mechanism).
# Nothing has run; a relaunch has another pid, so another name. chewbacca lost five tests to it on
# 2026-11-04 (`pro run`, `pro dis`, the one-rank VPP helper), on both runners. Launches under
# mpiexec are immune: hydra's names hash a random number.
MPI_LAUNCH_FAILURES = ('ptl_tool: problems getting address', 'It looks like MPI_INIT failed',
                       'MPIDI_POSIX_comm_bootstrap')


def mpi_launch_never_started(result):
    """True for a ``subprocess.CompletedProcess`` whose mpiexec failed before any rank ran."""
    err = getattr(result, 'stderr', None)
    if isinstance(err, bytes):
        err = err.decode(errors='replace')
    return (result.returncode != 0 and isinstance(err, str)
            and any(text in err for text in MPI_LAUNCH_FAILURES))


# A helper that has finished its work can still die at interpreter exit -- a SIGSEGV or SIGABRT
# in the teardown of PETSc, MPI or an extension, after its last line of output -- and on the
# macOS Intel runner it does, intermittently (test_parallel_space_multigrid's helper printed its
# complete RESULT line and then "exited on signal 11"). Such a launch is run once more; if the
# second attempt also dies, the test fails as it would have. Every relaunch is reported as a
# RuntimeWarning naming the command and the signal, so the count stays visible in the summary
# rather than vanishing into a green run, and the launched interpreters run with
# PYTHONFAULTHANDLER=1 so a crash prints the Python stack it happened in.
CRASH_SIGNALS = ('SIGSEGV', 'SIGABRT', 'SIGBUS', 'SIGILL', 'SIGFPE')
CRASH_MESSAGES = ('exited on signal', 'Segmentation fault', 'Bus error', 'Abort trap')


def died_by_signal(result):
    """The signal name a ``subprocess.CompletedProcess`` died by, or None."""
    import signal
    rc = result.returncode
    number = -rc if rc < 0 else rc - 128 if 128 < rc < 160 else None
    if number is not None:
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = None
        if name in CRASH_SIGNALS:
            return name
    err = getattr(result, 'stderr', None)
    if isinstance(err, bytes):
        err = err.decode(errors='replace')
    if rc != 0 and isinstance(err, str) and any(text in err for text in CRASH_MESSAGES):
        return 'a signal (see stderr)'
    return None


def scaled_timeout(timeout):
    """``timeout`` scaled for the machine, by PROTEUS_TEST_TIMEOUT_SCALE.

    The timeouts the tests give their subprocesses are sized for an idle workstation. A CI
    runner that is starved -- the 4-core macOS Intel runner took 2 h 15 min for a suite that
    takes 15 min on Linux, and a 4-rank launch inside it hit its 300 s budget -- needs more,
    and a timeout that fires there reports a hang that was not one. The scale is a number, or
    ``auto``: the one-minute load average per core, clamped to [1, 4], read at launch.
    """
    if timeout is None:
        return None
    setting = os.environ.get('PROTEUS_TEST_TIMEOUT_SCALE', '1').strip().lower()
    if setting == 'auto':
        try:
            load = os.getloadavg()[0] / max(os.cpu_count() or 1, 1)
        except (OSError, AttributeError):
            load = 1.0
        scale = min(max(load, 1.0), 4.0)
    else:
        try:
            scale = float(setting)
        except ValueError:
            scale = 1.0
    return timeout * scale


# A test's MPI launch is on one host, so Open MPI's TCP traffic between its ranks goes over the loopback interface only.
# On ceds-model a 2-rank Open MPI launch hung for its whole 1800 s budget (run 42114715099 on ceds-model-2): shared
# memory failed for one peer (an open(2) ENOENT), the ranks fell back to the TCP BTL, and a connection came from the
# Docker bridge's address (172.17.0.1), which the peer dropped. Measured with Open MPI 5.0.11 (proteus-ci-main) on
# ceds-model and spark2 and 5.0.10 on the Mac, 4 ranks with the TCP BTL forced (OMPI_MCA_btl=tcp,self), the
# fallback's path: nothing set hangs on both Linux hosts (killed at 40 s); OMPI_MCA_btl_tcp_if_include=127.0.0.1/8
# runs clean on all three, 10 of 10, and the same with shared memory on. The exclude lists that were proposed do not
# work everywhere: docker0,lo on ceds-model leaves only interfaces Open MPI cannot pair ("Unable to find reachable
# pairing", 12 warnings per launch, and an abort under TCP alone); docker0 alone still hangs on spark2, which has a
# second bridge. An interface name is not portable either (lo on Linux, lo0 on macOS); the subnet is. The launcher's
# own channel (PRRTE) is left alone: it ignores OMPI_MCA_oob_*, segfaults on any two-entry PRTE_MCA_prte_if_exclude,
# and refuses a loopback-only PRTE_MCA_prte_if_include on Linux. A default only: a host that chose its interfaces,
# by an include or an exclude (Open MPI refuses both at once), keeps its choice. MPICH reads neither variable.
OPEN_MPI_TCP_LOOPBACK = ('OMPI_MCA_btl_tcp_if_include', '127.0.0.1/8')


def launched_environment(env):
    """The environment a test's subprocess runs in: the caller's, with faulthandler on and Open
    MPI's TCP traffic on the loopback interface (OPEN_MPI_TCP_LOOPBACK), unless the caller chose."""
    env = dict(os.environ if env is None else env)
    env.setdefault('PYTHONFAULTHANDLER', '1')
    if 'OMPI_MCA_btl_tcp_if_exclude' not in env:
        env.setdefault(*OPEN_MPI_TCP_LOOPBACK)
    return env


# A launch that is cut short must take its ranks with it, and under Open MPI 5 it did not. prterun puts every rank in a
# process group of its own (inside prterun's session), so subprocess.run's timeout, which SIGKILLs prterun alone, left
# the ranks running, reparented to init (measured on ceds-model: three of three alive after the timeout). A cancelled
# CI job did the same: on 2026-11-03 a rank of bc_transfers_mpi.py ran for 10 h 43 min on a self-hosted runner after
# the runner's clean-up had killed its prterun. (Under MPICH every rank takes a session of its own, but dies with its
# hydra proxy, which stays in the launcher's process group.) So every launch runs in a session of its own, and when it
# times out, is interrupted or exits non-zero, the launcher's process group and then whatever is left in its session
# are terminated, then killed. A SIGTERM to the pytest process (a cancelled job) ends the launches still running.
_ACTIVE_SESSIONS = set()


def session_members(sid):
    """The pids of the processes in session ``sid``, this process excluded."""
    try:
        candidates = [int(name) for name in os.listdir('/proc') if name.isdigit()]
    except OSError:     # macOS: no /proc
        ps = subprocess.Popen(['ps', '-A', '-o', 'pid='], stdout=subprocess.PIPE, text=True)
        candidates = [int(pid) for pid in ps.communicate()[0].split()]
    members = []
    for pid in candidates:
        try:
            if pid != os.getpid() and os.getsid(pid) == sid:
                members.append(pid)
        except OSError:
            pass
    return members


def end_session(sid, process=None, grace=5.0):
    """Terminate, then kill, process group ``sid`` and every process in session ``sid``.

    ``process`` is the session leader's Popen, reaped as it goes so that its zombie does not
    count as a survivor. Never acts on this process's own session.
    """
    import signal
    if sid == os.getsid(0):
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(sid, sig)
        except OSError:
            pass
        for pid in session_members(sid):
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        deadline = time.monotonic() + wait
        while True:
            if process is not None:
                process.poll()
            left = [pid for pid in session_members(sid)
                    if process is None or pid != process.pid or process.returncode is None]
            if not left or time.monotonic() > deadline:
                break
            time.sleep(0.1)
        if not left:
            return


def run_in_own_session(*popenargs, input=None, capture_output=False, timeout=None, check=False, **kwargs):
    """``subprocess.run``, with the launch in a session of its own that is ended (end_session) if
    the launch times out, is interrupted or exits non-zero: ranks a dead launcher left go too."""
    if input is not None:
        kwargs['stdin'] = subprocess.PIPE
    if capture_output:
        kwargs['stdout'] = kwargs['stderr'] = subprocess.PIPE
    own = kwargs.setdefault('start_new_session', True)
    stdout = stderr = None
    with subprocess.Popen(*popenargs, **kwargs) as process:
        if own:
            _ACTIVE_SESSIONS.add(process.pid)
        try:
            try:
                stdout, stderr = _communicate(process, input, timeout, own)
            except subprocess.TimeoutExpired as expired:
                end_session(process.pid, process) if own else process.kill()
                try:
                    expired.stdout, expired.stderr = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    pass    # something outside the session still holds the pipes; keep what was read
                raise
            except BaseException:
                end_session(process.pid, process) if own else process.kill()
                raise
            if own and process.returncode != 0:
                end_session(process.pid, process)
        finally:
            _ACTIVE_SESSIONS.discard(process.pid)
    if check and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, process.args, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _communicate(process, input, timeout, own):
    """``process.communicate(input, timeout)``, except that once the launcher has exited, a process
    it left in its session that still holds the output pipes is not waited for: the session is
    ended and the output read. (A rank whose launcher died would otherwise hold the launch open
    until its timeout, or for ever without one.)"""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        step = 1.0 if deadline is None else max(0.0, min(1.0, deadline - time.monotonic()))
        try:
            return process.communicate(input, timeout=step)
        except subprocess.TimeoutExpired:
            input = None    # sent with the first call; a retry may not pass it again
            if own and process.poll() is not None:
                end_session(process.pid, process)
                return process.communicate(timeout=10)
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(process.args, timeout)


def _end_launches_on_sigterm(signum, frame):
    import signal
    for sid in list(_ACTIVE_SESSIONS):
        end_session(sid, grace=1.0)
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


@pytest.fixture
def case_copy(tmp_path):
    """``case_copy(directory)``: a copy of a case directory (its decks and inputs) under tmp_path.

    A test that runs a case in its own source directory leaves the run's output there (meshes,
    logs, .h5 and gauge files), so a checkout looks edited after a run and the next run starts from
    the last one's files. Run the case in the copy instead."""
    import shutil

    def copy(directory):
        dest = tmp_path / os.path.basename(os.path.normpath(str(directory)))
        shutil.copytree(str(directory), str(dest), ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        return dest
    return copy


@pytest.fixture
def launch_policy():
    """The launch policy's pieces, for the tests of it (a conftest is not importable under
    --import-mode=importlib, so they reach them through this fixture)."""
    import types
    return types.SimpleNamespace(scaled_timeout=scaled_timeout, died_by_signal=died_by_signal,
                                 launched_environment=launched_environment, retrying_run=retrying_run,
                                 run_in_own_session=run_in_own_session, session_members=session_members,
                                 os=os)


def retrying_run(run):
    """``run`` (a ``subprocess.run``), hardened for a shared runner:

    * launched again, up to three times in all, if an mpiexec never started
      (MPI_LAUNCH_FAILURES); the pause before a relaunch is PROTEUS_TEST_MPI_RETRY_DELAY
      seconds (default 3), doubled before the last attempt;
    * launched once more, with a RuntimeWarning, if it died by a signal (died_by_signal);
    * its ``timeout`` scaled by PROTEUS_TEST_TIMEOUT_SCALE (scaled_timeout);
    * run with PYTHONFAULTHANDLER=1 (launched_environment);
    * given PROTEUS_TEST_DUMP_AFTER, 80% of its timeout, which the multilevel helpers use to
      dump every rank's stack before the launch is killed (mpi_failfast), and the stderr of a
      launch that did time out is written out, so the report says where it was stuck.

    A function of ``run``, rather than closed over ``subprocess.run`` inside the fixture, so
    that a test can hand it a stand-in and look at what it was called with.
    """
    import warnings

    def run_with_retry(cmd, *args, **kwargs):
        if 'timeout' in kwargs:
            kwargs['timeout'] = scaled_timeout(kwargs['timeout'])
        kwargs['env'] = launched_environment(kwargs.get('env'))
        if kwargs.get('timeout'):
            kwargs['env'].setdefault('PROTEUS_TEST_DUMP_AFTER', str(int(0.8 * kwargs['timeout'])))
        crashed_once = False
        attempt = 0
        while True:
            try:
                result = run(cmd, *args, **kwargs)
            except subprocess.TimeoutExpired as expired:
                stderr = expired.stderr
                if isinstance(stderr, bytes):
                    stderr = stderr.decode(errors='replace')
                sys.stderr.write('launch timed out after %s s; stderr so far:\n%s\n'
                                 % (expired.timeout, (stderr or '')[-6000:]))
                raise
            if mpi_launch_never_started(result) and attempt < 2:
                time.sleep(float(os.environ.get('PROTEUS_TEST_MPI_RETRY_DELAY', 3.0)) * (attempt + 1))
                attempt += 1
                continue
            signal_name = died_by_signal(result)
            if signal_name is not None and not crashed_once:
                crashed_once = True
                warnings.warn(
                    'relaunching %s: it died by %s (exit status %d); the launched interpreter '
                    'ran with PYTHONFAULTHANDLER=1, so its stderr holds the Python stack if the '
                    'crash was in Python: %s'
                    % (' '.join(str(c) for c in cmd), signal_name, result.returncode,
                       (result.stderr if isinstance(result.stderr, str) else '')[-1500:]),
                    RuntimeWarning, stacklevel=2)
                time.sleep(float(os.environ.get('PROTEUS_TEST_MPI_RETRY_DELAY', 3.0)))
                continue
            return result
    return run_with_retry


@pytest.fixture(autouse=True)
def _retry_an_mpi_launch_that_never_started(monkeypatch):
    """Every ``subprocess.run`` from a test goes through :func:`retrying_run`, each launch in a
    session of its own (:func:`run_in_own_session`)."""
    monkeypatch.setattr(subprocess, 'run', retrying_run(run_in_own_session))


@pytest.fixture(autouse=True)
def _restore_petsc_options():
    """Put PETSc's options database back as it was before each test.

    The database is a process-wide singleton: an option a test sets (directly,
    or through a numerics module that writes it at import) otherwise stays set
    for every later test in the process, and a test's outcome can depend on
    which tests ran before it. CI runs each test in a fork (--forked), which
    hides this; a plain local pytest run does not.

    Only when petsc4py is already loaded: importing it here would initialize
    PETSc before a test could hand it its own arguments.
    """
    if "petsc4py.PETSc" not in sys.modules:
        yield
        return
    from petsc4py import PETSc
    database = PETSc.Options()
    before = database.getAll()
    yield
    after = database.getAll()
    for name in after:
        if name not in before:
            database.delValue(name)
    for name, value in before.items():
        if after.get(name) != value:
            database.setValue(name, value)


@pytest.fixture
def opts():
    """A fresh proteus.runtime.RunOptions, the options NS_base reads, with
    Proteus started (proteus.runtime.start: PETSc with no options, and
    proteus.log open in the working directory).

    It replaces ``from proteus.iproteus import opts``, which was one object
    shared by every test that imported it.
    """
    from proteus import runtime
    runtime.start()
    return runtime.RunOptions()


_DEFAULTS_MODULES = ("proteus.default_p", "proteus.default_n", "proteus.default_so", "proteus.default_s")


def _copied_vars(module):
    """A module's namespace, its dicts and lists copied (a deck that sets
    n.femSpaces[0] changes default_n's dict, not the name)."""
    return dict((key, (value.copy() if type(value) in (dict, list) else value))
                for key, value in vars(module).items())


def _snapshot_defaults():
    """The namespaces of default_p/default_n/default_so/default_s (_copied_vars).

    proteus.defaults goes first: when it is first imported it takes its lists of
    physics and numerics option names from dir(default_p) and dir(default_n), so
    first imported by a deck in the middle of changing them, it would count that
    deck's names (p.rdomain) as Proteus options, and Physics_base() would fail
    once they were put back. (It imports petsc4py, as the first test module
    collected would; Comm.init hands Comm.argv to the options database when PETSc
    has already started.)"""
    import importlib
    import proteus.defaults  # noqa: F401
    return dict((name, _copied_vars(importlib.import_module(name))) for name in _DEFAULTS_MODULES)


def _restore_defaults(snapshot):
    for name, saved in snapshot.items():
        namespace = vars(sys.modules[name])
        for key in list(namespace):
            if key not in saved:
                del namespace[key]
        namespace.update(saved)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    """Put default_p/default_n/default_so back after each test module is imported.

    Some test modules build their problem at import, and the decks they import
    reload proteus.default_p/default_n and set attributes on them (the BDM and
    mass-matrix templates: n.triangleOptions = "Yp", p.nd, ...). Importing is
    collecting, which happens before any test and outside the per-test restore
    below, so every test file collected after one of them star-imported its
    settings as Proteus's defaults: run reversed, poisson_2d, TADR, VOF, NCLS,
    vortex and CLSVOF+RANS2P failed. Forward, they were collected first and
    passed.
    """
    if not isinstance(collector, pytest.Module):
        yield
        return
    snapshot = _snapshot_defaults()
    yield
    _restore_defaults(snapshot)


@pytest.fixture(autouse=True)
def _restore_proteus_process_state():
    """Put proteus.iproteus.opts, the default_p/n/so/s modules, proteus.Context,
    sys.modules (for modules under test/) and the working directory back after
    each test.

    opts is a process-wide object that tests import by reference and set
    (opts.hotStart, opts.dataDir, ...); a value one test sets otherwise holds
    for every test after it.
    """
    cwd = os.getcwd()
    module = sys.modules.get("proteus.iproteus")
    saved = dict(vars(module.opts)) if module is not None else None
    # default_p/default_n/default_so/default_s are modules model files star-import
    # their defaults from; a test that assigns to one (POD did: numerics = default_n,
    # then numerics.subgridError = ...; poisson_2d sets so = default_so, so.sList),
    # or changes one of its dicts in place (nList[0].femSpaces[0] = ... on a deck
    # that star-imported femSpaces), changes every later model's defaults. Put
    # back with their dicts and lists as copies, so that a test needs no
    # reload(default_n) of its own. Only the ones already imported: importing
    # one here would import proteus before a test could set it up.
    defaults_snapshot = dict((name, _copied_vars(sys.modules[name]))
                             for name in _DEFAULTS_MODULES if name in sys.modules)
    context = sys.modules.get("proteus.Context")
    saved_context = ((context.context, context.contextOptionsString)
                     if context is not None else None)
    modules_before = set(sys.modules)
    yield
    # Model modules a test imports by bare name from its own directory
    # (twp_navier_stokes_p, cylinder, ...) share names across test directories;
    # left in sys.modules, the next test that imports that name from its own
    # directory gets this one instead (cylinder2D/ibm_rans2p got
    # conforming_rans3p's twp_navier_stokes_p: no RANS2P). Such modules
    # imported during the test go again. Package-relative ones
    # (HotStart_3P.NS_hotstart_so) have unique names and stay: tests keep and
    # reload them.
    for name in set(sys.modules) - modules_before:
        if "." in name:
            continue
        added = sys.modules.get(name)
        filename = getattr(added, "__file__", None) or ""
        if os.path.abspath(filename).startswith(_TEST_ROOT):
            del sys.modules[name]
    if saved is not None:
        vars(module.opts).clear()
        vars(module.opts).update(saved)
    _restore_defaults(defaults_snapshot)
    if saved_context is not None:
        # Context.contextOptionsString (parun -C) is read by every model module
        # that calls Context.Options; one a test leaves set reconfigures every
        # later model (IFEM's refinement=N renamed MCorr's output).
        context.context, context.contextOptionsString = saved_context
    os.chdir(cwd)
