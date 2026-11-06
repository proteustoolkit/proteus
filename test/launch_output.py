"""Run a helper (under mpiexec or not) for a test and, when it fails, keep everything it printed.

The partition-independence tests used to assert ``out.stdout[-2000:] + out.stderr[-2000:]``: a crash's backtrace (a
2-rank segfault in the twophaseflow fixture on ceds-model-4, 2026-11-05) was cut away, and a hang lost all of its
output (subprocess.TimeoutExpired). Here a failed launch writes the whole stdout and stderr to ``launch.stdout.txt`` and
``launch.stderr.txt`` in its working directory (under the test's tmp_path) and the assertion message names them, with
the exit status (and signal), the command and the last 8000 characters of stderr and 3000 of stdout.

Load it by path (CI runs pytest with --import-mode=importlib, which does not put test/ on sys.path).
"""
import os
import signal
import subprocess

STDERR_TAIL, STDOUT_TAIL = 8000, 3000


def _text(data):
    if data is None:
        return ""
    return data.decode(errors="replace") if isinstance(data, bytes) else data


def _status(returncode):
    """'exit status 139 (SIGSEGV)', 'killed by SIGSEGV', 'exit status 1'."""
    sig = -returncode if returncode < 0 else returncode - 128 if returncode > 128 else None
    try:
        name = signal.Signals(sig).name if sig else None
    except ValueError:
        name = None
    if returncode < 0:
        return "killed by %s" % (name or "signal %d" % -returncode)
    return "exit status %d%s" % (returncode, " (%s?)" % name if name else "")


def report(cwd, what, argv, stdout, stderr):
    """The assertion message for a failed launch; writes the full output under ``cwd``."""
    stdout, stderr = _text(stdout), _text(stderr)
    paths = []
    for name, text in (("launch.stdout.txt", stdout), ("launch.stderr.txt", stderr)):
        path = os.path.join(str(cwd), name)
        try:
            with open(path, "w") as f:
                f.write(text)
            paths.append(path)
        except OSError as e:
            paths.append("%s (not written: %s)" % (path, e))
    return ("%s: %s\nfull output: %s\n--- stderr, last %d of %d characters ---\n%s\n"
            "--- stdout, last %d of %d characters ---\n%s"
            % (" ".join(str(a) for a in argv), what, ", ".join(paths), STDERR_TAIL, len(stderr),
               stderr[-STDERR_TAIL:], STDOUT_TAIL, len(stdout), stdout[-STDOUT_TAIL:]))


def run(argv, cwd, env, timeout):
    """subprocess.run(argv, capture_output=True, text=True, ...) that raises AssertionError with ``report``'s message
    on a nonzero exit status or a timeout, and returns the CompletedProcess otherwise."""
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env, cwd=str(cwd))
    except subprocess.TimeoutExpired as e:
        raise AssertionError(report(cwd, "timed out after %s s" % timeout, argv, e.stdout, e.stderr)) from None
    if out.returncode != 0:
        raise AssertionError(report(cwd, _status(out.returncode), argv, out.stdout, out.stderr))
    return out
