"""Stage a deck's files into the working directory of a test run under mpiexec, from every rank.

The ``*_mpi.py`` helpers run on every rank and each rank copies (and edits) the deck's files into the shared working
directory before importing them. Written in place (``open(path, "w").write(text)``, ``shutil.copy``, an append), one
rank truncates a file another rank is importing: that rank star-imports a truncated module and stops with NameError,
and the other ranks hang in their first collective (test_mcorr_partition_independence on the GitHub aarch64 runner,
2026-11-06: ``name 'RD_model' is not defined``, then a 2809 s timeout). Here every rank writes the same bytes to a name
of its own and renames it into place (``os.replace`` is atomic on POSIX), so a reader sees a whole file, the previous
one or this one, and the files are the same on every rank. No collective, so it can run before ``Comm.init``.
"""
import os
import time


def write(path, data, _pause=0.0):
    """Put ``data`` (str or bytes) at ``path`` without a moment at which another process can read part of it.
    ``_pause`` (seconds, tests of this module only) holds the written temporary file before the rename."""
    tmp = "%s.%d.staging" % (path, os.getpid())
    with open(tmp, "wb") as f:
        f.write(data.encode() if isinstance(data, str) else data)
    if _pause:
        time.sleep(_pause)
    os.replace(tmp, path)


def copy(src, dst):
    """``shutil.copy(src, dst)`` (dst a file or a directory), staged as ``write`` does."""
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    with open(src, "rb") as f:
        write(dst, f.read())
    return dst
