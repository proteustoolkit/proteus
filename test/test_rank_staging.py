"""rank_staging: the ``*_mpi.py`` helpers stage a deck's files from every rank without a moment at which another rank
can import part of one (test_mcorr_partition_independence hung the GitHub aarch64 leg on 2026-11-06: one rank
star-imported a module another rank had just truncated, NameError, and the other rank waited 2809 s)."""
import ast
import glob
import importlib.util
import os
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# by path: CI's --import-mode=importlib does not put test/ on sys.path (the helpers, run as scripts, have it there)
_spec = importlib.util.spec_from_file_location("rank_staging", os.path.join(HERE, "rank_staging.py"))
rank_staging = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rank_staging)
A, B = "x = 1\n" * 20000, "y = 2\n" * 30000


def _read_while_writing(tmp_path, writer):
    path = str(tmp_path / "m.py")
    rank_staging.write(path, A)
    seen, stop = set(), threading.Event()

    def reader():
        while not stop.is_set():
            with open(path) as f:
                seen.add(len(f.read()))
    t = threading.Thread(target=reader)
    t.start()
    for text in (B, A, B, A):
        writer(path, text)
    stop.set()
    t.join()
    return seen


def test_a_reader_sees_only_whole_files(tmp_path):
    seen = _read_while_writing(tmp_path, lambda path, text: rank_staging.write(path, text, _pause=0.02))
    assert seen <= {len(A), len(B)}, sorted(seen)


def test_the_in_place_write_it_replaces_is_seen_part_written(tmp_path):
    """The control: open(path, "w") then write, held 20 ms in between as a descheduled rank would be."""
    def in_place(path, text):
        with open(path, "w") as f:
            time.sleep(0.02)
            f.write(text)
    seen = _read_while_writing(tmp_path, in_place)
    assert seen - {len(A), len(B)}, sorted(seen)


def test_copy_into_a_directory(tmp_path):
    src = tmp_path / "src.py"
    src.write_text(A)
    (tmp_path / "d").mkdir()
    assert rank_staging.copy(str(src), str(tmp_path / "d")) == str(tmp_path / "d" / "src.py")
    assert (tmp_path / "d" / "src.py").read_text() == A
    assert os.listdir(str(tmp_path / "d")) == ["src.py"]


# serial only (test_step_retry runs it without mpiexec)
SERIAL_HELPERS = {"step_retry_mpi.py"}
#: what a helper writes as its result (per rank or from rank 0), not a file another rank imports
OUTPUTS = (".json", ".csv", ".txt", ".npz", ".npy", ".h5")


def test_no_mpi_helper_stages_files_in_place():
    """Every rank runs a helper's module level: none may copy, write or append a deck file there except through
    rank_staging (results, by extension, writes under ``if rank ...`` and inside functions are not looked at)."""
    wrong = []
    for path in sorted(glob.glob(os.path.join(HERE, "**", "*_mpi.py"), recursive=True)):
        if os.path.basename(path) in SERIAL_HELPERS:
            continue
        tree = ast.parse(open(path).read())
        stack = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.If))]
        stack += [n for n in tree.body if isinstance(n, ast.If) and "rank" not in ast.unparse(n.test)]
        for top in stack:
            for node in ast.walk(top):
                if not isinstance(node, ast.Call):
                    continue
                name = ast.unparse(node.func)
                mode = [a for a in node.args[1:2] if isinstance(a, ast.Constant)]
                target = ast.unparse(node.args[0]) if node.args else ""
                if name in ("shutil.copy", "shutil.copyfile", "shutil.copy2") or (
                        name == "open" and mode and str(mode[0].value)[:1] in "wa"
                        and not any(ext in target for ext in OUTPUTS)):
                    wrong.append("%s:%d %s" % (os.path.relpath(path, HERE), node.lineno, ast.unparse(node)[:80]))
    assert wrong == [], "\n".join(wrong)
