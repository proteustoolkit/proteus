"""``pro``: the Proteus command line, one command with subcommands.

``pro run`` runs a model, as ``parun`` does. The others mesh it (``msh``),
report where it stands (``st``), open a shell on it (``sh``), and take a
YMF spec through its steps: check it (``sp``), write its weak form
(``wk``) and its discrete form (``dis``). See PRO_CLI_RESEARCH.md in the
planning notes.

Each command has a short name, which is what ``pro`` prints in hints and
records, and a whole-word long name that is the same command: ``pro wk``
is ``pro weak``, ``pro dis`` is ``pro discrete`` (:data:`COMMANDS`).
Neither is matched by a prefix, and no option is either.

Each subcommand's module imports what it needs inside its ``run``, never
at module level, so ``pro --help`` and argument errors are answered without
starting MPI or PETSc. That matters on login nodes, where starting MPI may
be slow or forbidden.

Commands that need the front-end extras (``ymf[symbolic]``: sympy,
strictyaml) are listed whether or not those are installed. Without them a
command says which extra it needs, instead of argparse reporting an
unknown command.
"""
import argparse
import importlib.util
import sys

__all__ = ["main", "build_parser", "missing", "json_stdout", "load_error", "load_errors",
           "command", "add_command", "COMMANDS"]

#: short name -> (module in this package, long name). Each command has a
#: short name of two or three letters, which is what pro prints, and a long
#: name of one whole word, its alias: the same command, options and help.
#: The short name abbreviates the long one, and the long one names what the
#: command produces (PRO_COMMAND_NAMES_CAMPAIGN.md in the planning notes).
#: Each module defines ``add_parser(subparsers)``, which calls
#: :func:`add_command`, and ``run(args) -> int``.
COMMANDS = {
    "run": ("run", "run"),
    "msh": ("msh", "mesh"),
    "st": ("status", "status"),
    "sh": ("shell", "shell"),
    "sp": ("spec", "spec"),
    "wk": ("weak", "weak"),
    "dis": ("dis", "discrete"),
}


def _names(module):
    for short, (mod, long) in COMMANDS.items():
        if mod == module:
            return short, long
    raise KeyError(module)


def command(module):
    """What ``pro`` prints for the command a module implements: ``"pro wk"``
    for ``weak``. Hints are copied as commands, so they give the short name."""
    return "pro " + _names(module)[0]


def add_command(subparsers, module, **kwargs):
    """The subparser for ``module``'s command, under its short name with its
    long name as an alias, and :func:`parser_kwargs`."""
    short, long = _names(module)
    aliases = [long] if long != short else []
    return subparsers.add_parser(short, aliases=aliases, **dict(parser_kwargs(), **kwargs))

#: front-end subcommands -> (help, modules they import); not implemented
#: yet, but listed so the help shows where they will go. None at present.
FRONTEND = {}


def missing(*modules):
    """The modules in ``modules`` that cannot be imported, without
    importing them."""
    absent = []
    for name in modules:
        try:
            found = importlib.util.find_spec(name) is not None
        except ImportError:  # a parent package is absent
            found = False
        if not found:
            absent.append(name)
    return absent


def json_stdout():
    """A text stream on the process's stdout, kept for output a program reads
    (``--json``), with file descriptor 1 pointed at stderr from here on.

    MPI stacks write diagnostics to stdout from C -- Open MPI's UCX layer,
    for one, when it starts on a machine whose network it cannot use -- and
    a Python-level redirect of ``sys.stdout`` does not reach them. Call this
    before importing anything that starts MPI; whatever is printed after it,
    by Python or by C, at start-up or at exit, goes to stderr, and only what
    is written to the returned stream reaches stdout.
    """
    import os
    sys.stdout.flush()
    kept = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(kept, "w", encoding="utf-8")


def load_error(exc, spec):
    """One line for a spec that does not load: a ValueError (the loader's,
    composition's, a classifier's), or strictyaml's schema or syntax error,
    which is not a ValueError and would otherwise end in a traceback. Its
    full location is ``pro spec``'s to give, and the line says so."""
    problem = getattr(exc, "problem", None)
    if problem:
        mark = getattr(exc, "problem_mark", None)
        where = ", line %d" % (mark.line + 1,) if mark is not None else ""
        text = "%s%s: %s" % (spec, where, problem)
    else:
        text = str(exc)
    return "%s (run %s %s)" % (text, command("spec"), spec)


def load_errors():
    """The exceptions :func:`load_error` turns into a line."""
    try:
        from strictyaml.ruamel.error import YAMLError
    except ImportError:                   # no front end: nothing raises it
        return (ValueError,)
    return (ValueError, YAMLError)


def parser_kwargs():
    """ArgumentParser options for every ``pro`` parser.

    No option is matched by a prefix (``--cel`` is not ``--cells``), so a
    script's abbreviation cannot change meaning when an option is added.
    3.14 also suggests the nearest command on a typo ("maybe you meant
    'run'?") and colours help and errors; older Pythons get neither.
    """
    kwargs = {"allow_abbrev": False}
    if sys.version_info >= (3, 14):
        kwargs.update(suggest_on_error=True, color=True)
    return kwargs


def _version():
    from proteus import __version__
    return __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="pro",
        description="Proteus: set up, mesh and run models.",
        **parser_kwargs())
    parser.add_argument("--version", action="version",
                        version="%(prog)s (proteus) " + _version())
    subparsers = parser.add_subparsers(
        dest="command", metavar="<command>", required=True)
    for module, _ in COMMANDS.values():
        importlib.import_module("." + module, __name__).add_parser(subparsers)
    for name, (summary, needs) in FRONTEND.items():
        absent = missing(*needs)
        note = " (needs ymf[symbolic])" if absent else " (not implemented yet)"
        sub = subparsers.add_parser(name, help=summary + note,
                                    description=summary + note,
                                    **parser_kwargs())
        sub.set_defaults(handler=_frontend_stub(name, absent))
    return parser


def _frontend_stub(name, absent):
    def handler(args):
        if absent:
            print("pro {0}: needs {1}, which this environment lacks.\n"
                  "Install the front end with: pip install 'ymf[symbolic]'"
                  .format(name, ", ".join(absent)), file=sys.stderr)
        else:
            print("pro {0}: not implemented yet.".format(name),
                  file=sys.stderr)
        return 2
    return handler


#: what turning a spec into a Proteus problem imports (ymf[symbolic])
SPEC_NEEDS = ("ymf.symbolic", "sympy", "strictyaml")


def _spec_command(argv):
    """``"run"`` or ``"msh"`` when ``pro run`` or ``pro msh`` (by either of
    its names) is given a YMF spec or archive rather than Python modules;
    else None. Those go to the YMF runner, whose options are its own."""
    names = {n: module for short, (module, long) in COMMANDS.items()
             if module in ("run", "msh") for n in (short, long)}
    if argv[:1] and argv[0] in names and any(
            a.endswith(".ymf") for a in argv[1:] if not a.startswith("-")):
        return names[argv[0]]
    return None


def run_spec(argv, mesh_only=False):
    """``pro run SPEC.ymf ...``: the YMF runner (:mod:`proteus.cli.ymfrun`),
    which has options of its own (``--cells``, ``--check``, ``--emit-pn``...).
    ``pro msh SPEC.ymf`` (``mesh_only``) makes the meshes it would need.
    """
    absent = missing(*SPEC_NEEDS)
    if absent:
        print("pro run: a .ymf spec needs ymf[symbolic] to become a Proteus\n"
              "problem, and this environment lacks {0}.\n"
              "Where the front end is installed, write the problem as Proteus\n"
              "files with\n"
              "    pro run SPEC.ymf --emit-pn\n"
              "and run the <stem>_<digest>_so.py it writes here with pro run,\n"
              "which needs no sympy. Or: pip install 'ymf[symbolic]'"
              .format(", ".join(absent)), file=sys.stderr)
        return 2
    # importing ymfrun pins the BLAS/OpenMP threads; nothing numeric is
    # loaded yet at this point
    from . import ymfrun
    return ymfrun.main(argv, prog=command("msh" if mesh_only else "run"),
                       mesh_only=mesh_only)


def main(argv=None):
    """Entry point for the ``pro`` console script; returns the exit status."""
    argv = sys.argv[1:] if argv is None else list(argv)
    # under mpiexec, an exception one rank raises and nothing catches aborts
    # the whole job instead of leaving the other ranks waiting in a collective
    from proteus import runtime
    runtime.abort_on_uncaught_exception("pro")
    spec = _spec_command(argv)
    if spec is not None:
        return run_spec(argv[1:], mesh_only=spec == "msh")
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
