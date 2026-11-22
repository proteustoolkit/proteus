"""No argsDict entry is handed a tuple (or another non-array literal).

``argsDict["k"] = arr,`` -- note the trailing comma -- stores the 1-tuple
``(arr,)``, not ``arr``. ArgumentsDict.__setitem__ only has overloads for
int/double arrays and int/double scalars, so proteus::pyarray rejects the
tuple with a TypeError. The mistake only surfaces when that model's residual
or Jacobian is first evaluated, and several of the models involved have no
test that does so. This scan catches it in every module without running any
of them.
"""
import ast
import os
import textwrap

import pytest

import proteus

PROTEUS_ROOT = os.path.dirname(os.path.abspath(proteus.__file__))

#: Node types that are never a valid ArgumentsDict value.
_BAD_NODES = (ast.Tuple, ast.List, ast.Set, ast.Dict,
              ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _arguments_dict_names(tree):
    """Names bound to an ArgumentsDict() in this module, plus 'argsDict'."""
    names = {"argsDict"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            fname = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if fname == "ArgumentsDict":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
                    elif isinstance(target, ast.Attribute):
                        names.add(target.attr)
    return names


def _bad_value(value):
    if isinstance(value, _BAD_NODES):
        return True
    # None, strings, bytes: no overload takes them.
    return isinstance(value, ast.Constant) and not isinstance(value.value, (int, float))


def find_bad_assignments(source, filename="<string>"):
    """(line, text) for each `<argsDict>[...] = <non-array literal>` in source."""
    tree = ast.parse(source, filename)
    names = _arguments_dict_names(tree)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not _bad_value(node.value):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            base = target.value
            bname = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
            if bname in names:
                found.append((node.lineno, ast.unparse(node)))
    return found


def _proteus_sources():
    for dirpath, dirnames, filenames in os.walk(PROTEUS_ROOT):
        dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def test_scanner_catches_trailing_comma():
    source = textwrap.dedent("""
        from proteus import cArgumentsDict
        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["ok_array"] = a
        argsDict["ok_scalar"] = 1.0
        argsDict["bad_tuple"] = a,
        argsDict["bad_commented"] = a,  # trailing comment
        argsDict["bad_none"] = None
        self.args = cArgumentsDict.ArgumentsDict()
        self.args["bad_attr"] = (a,
                                 )
    """)
    lines = [line for line, _ in find_bad_assignments(source)]
    assert lines == [6, 7, 8, 10]


def test_no_tuple_values_assigned_into_argsdict():
    offenders = []
    for path in _proteus_sources():
        with open(path, encoding="utf-8") as f:
            source = f.read()
        try:
            found = find_bad_assignments(source, path)
        except SyntaxError as e:
            pytest.fail("could not parse %s: %s" % (path, e))
        rel = os.path.relpath(path, PROTEUS_ROOT)
        offenders += ["proteus/%s:%d: %s" % (rel, line, text) for line, text in found]
    assert not offenders, (
        "ArgumentsDict values must be ndarrays or int/float scalars "
        "(check for a stray trailing comma):\n  " + "\n  ".join(offenders))
