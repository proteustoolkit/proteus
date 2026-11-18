"""A ymf sub-problem record -> the generator's JSON input (needs nothing but the record: plain data in, plain data out).

``split_problem(doc, discretization)`` returns, for each sub-problem, its ADR form and the words round 5 of ymf added
(``stabilization``, ``lagged_coefficients``, ``coefficient_roles``, boundary conditions with their ``kind``); the top level
states the mesh symbols (``bound``: ``h: element_size``). This module copies what generate.py reads and nothing else, and it
chooses nothing by the name of the sub-problem: the kernel is named after it, that is all.
"""
import json
import re

# what generate.py reads of a sub-problem record, key by key
_COPIED = ("adr", "reads", "stabilization", "coefficient_roles", "lagged_coefficients", "coefficients", "named", "regularized",
           "unknowns", "time_integration", "finite_element")
_CONDITION_KEYS = ("type", "variable", "region", "only", "kind", "penalty", "form", "component")


def kernel_input(split, sub, kernel=None):
    """The input of generate.py for ``sub`` (a record of ``split["subproblems"]``), or None if it has no ADR form."""
    if "adr" not in sub:
        return None
    text = json.dumps(sub["adr"], ensure_ascii=False)
    spec = {"kernel": kernel or sub["name"].upper(), "dim": split["dim"], "subproblem": sub["name"]}
    for key in _COPIED:
        if sub.get(key) is not None:
            spec[key] = sub[key]
    spec["mesh_symbols"] = [k for k, meaning in (split.get("bound") or {}).items()
                            if meaning == "element_size" and re.search(r"(?<![\w.])%s(?![\w])" % re.escape(k), text)]
    spec["boundary_conditions"] = [{k: bc[k] for k in _CONDITION_KEYS if k in bc} for bc in sub.get("boundary_conditions") or []]
    return spec
