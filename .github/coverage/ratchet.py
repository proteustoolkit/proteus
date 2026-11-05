"""The coverage ratchet: per-module statement coverage against recorded floors.

    python ratchet.py coverage.xml floors.json [--partial] [--write-floors]

Reads a Cobertura report (`coverage xml`), prints the per-subpackage and
per-module tables as Markdown, and exits 1 if a module's statement coverage
fell below its floor in floors.json. A floor is not a target: it is the
coverage a module had when it was recorded, so code can only lose its tests on
purpose, by lowering the floor in the same pull request.

A module may sit a little under its floor before that counts as a drop: half a
percentage point, or two statements' worth in a small module, so that renaming
a covered line does not fail the run. A module new since the floors were
recorded has no floor and is reported, not checked; a module that is gone is
reported too. --partial (one of the measured suites is missing) prints
the tables and the drops but never fails. --write-floors rewrites floors.json
from this report: a module with no floor gets today's value, and a floor is
raised to today's value only when that is at least a point (RISE_PP) above it
-- the modules in the "at least a point above the floor" table -- so a floor
never chases the tenths a run's MPI timing can move. Every other floor stays.

A floor may instead be a string: the module is measured and listed with that
reason, but not checked (a module waiting for its first test, or one excluded
until a campaign decides its fate). --write-floors leaves those entries as
they are; turning one into a number is a reviewed edit.
"""
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

TOLERANCE_PP = 0.5
TOLERANCE_STATEMENTS = 2
RISE_PP = 1.0


def module_name(filename, source=''):
    """``proteus/...`` for a file measured in a checkout or in site-packages.

    A combined report names files relative to its <source> (the checkout's proteus/ directory), a single
    run's report by their absolute site-packages path; both end in .../proteus/<module path>, and the
    package has no directory of its own called proteus.
    """
    path = filename.replace(os.sep, '/')
    if source and not path.startswith('/'):
        path = source.replace(os.sep, '/').rstrip('/') + '/' + path
    if '/proteus/' in '/' + path:
        return 'proteus/' + ('/' + path).rsplit('/proteus/', 1)[1]
    return path


def measure(report):
    """{module: (statements, covered)} from a Cobertura report."""
    modules = {}
    root = ET.parse(report).getroot()
    sources = [s.text or '' for s in root.iter('source')]
    source = sources[0] if len(sources) == 1 else ''
    for cls in root.iter('class'):
        lines = [int(line.get('hits')) for line in cls.iter('line')]
        name = module_name(cls.get('filename'), source)
        statements, covered = modules.get(name, (0, 0))
        modules[name] = (statements + len(lines), covered + sum(1 for hits in lines if hits > 0))
    return modules


def percent(statements, covered):
    return 100.0 * covered / statements if statements else 100.0


def subpackage(module):
    parts = module.split('/')
    return '/'.join(parts[:2]) if len(parts) > 2 else 'proteus/ (top level)'


def tolerance(statements):
    return max(TOLERANCE_PP, 100.0 * TOLERANCE_STATEMENTS / statements) if statements else TOLERANCE_PP


def compare(modules, floors):
    drops, rises, new, unchecked = [], [], [], []
    for name, (statements, covered) in sorted(modules.items()):
        now = percent(statements, covered)
        if name not in floors:
            new.append((name, statements, now))
        elif isinstance(floors[name], str):
            unchecked.append((name, statements, now, floors[name]))
        elif now < floors[name] - tolerance(statements):
            drops.append((name, statements, floors[name], now))
        elif now >= floors[name] + RISE_PP:
            rises.append((name, statements, floors[name], now))
    gone = sorted(set(floors) - set(modules))
    return drops, rises, new, unchecked, gone


def tables(modules, drops, rises, new, unchecked, gone, partial, worst=40):
    out = []
    statements = sum(s for s, _ in modules.values())
    covered = sum(c for _, c in modules.values())
    out.append('## Coverage of the proteus package (Python statements)\n')
    out.append('**%d statements, %.1f%% covered** in %d modules%s.\n'
               % (statements, percent(statements, covered), len(modules),
                  ' (partial: a measured suite is missing, nothing is enforced)' if partial else ''))
    if drops:
        out.append('### Below the recorded floor\n')
        out.append('| module | statements | floor | now |\n|---|---:|---:|---:|')
        out += ['| `%s` | %d | %.1f%% | **%.1f%%** |' % d for d in drops]
        out.append('\nIf the drop is intended, lower the module\'s floor in `.github/coverage/floors.json` '
                   'in the same pull request.\n')
    by_package = {}
    for name, (s, c) in modules.items():
        total = by_package.setdefault(subpackage(name), [0, 0])
        total[0] += s
        total[1] += c
    out.append('### By subpackage\n')
    out.append('| package | statements | covered | missing |\n|---|---:|---:|---:|')
    for package, (s, c) in sorted(by_package.items(), key=lambda kv: -kv[1][0]):
        out.append('| `%s` | %d | %.1f%% | %d |' % (package, s, percent(s, c), s - c))
    out.append('\n### The %d modules with the most untested statements\n' % worst)
    out.append('| module | statements | covered | missing |\n|---|---:|---:|---:|')
    ranked = sorted(modules.items(), key=lambda kv: -(kv[1][0] - kv[1][1]))[:worst]
    out += ['| `%s` | %d | %.1f%% | %d |' % (name, s, percent(s, c), s - c) for name, (s, c) in ranked]
    if rises:
        out.append('\n### At least a point above the floor (raise it with `--write-floors`)\n')
        out.append('| module | statements | floor | now |\n|---|---:|---:|---:|')
        out += ['| `%s` | %d | %.1f%% | %.1f%% |' % r for r in rises]
    if unchecked:
        out.append('\n### Measured, not checked\n')
        out.append('| module | statements | covered | why |\n|---|---:|---:|---|')
        out += ['| `%s` | %d | %.1f%% | %s |' % u for u in unchecked]
    if new:
        out.append('\n### No floor yet\n')
        out += ['- `%s`: %d statements, %.1f%%' % n for n in new]
    if gone:
        out.append('\n### Floors for modules no longer measured\n')
        out += ['- `%s`' % name for name in gone]
    return '\n'.join(out) + '\n'


def write_floors(path, modules, floors):
    merged = dict(floors)
    for name, (statements, covered) in modules.items():
        if isinstance(floors.get(name), str):
            continue
        now = percent(statements, covered)
        if name not in floors or now >= floors[name] + RISE_PP:
            merged[name] = math.floor(10.0 * now) / 10.0
    with open(path, 'w') as f:
        f.write('{\n')
        f.write(',\n'.join('  %s: %s' % (json.dumps(name), json.dumps(merged[name]) if isinstance(merged[name], str)
                                          else '%.1f' % merged[name]) for name in sorted(merged)))
        f.write('\n}\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('report')
    parser.add_argument('floors')
    parser.add_argument('--partial', action='store_true')
    parser.add_argument('--write-floors', action='store_true')
    args = parser.parse_args(argv)
    modules = measure(args.report)
    floors = {}
    if os.path.exists(args.floors):
        with open(args.floors) as f:
            floors = json.load(f)
    drops, rises, new, unchecked, gone = compare(modules, floors)
    sys.stdout.write(tables(modules, drops, rises, new, unchecked, gone, args.partial))
    # a report whose module names do not match the floors' checks nothing, and would pass
    if floors and not args.write_floors and len(set(floors) & set(modules)) < len(floors) // 2:
        sys.stderr.write('only %d of %d floors name a module in this report (first module: %s): '
                         'the report and the floors disagree about module names\n'
                         % (len(set(floors) & set(modules)), len(floors), min(modules) if modules else 'none'))
        return 1
    if args.write_floors:
        write_floors(args.floors, modules, floors)
        return 0
    if drops:
        sys.stderr.write('%d module(s) below their coverage floor: %s\n'
                         % (len(drops), ', '.join(d[0] for d in drops)))
        return 0 if args.partial else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
