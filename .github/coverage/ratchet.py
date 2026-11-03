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
from this report, keeping every floor that is higher than today's value.
"""
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

TOLERANCE_PP = 0.5
TOLERANCE_STATEMENTS = 2


def module_name(filename):
    """``proteus/...`` for a file measured in a checkout or in site-packages."""
    filename = filename.replace(os.sep, '/')
    if 'site-packages/' in filename:
        filename = filename.rsplit('site-packages/', 1)[1]
    elif not filename.startswith('proteus/') and '/proteus/proteus/' in filename:
        filename = 'proteus/' + filename.rsplit('/proteus/proteus/', 1)[1]
    return filename


def measure(report):
    """{module: (statements, covered)} from a Cobertura report."""
    modules = {}
    for cls in ET.parse(report).iter('class'):
        lines = [int(line.get('hits')) for line in cls.iter('line')]
        name = module_name(cls.get('filename'))
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
    drops, rises, new = [], [], []
    for name, (statements, covered) in sorted(modules.items()):
        now = percent(statements, covered)
        if name not in floors:
            new.append((name, statements, now))
        elif now < floors[name] - tolerance(statements):
            drops.append((name, statements, floors[name], now))
        elif now >= floors[name] + 1.0:
            rises.append((name, statements, floors[name], now))
    gone = sorted(set(floors) - set(modules))
    return drops, rises, new, gone


def tables(modules, drops, rises, new, gone, partial, worst=40):
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
        today = math.floor(10.0 * percent(statements, covered)) / 10.0
        merged[name] = max(today, floors.get(name, 0.0))
    with open(path, 'w') as f:
        f.write('{\n')
        f.write(',\n'.join('  %s: %.1f' % (json.dumps(name), merged[name]) for name in sorted(merged)))
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
    drops, rises, new, gone = compare(modules, floors)
    sys.stdout.write(tables(modules, drops, rises, new, gone, args.partial))
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
