"""coverage.py plugin for the Cython-compiled .py modules: Cython.Coverage plus one hook.

Why a subclass. With `source = ["proteus"]` every process, when it saves its data, adds each .py file of the package
that it did not execute as an empty entry ("0% covered"), with no file tracer. The three modules built with line
tracing (see setup.py, PROTEUS_CYTHON_LINETRACE) are executed, through Cython.Coverage, by the processes that import
them, and those processes record them with the plugin as tracer. Combining the two kinds of data file then fails:

    coverage.exceptions.DataError: Conflicting file tracer name for '.../proteus/WaveTools.py':
    'Cython.Coverage.Plugin' vs ''

(cekees/proteus#325, run 44417220906, in pytest-cov's combine on the xdist controller: INTERNALERROR after 4,603
passed). coverage asks every plugin for the files it is responsible for in each source directory
(`find_executable_files`) and touches them with the plugin's name; Cython's plugin has no such hook, so this one adds it.
Every process then records these three files with the same tracer, executed or not.

Used by the CI coverage leg and its coverage job through `plugins = ["cython_traced"]` in a copy of the coverage
configuration, with .github/coverage on PYTHONPATH; nothing else imports it.
"""
import os

from Cython.Coverage import Plugin as _CythonPlugin

# relative to the package directory, '/'-separated
TRACED_MODULES = ('BoundaryConditions.py', 'WaveTools.py')


class Plugin(_CythonPlugin):
    def find_executable_files(self, src_dir):
        for name in TRACED_MODULES:
            path = os.path.join(src_dir, *name.split('/'))
            if os.path.exists(path):
                yield path


def coverage_init(reg, options):
    plugin = Plugin()
    reg.add_configurer(plugin)
    reg.add_file_tracer(plugin)
