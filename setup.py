import sys, os
import platform
import subprocess
import setuptools
from distutils import sysconfig
cfg_vars = sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = cfg_vars[key].replace("-Wstrict-prototypes", "")
        cfg_vars[key] = cfg_vars[key].replace("-Wall", "-w")

# Some conda-forge Python builds (confirmed: the environment-dev.yml-pinned
# python=3.12.5 osx-64 build) bake '-Wl,-rpath,<dir>' into their own
# sysconfig LDSHARED/LDCXXSHARED *twice* already -- i.e. every C/C++
# extension this interpreter links inherits a duplicate rpath before
# anything in this setup.py adds a single flag of its own. On current
# macOS, dyld refuses to even open a .so with a literal duplicate LC_RPATH
# load command ("dlopen(...): tried: ... (duplicate LC_RPATH ...)"), so
# every extension built with such an interpreter fails to import. Separately,
# the conda-forge `compilers` package's own activation script sets $LDFLAGS
# to the *same* '-Wl,-rpath,<dir>' -- and setuptools appends $LDFLAGS to
# every link command in addition to LDSHARED/LDCXXSHARED, so even after
# collapsing LDSHARED's own internal duplicate down to one copy, that one
# copy plus LDFLAGS's copy still add up to two identical LC_RPATH entries.
# dyld's duplicate check is on the *final linked binary*, not the source of
# each flag, so both angles have to be deduped together: collapse repeats
# within LDSHARED/LDCXXSHARED themselves, then drop any rpath token from
# them that's already present in $LDFLAGS (leaving LDFLAGS's copy as the
# single source of truth). Patched here, in the same place this file
# already patches other sysconfig quirks above, rather than in every
# individual Extension(). Python 3.12 removed distutils from the stdlib, so
# setuptools' build_ext actually customizes the compiler via the *stdlib*
# `sysconfig` module's own config cache, not `distutils.sysconfig`'s (they
# are two distinct dicts on this setuptools/Python combination, confirmed
# by identity check) -- patch both so this holds regardless of which one
# the installed setuptools version ends up reading from.
def _dedup_rpath_tokens(flags):
    ldflags_env = os.environ.get('LDFLAGS', '')
    seen = set()
    out = []
    for tok in flags.split():
        if tok.startswith('-Wl,-rpath,'):
            if tok in seen or tok in ldflags_env:
                continue
            seen.add(tok)
        out.append(tok)
    return ' '.join(out)
import sysconfig as _stdlib_sysconfig
for _cfg_vars in (cfg_vars, _stdlib_sysconfig.get_config_vars()):
    for key in ('LDSHARED', 'LDCXXSHARED'):
        if isinstance(_cfg_vars.get(key), str):
            _cfg_vars[key] = _dedup_rpath_tokens(_cfg_vars[key])

# '-partition=none' is a *mangled* '-flto-partition=none'. conda-forge's
# python recipe strips its own build-time LTO flags back out of the installed
# _sysconfigdata, and a substring removal of '-flto' turns
# '-flto-partition=none' into '-partition=none' -- which no gcc/g++ accepts
# ("unrecognized command-line option '-partition=none'; did you mean
# '-flto-partition=none'?"). The token sits in the interpreter's own
# sysconfig CFLAGS, so every extension this interpreter builds inherits it
# and the whole wheel build dies at the first compile, before any of
# proteus's own sources are even parsed.
#
# This is per python *build*, not per version, which is why the same source
# builds locally and fails in CI: linux-64 python 3.13.15 from
# python-split_1786366211553 has the LTO flags cleanly removed (blank gaps in
# CFLAGS where they were), while 3.13.15 from python-split_1788361500653 --
# the build the sdist.yml "linux-64 / py3.13 (locked)" leg resolves, since
# the environment files pin the version but not the build string -- carries
# the mangled token. Dropping it is safe rather than a workaround for a flag
# we want: the '-flto' it was an option to is already gone, and the
# '-fuse-linker-plugin'/'-ffat-lto-objects' left beside it are valid options
# that are inert without it.
#
# Patched over *both* config dicts for the same reason the rpath dedup above
# is: depending on the setuptools version the compiler is customized from the
# stdlib `sysconfig` cache rather than `distutils.sysconfig`'s, and pip's
# build isolation installs its own newer setuptools regardless of what the
# conda environment pins -- the failing CI compile lines still carry the
# leading '-Wall' that the distutils-only rewrite at the top of this file
# replaces with '-w', which is direct evidence that under an isolated build
# it is the stdlib dict, not that one, that reaches the compiler.
for _cfg_vars in (cfg_vars, _stdlib_sysconfig.get_config_vars()):
    for key, value in _cfg_vars.items():
        if isinstance(value, str) and '-partition=none' in value:
            _cfg_vars[key] = value.replace('-partition=none', '')

# setuptools appends the *environment's* CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS to
# each compile and link line on top of whatever sysconfig supplies, so scrub
# those too in case an activation script re-exports the same broken flags.
for _env_var in ('CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS'):
    if '-partition=none' in os.environ.get(_env_var, ''):
        os.environ[_env_var] = os.environ[_env_var].replace('-partition=none', '')

from distutils.core import setup
from Cython.Build import cythonize
from Cython.Distutils.extension import Extension
from Cython.Distutils import build_ext

class custom_build_ext(build_ext):
    def build_extension(self, ext):
        # Four sources are listed by two extensions each: mesh.cpp and
        # meshio.cpp (cmeshTools, cpartitioning), postprocessing.c
        # (cfemIntegrals, cpostprocessing) and
        # SubsurfaceTransportCoefficients.cpp (cSubsurfaceTransportCoefficients,
        # cTwophaseDarcyCoefficients). setuptools derives each object path from
        # the source path under a single self.build_temp, so both members of a
        # pair compile to the *same* .o -- and since build_extensions sets
        # self.parallel, setuptools builds extensions concurrently in a
        # ThreadPoolExecutor over os.cpu_count() threads. Two threads then write
        # one object file at once.
        #
        # This fails two ways, both silent. The linker can read a half-written
        # object, yielding a .so missing every symbol from that source -- the
        # build succeeds and the failure surfaces only at import ("undefined
        # symbol: regularMeshNodes2D"). Or both compiles complete and whichever
        # lands last is linked into both extensions, which is worse here because
        # the pairs do not use the same flags: cpartitioning adds -std=c++20
        # where cmeshTools does not, and cfemIntegrals defines PROTEUS_SUPERLU_H
        # where cpostprocessing does not.
        #
        # Give every extension its own object directory. Mutating
        # self.build_temp in place would itself race, so hand build_extension a
        # shallow copy of this command carrying a private build_temp; the
        # compiler is passed output_dir explicitly, so the copy is enough.
        # imported here, not at module scope: the only `os` in this file's
        # namespace arrives incidentally via `from proteus.config import *`.
        import copy as _copy, os as _osmod
        private = _copy.copy(self)
        private.build_temp = _osmod.path.join(self.build_temp, '_ext',
                                              ext.name.replace('.', '_'))
        _osmod.makedirs(private.build_temp, exist_ok=True)
        return build_ext.build_extension(private, ext)

    def build_extensions(self):
        self.parallel=True
        # OpenMPI's/MPICH's mpi.h transparently pulls in its legacy C++
        # bindings (the `MPI::` namespace, removed from the MPI standard
        # since 3.0) the moment mpi.h is included from a C++ translation
        # unit. Several extensions (cmeshTools.pyx/cpartitioning.pyx/
        # RANS2P.cpp and others) pull mpi.h in this way via generated
        # Cython code that `cimport`s mpi4py.MPI, which #includes mpi.h
        # before any of proteus's own headers get a chance to guard
        # against this -- and per-extension `extra_compile_args` wiring is
        # inconsistent across this file's ~100 Extension() entries (some
        # include PROTEUS_EXTRA_COMPILE_ARGS, several of the mprans.c*
        # extensions don't), so a couple of these silently slipped through
        # a first attempt at fixing this per-extension. Some of the
        # bindings' methods aren't fully header-inline, so merely including
        # the header -- without proteus's own source ever writing `MPI::`
        # anywhere -- silently requires linking a separate libmpi_cxx that
        # these extensions never ask for. Confirmed via a real build
        # against Ubuntu's system OpenMPI package: cpartitioning and
        # mprans.cRANS2P both failed to import with "undefined symbol:
        # _ZN3MPI8Datatype4FreeEv". Setting this here, on the compiler
        # instance itself right before any extension is actually compiled
        # (same idea as the -fPIE handling below, applied globally rather
        # than per-Extension), guarantees every extension gets it
        # regardless of that extension's own extra_compile_args. Getting
        # the right attribute took two tries: patching compiler_so alone
        # only reached .c files, and compiler_cxx turned out to be a red
        # herring too -- on this setuptools/Python combination, UnixCCompiler
        # ._compile() actually dispatches C++ sources through
        # compiler_so_cxx (confirmed by reading _compile()'s source
        # directly: it builds the g++ command from compiler_so_cxx, not
        # compiler_cxx, which distutils only ever uses as a bare
        # [executable-name] placeholder). Patch every list that could
        # plausibly be consulted so this isn't sensitive to yet another
        # distutils-vs-setuptools-vendored-copy naming difference.
        for macro in ('-DOMPI_SKIP_MPICXX', '-DMPICH_SKIP_MPICXX'):
            for attr in ('compiler_so', 'compiler_so_cxx', 'compiler_cxx', 'compiler'):
                lst = getattr(self.compiler, attr, None)
                if isinstance(lst, list) and macro not in lst:
                    lst.append(macro)
        try:
            self.compiler.linker_so.remove('-Wl,-pie')
            self.compiler.compiler_so.remove('-fPIE')
            self.compiler.linker_so.remove('-fPIE')
            self.compiler.compiler.remove('-fPIE')
        except:
            pass
        build_ext.build_extensions(self)
        
import numpy
## \file setup.py setup.py
#  \brief The python script for building proteus
#
#  Set the DISTUTILS_DEBUG environment variable to print detailed information while setup.py is running.
#

# insert (not append): an already-installed proteus package earlier on
# sys.path (e.g. via PYTHONPATH pointing at a previous --target install)
# would otherwise shadow *this* checkout's own proteus/config, silently
# building against a stale config.py instead of the one actually being
# edited/rebuilt.
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from proteus import config
from proteus.config import *
###to turn on debugging in c++
##\todo Finishing cleaning up setup.py/setup.cfg, config.py...

PROTEUS_PETSC_EXTRA_LINK_ARGS = getattr(config, 'PROTEUS_PETSC_EXTRA_LINK_ARGS', [])
PROTEUS_PETSC_EXTRA_COMPILE_ARGS = getattr(config, 'PROTEUS_PETSC_EXTRA_COMPILE_ARGS', [])
PROTEUS_CHRONO_CXX_FLAGS = getattr(config, 'PROTEUS_CHRONO_CXX_FLAGS', [])

proteus_install_path = os.path.join(sysconfig.get_python_lib(), 'proteus')

# handle non-system installations
for arg in sys.argv:
    if arg.startswith('--root'):
        proteus_install_path = proteus_install_path.partition(sys.prefix + '/')[-1]
        break
    if arg.startswith('--prefix'):
        proteus_install_path = proteus_install_path.partition(sys.prefix + '/')[-1]
        break

def get_pybind_include_dirs():
    """Include path for the pybind11 extensions.

    These need pybind11, numpy and proteus's own headers (`proteus/pyarray.h`
    in particular, via the `proteus` entry).
    """
    return [str(get_pybind_include()),
            str(get_pybind_include(user=True)),
            str(get_numpy_include()),
            os.path.join(prefix, 'include'),
            os.path.join(sys.prefix, 'include'),
            os.path.join(sys.prefix, 'Library', 'include'),
            'proteus']

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


class get_numpy_include(object):
    """Helper class to determine the numpy include path

    The purpose of this class is to postpone importing numpy
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self):
        pass

    def __str__(self):
        import numpy as np
        return np.get_include()

# -mavx is x86-only; unconditionally requesting it fails outright on arm64
# ("unsupported option '-mavx' for target ...") rather than just being a
# missed optimization. But not every non-arm64 host actually supports AVX
# either (older Xeons/Atoms, and some virtualized/cloud CPU profiles that
# expose a conservative feature mask, e.g. QEMU's default "qemu64" model) --
# requesting it there doesn't fail the build, it silently bakes an illegal
# instruction into mprans.MeshSmoothing/mprans.cMoveMeshMonitor that SIGILLs
# the first time either module is imported. So probe the actual build host's
# CPU capability instead of just checking platform.machine(). If detection
# itself isn't possible (unknown platform, sandboxed build host, etc.),
# default to no AVX -- a missed optimization is far cheaper than a crash.
def _host_supports_avx():
    if platform.machine() not in ('x86_64', 'AMD64', 'i386', 'i686'):
        return False
    if platform.system() == 'Linux':
        try:
            with open('/proc/cpuinfo') as f:
                return any(
                    'avx' in line.split(':', 1)[1].split()
                    for line in f if line.startswith('flags')
                )
        except (OSError, IndexError):
            return False
    if platform.system() == 'Darwin':
        try:
            out = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.features'],
                stderr=subprocess.DEVNULL).decode()
            return 'AVX' in out.split()
        except (OSError, subprocess.CalledProcessError):
            return False
    # e.g. Windows -- no cheap portable probe here; be conservative.
    return False

PROTEUS_AVX_FLAGS = ['-mavx'] if _host_supports_avx() else []


def _probe_pumi_partitioners(scorec_lib_dir):
    """Which graph partitioners the PUMI (SCOREC core) install was built with.

    SCOREC's installed CMake target files carry PUMI_HAS_ZOLTAN on apf_zoltan
    and PUMI_HAS_METIS on apf_metis when core was built with them (the
    libraries themselves always exist, as an empty stub when not enabled).
    Returns {'zoltan': 1|0|-1, 'metis': 1|0|-1}; -1 means the install has no
    CMake target files to read, so the answer is unknown.
    """
    found = {}
    for name, tgt in (('zoltan', 'apf_zoltan'), ('metis', 'apf_metis')):
        found[name] = -1
        cmake_dir = os.path.join(scorec_lib_dir or '', 'cmake', 'SCOREC')
        files = [os.path.join(cmake_dir, f) for f in (
            os.listdir(cmake_dir) if os.path.isdir(cmake_dir) else [])
            if f.startswith(tgt + '-target')]
        if files:
            text = ''.join(open(f).read() for f in files)
            found[name] = int(('PUMI_HAS_' + name.upper()) in text)
    return found

PROTEUS_PUMI_PARTITIONERS = _probe_pumi_partitioners(PROTEUS_SCOREC_LIB_DIR)
print("proteus build: PUMI partitioners (%s): %s" % (
    PROTEUS_SCOREC_LIB_DIR,
    ", ".join("%s=%s" % (k, {1: 'yes', 0: 'no', -1: 'unknown'}[v])
              for k, v in sorted(PROTEUS_PUMI_PARTITIONERS.items()))))

# What is expected to vectorize (C30, ON_NODE_PERFORMANCE_BASELINE.md §6;
# decided 2026-10-24). The extensions compile at sysconfig's -O2 with no
# -march, plus PROTEUS_OPT from the environment for a site's choice. The
# legacy element kernels in femIntegrals.c are scaffolding on the way to the
# generated kernels (campaign C19) and are not expected to vectorize: their
# flat index arithmetic and in-memory accumulators keep the compiler out at
# any flag level (clang -Rpass: 150 of 369 loops at -O2, none of the passes
# that dominate assembly; -O3 -mcpu=native changes the residual by 10% and the
# Jacobian by nothing). The generated kernels are expected to, over the
# quadrature points of an element, which needs local accumulators, restrict
# (or references into fixed-size element arrays) and compile-time element
# shape. Keep -O2 the default; do not tune the legacy passes for flags.

EXTENSIONS_TO_BUILD = [
    Extension("MeshAdaptPUMI.MeshAdapt",
              sources = ['proteus/MeshAdaptPUMI/MeshAdapt.pyx', 'proteus/MeshAdaptPUMI/cMeshAdaptPUMI.cpp',
                         'proteus/MeshAdaptPUMI/MeshConverter.cpp', 'proteus/MeshAdaptPUMI/ParallelMeshConverter.cpp',
                         'proteus/MeshAdaptPUMI/MeshFields.cpp', 'proteus/MeshAdaptPUMI/SizeField.cpp',
                         'proteus/MeshAdaptPUMI/DumpMesh.cpp',
                         'proteus/MeshAdaptPUMI/ErrorResidualMethod.cpp','proteus/MeshAdaptPUMI/VMS.cpp','proteus/MeshAdaptPUMI/createAnalyticGeometry.cpp'],
              depends=["proteus/partitioning.h",
                       "proteus/partitioning.cpp",
                       "proteus/cpartitioning.pyx",
                       "proteus/cmeshTools.pxd",
                       "proteus/mesh.h",
                       "proteus/index_type.h",
                       'proteus/mesh.cpp',
                       'proteus/meshio.cpp'],
              define_macros=[('PROTEUS_SUPERLU_H',PROTEUS_SUPERLU_H),
                             ('PROTEUS_PUMI_HAS_ZOLTAN',PROTEUS_PUMI_PARTITIONERS['zoltan']),
                             ('PROTEUS_PUMI_HAS_METIS',PROTEUS_PUMI_PARTITIONERS['metis']),
                             ('PROTEUS_PUMI_PROBED_DIR','"%s"' % PROTEUS_SCOREC_LIB_DIR)],
              language='c++',
              include_dirs=[numpy.get_include(),'include',
                            'proteus','proteus/MeshAdaptPUMI']+
              PROTEUS_SCOREC_INCLUDE_DIRS,
              library_dirs=PROTEUS_SCOREC_LIB_DIRS,
              libraries=PROTEUS_SCOREC_LIBS,
              extra_compile_args=['-std=c++20']+PROTEUS_SCOREC_EXTRA_COMPILE_ARGS+PROTEUS_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_SCOREC_EXTRA_LINK_ARGS+PROTEUS_EXTRA_LINK_ARGS),
    Extension(
        'mprans.cArgumentsDict',
        sources = ['proteus/mprans/ArgumentsDict.cpp'],
        depends=['proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h'],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cPres',
        sources = ['proteus/mprans/Pres.cpp'],
        depends=['proteus/mprans/Pres.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cPresInit',
        sources = ['proteus/mprans/PresInit.cpp'],
        depends=['proteus/mprans/PresInit.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cPresInc',
        sources = ['proteus/mprans/PresInc.cpp'],
        depends = ['proteus/mprans/PresInc.h', 'proteus/mprans/PresInc.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension('mprans.cAddedMass',
              sources = ['proteus/mprans/AddedMass.cpp'],
              depends=['proteus/mprans/AddedMass.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
              language='c++',
              include_dirs=get_pybind_include_dirs(),
              extra_compile_args=PROTEUS_OPT+['-std=c++20']),
    Extension('mprans.SedClosure',
              sources = ['proteus/mprans/SedClosure.cpp'],
              depends = ['proteus/mprans/SedClosure.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
              language='c++',
              include_dirs=get_pybind_include_dirs(),
              extra_compile_args=PROTEUS_OPT+['-std=c++20']),
    Extension('mprans.cVOF3P',
              sources = ['proteus/mprans/VOF3P.cpp'],
              depends = ['proteus/mprans/VOF3P.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
              language='c++',
              include_dirs=get_pybind_include_dirs(),
              extra_compile_args=PROTEUS_OPT+['-std=c++20']),
    Extension(
        'mprans.cVOS3P',
        sources = ['proteus/mprans/VOS3P.cpp'],
        depends = ['proteus/mprans/VOS3P.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension('mprans.cNCLS3P',
              sources=['proteus/mprans/NCLS3P.cpp'],
              depends=['proteus/mprans/NCLS3P.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h' , 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
              language='c++',
              include_dirs=get_pybind_include_dirs(),
              extra_compile_args=PROTEUS_OPT+['-std=c++20']),
    Extension('mprans.cMCorr3P',
              sources=['proteus/mprans/MCorr3P.cpp'],
              depends=['proteus/mprans/MCorr3P.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
              language='c++',
              include_dirs=get_pybind_include_dirs(),
              extra_compile_args=PROTEUS_OPT+['-std=c++20'],
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS,
              define_macros=[('PROTEUS_LAPACK_H',
                              PROTEUS_LAPACK_H),
                             ('PROTEUS_LAPACK_INTEGER',
                              PROTEUS_LAPACK_INTEGER),
                             ('PROTEUS_BLAS_H',
                              PROTEUS_BLAS_H)],
              library_dirs=[PROTEUS_LAPACK_LIB_DIR,
                            PROTEUS_BLAS_LIB_DIR],
              libraries=['m',PROTEUS_LAPACK_LIB,
                         PROTEUS_BLAS_LIB],
              ),
    Extension(
        'mprans.cRANS3PSed',
        sources=['proteus/mprans/RANS3PSed.cpp'],
        depends=['proteus/mprans/RANS3PSed.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cRANS3PSed2D',
        sources=['proteus/mprans/RANS3PSed2D.cpp'],
        depends=['proteus/mprans/RANS3PSed2D.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'richards.cRichards',
        sources=['proteus/richards/cRichards.cpp'],
        depends=['proteus/richards/Richards.h',  'proteus/pskRelations.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h' ,'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
        include_dirs=get_pybind_include_dirs(),
        language='c++',
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
    ),
    Extension(
        # proteus/richards/ADR.py does `from .cADR import *`. This entry was
        # dropped in merge 9e2aca55 (2026-08-12); see erdc/proteus#1285.
        'richards.cADR',
        sources=['proteus/richards/cADR.cpp'],
        depends=['proteus/richards/ADR.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
        include_dirs=get_pybind_include_dirs(),
        language='c++',
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
    ),

    Extension(
        'm_comp_co2.cm_comp_co2',
        sources=['proteus/m_comp_co2/cm_comp_co2.cpp'],
        depends=['proteus/m_comp_co2/m_comp_co2.h', 'proteus/pskRelations.h', 'proteus/m_comp_co2/co2_brine_flash.h', 'proteus/m_comp_co2/co2_brine_eos.h', 'proteus/m_comp_co2/jet2.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h' ,'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
        include_dirs=get_pybind_include_dirs(),
        language='c++',
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
    ),

    Extension(
        'elastoplastic.cElastoPlastic',
        sources=['proteus/elastoplastic/cElastoPlastic.cpp'],
        define_macros=[('PROTEUS_LAPACK_H',
                        PROTEUS_LAPACK_H),
                       ('PROTEUS_LAPACK_INTEGER',
                        PROTEUS_LAPACK_INTEGER),
                       ('PROTEUS_BLAS_H',
                        PROTEUS_BLAS_H)],
        depends=['proteus/elastoplastic/ElastoPlastic.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h'],
        include_dirs=get_pybind_include_dirs(),
        language='c++',
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        library_dirs=[PROTEUS_LAPACK_LIB_DIR,
                      PROTEUS_BLAS_LIB_DIR],
        libraries=['m',PROTEUS_LAPACK_LIB,
                   PROTEUS_BLAS_LIB],
        extra_link_args=PROTEUS_EXTRA_LINK_ARGS
    ),
    Extension(
        'mprans.cRANS3PF',
        sources=['proteus/mprans/RANS3PF.cpp'],
        depends=['proteus/mprans/RANS3PF.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h',
                 'proteus/equivalent_polynomials.h',
                 'proteus/equivalent_polynomials_utils.h',
                 'proteus/equivalent_polynomials_coefficients.h',
                 'proteus/equivalent_polynomials_coefficients_quad.h'],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cRANS3PF2D',
        sources=['proteus/mprans/RANS3PF2D.cpp'],
        depends=['proteus/mprans/RANS3PF2D.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h',
                 'proteus/equivalent_polynomials.h',
                 'proteus/equivalent_polynomials_utils.h',
                 'proteus/equivalent_polynomials_coefficients.h',
                 'proteus/equivalent_polynomials_coefficients_quad.h'],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension("Isosurface",
              sources=['proteus/Isosurface.pyx'],
              language='c',
              extra_compile_args=PROTEUS_OPT,
              include_dirs=[numpy.get_include(),'proteus'],
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS),
    Extension("BoundaryConditions",
              sources=['proteus/BoundaryConditions.py'],
              depends=['proteus/BoundaryConditions.pxd'],
              language='c++',
              extra_compile_args=PROTEUS_OPT+['-std=c++14'],
              include_dirs=[numpy.get_include(),'proteus']),
    Extension("mprans.BoundaryConditions",
              sources=['proteus/mprans/BoundaryConditions.py'],
              depends=['proteus/mprans/BoundaryConditions.pxd'],
              language='c++',
              extra_compile_args=PROTEUS_OPT+['-std=c++14'],
              include_dirs=[numpy.get_include(),'proteus']),
    Extension("mprans.MeshSmoothing",
              sources=['proteus/mprans/MeshSmoothing.pyx'],
              language='c++',
              include_dirs=[numpy.get_include(),'proteus',PROTEUS_INCLUDE_DIR],
              libraries=['stdc++','m'],
              extra_compile_args=["-std=c++20"]+PROTEUS_AVX_FLAGS+PROTEUS_OPT),
    Extension("mprans.cMoveMeshMonitor",
              sources=['proteus/mprans/cMoveMeshMonitor.pyx'],
              language='c++',
              include_dirs=[numpy.get_include(),'proteus',PROTEUS_INCLUDE_DIR],
              libraries=['stdc++','m'],
              extra_compile_args=["-std=c++20"]+PROTEUS_AVX_FLAGS+PROTEUS_OPT),
    Extension("mbd.CouplingFSI",
              sources=['proteus/mbd/CouplingFSI.pyx',
                       'proteus/mbd/ChVariablesBodyAddedMass.cpp',
                       'proteus/mbd/ChBodyAddedMass.cpp'],
              depends=['proteus/mbd/CouplingFSI.pxd',
                       'proteus/mbd/ChronoHeaders.pxd',
                       'proteus/mbd/ProtChBody.h',
                       'proteus/mbd/ProtChMoorings.h'],
              language='c++',
              include_dirs=[numpy.get_include(),
                            'proteus',
                            'proteus/mbd',
                            PROTEUS_INCLUDE_DIR,
                            PROTEUS_INCLUDE_DIR+'/eigen3',
                            PROTEUS_CHRONO_INCLUDE_DIR,
                            PROTEUS_CHRONO_INCLUDE_DIR+'/chrono',
                            PROTEUS_CHRONO_INCLUDE_DIR+'/chrono/collision/bullet',],
              library_dirs=[PROTEUS_CHRONO_LIB_DIR],
              libraries=['Chrono_core',
                         'stdc++',
                         'm'],
              extra_compile_args=["-std=c++20"]+PROTEUS_CHRONO_CXX_FLAGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS),
    Extension("WaveTools",
              sources=['proteus/WaveTools.py'],
              depends=['proteus/WaveTools.h'],
              language='c++',
              extra_compile_args=PROTEUS_OPT,
              include_dirs=[numpy.get_include(),'proteus']),
    Extension("fenton.Fenton",
              sources=['proteus/fenton/Fenton.pyx',
                       'proteus/fenton/Solve.cpp',
                       'proteus/fenton/Dpythag.cpp',
                       'proteus/fenton/Dsvbksb.cpp',
                       'proteus/fenton/Dsvdcmp.cpp',
                       'proteus/fenton/Inout.cpp',
                       'proteus/fenton/Subroutines.cpp',
                       'proteus/fenton/Util.cpp',],
              language='c++',
              include_dirs=[numpy.get_include(),
                            'proteus',
                            PROTEUS_INCLUDE_DIR,],
              libraries=['stdc++','m'],
              extra_compile_args=["-std=c++11"]+PROTEUS_OPT),
    Extension(
        'cADR',
        sources=['proteus/ADR.cpp'],
        depends=['proteus/ADR.h', 'proteus/mprans/ArgumentsDict.h', 'proteus/pyarray.h', 'proteus/ModelFactory.h', 'proteus/CompKernel.h',
                 'proteus/equivalent_polynomials.h',
                 'proteus/equivalent_polynomials_utils.h',
                 'proteus/equivalent_polynomials_coefficients.h',
                 'proteus/equivalent_polynomials_coefficients_quad.h'],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'
    ),
    Extension("subsurfaceTransportFunctions",
              sources=['proteus/subsurfaceTransportFunctions.pyx'],
              include_dirs=[numpy.get_include(),'proteus'],
              extra_compile_args=PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS),
    Extension("equivalent_polynomials",
              sources=['proteus/equivalent_polynomials.pyx'],
              depends=['proteus/equivalent_polynomials.pxd',
                       'proteus/equivalent_polynomials.h',
                       'proteus/equivalent_polynomials_utils.h',
                       'proteus/equivalent_polynomials_coefficients.h',
                       'proteus/equivalent_polynomials_coefficients_quad.h'],
              language='c++',
              extra_compile_args=PROTEUS_OPT,
              include_dirs=[numpy.get_include(),'proteus'],),
    Extension('cfemIntegrals',
              sources=['proteus/cfemIntegrals.pyx',
                       'proteus/femIntegrals.c',
                       'proteus/postprocessing.c'],
              depends=['proteus/femIntegrals.h'],
              define_macros=[('PROTEUS_SUPERLU_H',PROTEUS_SUPERLU_H),
                             ('PROTEUS_LAPACK_H',PROTEUS_LAPACK_H),
                             ('PROTEUS_LAPACK_INTEGER',PROTEUS_LAPACK_INTEGER),
                             ('PROTEUS_BLAS_H',PROTEUS_BLAS_H)],
              include_dirs=[numpy.get_include(),'proteus',
                            PROTEUS_SUPERLU_INCLUDE_DIR,
                            PROTEUS_LAPACK_INCLUDE_DIR,
                            PROTEUS_BLAS_INCLUDE_DIR],
              library_dirs=[PROTEUS_LAPACK_LIB_DIR,
                            PROTEUS_BLAS_LIB_DIR],
              libraries=['m',PROTEUS_LAPACK_LIB,PROTEUS_BLAS_LIB],
              extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS),
    Extension("csparsity",
              sources=['proteus/csparsity.pyx', 'proteus/sparsity.cpp'],
              depends=['proteus/sparsity.h'],
              language='c++',
              extra_compile_args=PROTEUS_OPT,
              include_dirs=[numpy.get_include(),'proteus'],),
    Extension("cmeshTools",
              sources=['proteus/cmeshTools.pyx', 'proteus/mesh.cpp', 'proteus/meshio.cpp'],
              language='c++',
              define_macros=[('PROTEUS_SUPERLU_H',PROTEUS_SUPERLU_H),
                             ('CMRVEC_BOUNDS_CHECK',1),
                             ('MV_VECTOR_BOUNDS_CHECK',1),
                             ('PETSCVEC_BOUNDS_CHECK',1),
                             ('F77_POST_UNDERSCORE',1),
                             ('USE_BLAS',1)],
              include_dirs=['proteus',
                            numpy.get_include(),
                            str(get_pybind_include()),
                            str(get_pybind_include(user=True)),
                            PROTEUS_SUPERLU_INCLUDE_DIR,
                            PROTEUS_HDF5_INCLUDE_DIR] + \
              PROTEUS_PETSC_INCLUDE_DIRS + \
              PROTEUS_MPI_INCLUDE_DIRS,
              library_dirs=PROTEUS_PETSC_LIB_DIRS+PROTEUS_MPI_LIB_DIRS+PROTEUS_HDF5_LIB_DIRS,
              libraries=['hdf5','stdc++','m']+PROTEUS_PETSC_LIBS+PROTEUS_MPI_LIBS+PROTEUS_HDF5_LIBS,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS + PROTEUS_PETSC_EXTRA_LINK_ARGS,
              extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS + PROTEUS_PETSC_EXTRA_COMPILE_ARGS+PROTEUS_OPT),
    Extension('ctransportCoefficients',
              sources=['proteus/ctransportCoefficients.pyx','proteus/transportCoefficients.c'],
              include_dirs=[numpy.get_include(),'proteus'],
              depends=["proteus/transportCoefficients.h"],
              language="c",
              libraries=['m'],
              extra_compile_args=PROTEUS_OPT),
    Extension('csubgridError',
              sources=['proteus/csubgridError.pyx','proteus/subgridError.c'],
              depends=["proteus/subgridError.h"],
              language="c",
              include_dirs=[numpy.get_include(),'proteus'],
              libraries=['m'],
              extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS),
    Extension('cshockCapturing',
              sources=['proteus/cshockCapturing.pyx','proteus/shockCapturing.c'],
              depends=["proteus/shockCapturing.h"],
              language="c",
              include_dirs=[numpy.get_include(),'proteus'],
              libraries=['m'],
              extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS),
    Extension('superluWrappers',
              sources=['proteus/superluWrappers.pyx'],
              define_macros=[('PROTEUS_SUPERLU_H',PROTEUS_SUPERLU_H),
                             ('PROTEUS_BLAS_H',PROTEUS_BLAS_H)],
              language="c",
              include_dirs=[numpy.get_include(),
                            'proteus',
                            PROTEUS_SUPERLU_INCLUDE_DIR],
              library_dirs=[PROTEUS_SUPERLU_LIB_DIR,
                            PROTEUS_LAPACK_LIB_DIR,
                            PROTEUS_BLAS_LIB_DIR,
                            PROTEUS_METIS_LIB_DIR],
              libraries=['m',
                         PROTEUS_SUPERLU_LIB,
                         PROTEUS_LAPACK_LIB,PROTEUS_BLAS_LIB,
                         PROTEUS_METIS_LIB],
              extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS),
    Extension("csmoothers",
              sources=["proteus/csmoothers.pyx", "proteus/smoothers.c"],
              define_macros=[('PROTEUS_SUPERLU_H',PROTEUS_SUPERLU_H),
                             ('PROTEUS_LAPACK_H',PROTEUS_LAPACK_H),
                             ('PROTEUS_LAPACK_INTEGER',PROTEUS_LAPACK_INTEGER),
                             ('PROTEUS_BLAS_H',PROTEUS_BLAS_H)],
              language="c",
              include_dirs=['proteus',
                            numpy.get_include(),
                            PROTEUS_SUPERLU_INCLUDE_DIR,
                            PROTEUS_LAPACK_INCLUDE_DIR,
                            PROTEUS_BLAS_INCLUDE_DIR,
              ],
              library_dirs=[PROTEUS_SUPERLU_INCLUDE_DIR,
                            PROTEUS_SUPERLU_LIB_DIR,
                            PROTEUS_LAPACK_LIB_DIR,
                            PROTEUS_BLAS_LIB_DIR,
                            PROTEUS_METIS_LIB_DIR],
              libraries=['m',
                         PROTEUS_SUPERLU_LIB,
                         PROTEUS_LAPACK_LIB,
                         PROTEUS_BLAS_LIB,
                         PROTEUS_METIS_LIB],
              extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS),
    Extension("canalyticalSolutions",
              sources=["proteus/canalyticalSolutions.pyx", "proteus/analyticalSolutions.c"],
              depends=["proteus/analyticalSolutions.h"],
              extra_compile_args=PROTEUS_OPT,
              language="c", include_dirs=[numpy.get_include(), 'proteus']),
    Extension("clapack",
              sources=["proteus/clapack.pyx"],
              depends=["proteus/proteus_lapack.h","proteus/proteus_blas.h"],
              extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS,
              language="c",
              include_dirs=[numpy.get_include(), 'proteus',
                            PROTEUS_LAPACK_INCLUDE_DIR,
                            PROTEUS_BLAS_INCLUDE_DIR],
              library_dirs=[PROTEUS_LAPACK_LIB_DIR,PROTEUS_BLAS_LIB_DIR],
              libraries=['m',
                         PROTEUS_LAPACK_LIB,
                         PROTEUS_BLAS_LIB]),
    Extension("cpostprocessing",
              sources=["proteus/cpostprocessing.pyx","proteus/postprocessing.c"],
              depends=["proteus/postprocessing.h","proteus/postprocessing.pxd"],
              define_macros=[('PROTEUS_LAPACK_H',PROTEUS_LAPACK_H),
                             ('PROTEUS_LAPACK_INTEGER',PROTEUS_LAPACK_INTEGER),
                             ('PROTEUS_BLAS_H',PROTEUS_BLAS_H)],
              extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS,
              language="c",
              include_dirs=[numpy.get_include(), 'proteus',
                            PROTEUS_LAPACK_INCLUDE_DIR,
                            PROTEUS_BLAS_INCLUDE_DIR],
              library_dirs=[PROTEUS_LAPACK_LIB_DIR,PROTEUS_BLAS_LIB_DIR],
              libraries=['m',
                         PROTEUS_LAPACK_LIB,
                         PROTEUS_BLAS_LIB]),
    Extension('cnumericalFlux',
              sources=['proteus/cnumericalFlux.pyx','proteus/numericalFlux.c'],
              depends=["proteus/numericalFlux.h"],
              extra_compile_args=PROTEUS_OPT,
              language="c", include_dirs=[numpy.get_include(), 'proteus']),
    Extension('ctimeIntegration',
              sources=['proteus/ctimeIntegration.pyx','proteus/timeIntegration.c'],
              depends=["proteus/timeIntegration.h"],
              extra_compile_args=PROTEUS_OPT,
              language="c", include_dirs=[numpy.get_include(), 'proteus']),
    Extension("cTwophaseDarcyCoefficients",
              sources=["proteus/cTwophaseDarcyCoefficients.pyx",
               "proteus/SubsurfaceTransportCoefficients.cpp"],
              depends=["proteus/SubsurfaceTransportCoefficients.h",
                       "proteus/pskRelations.h",
                       "proteus/pskRelations.pxd",
                       "proteus/densityRelations.h",
                       "proteus/twophaseDarcyCoefficients.pxd",
                       "proteus/twophaseDarcyCoefficients.h"],
              define_macros=[('PROTEUS_SUPERLU_H',PROTEUS_SUPERLU_H),
                             ('CMRVEC_BOUNDS_CHECK',1),
                             ('MV_VECTOR_BOUNDS_CHECK',1),
                             ('PETSCVEC_BOUNDS_CHECK',1),
                             ('F77_POST_UNDERSCORE',1),
                             ('USE_BLAS',1)],
              include_dirs=['proteus',
                            numpy.get_include(),
                            PROTEUS_SUPERLU_INCLUDE_DIR,
                            PROTEUS_HDF5_INCLUDE_DIR] + \
              PROTEUS_PETSC_INCLUDE_DIRS + \
              PROTEUS_MPI_INCLUDE_DIRS,
              language="c++",
              library_dirs=PROTEUS_PETSC_LIB_DIRS+PROTEUS_MPI_LIB_DIRS+PROTEUS_HDF5_LIB_DIRS,
              libraries=['hdf5','stdc++','m']+PROTEUS_PETSC_LIBS+PROTEUS_MPI_LIBS+PROTEUS_HDF5_LIBS,
              extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS + PROTEUS_PETSC_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS + PROTEUS_PETSC_EXTRA_LINK_ARGS,
    ),
    Extension("cSubsurfaceTransportCoefficients",
              sources=["proteus/cSubsurfaceTransportCoefficients.pyx","proteus/SubsurfaceTransportCoefficients.cpp"],
              depends=["proteus/SubsurfaceTransportCoefficients.pxd",
                       "proteus/SubsurfaceTransportCoefficients.h"],
              define_macros=[('PROTEUS_SUPERLU_H',PROTEUS_SUPERLU_H),
                             ('CMRVEC_BOUNDS_CHECK',1),
                             ('MV_VECTOR_BOUNDS_CHECK',1),
                             ('PETSCVEC_BOUNDS_CHECK',1),
                             ('F77_POST_UNDERSCORE',1),
                             ('USE_BLAS',1)],
              include_dirs=['proteus',
                            numpy.get_include(),
                            PROTEUS_SUPERLU_INCLUDE_DIR,
                            PROTEUS_HDF5_INCLUDE_DIR] + \
              PROTEUS_PETSC_INCLUDE_DIRS + \
              PROTEUS_MPI_INCLUDE_DIRS,
              language="c++",
              library_dirs=PROTEUS_PETSC_LIB_DIRS+PROTEUS_MPI_LIB_DIRS+PROTEUS_HDF5_LIB_DIRS,
              libraries=['hdf5','stdc++','m']+PROTEUS_PETSC_LIBS+PROTEUS_MPI_LIBS+PROTEUS_HDF5_LIBS,
              extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS + PROTEUS_PETSC_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS + PROTEUS_PETSC_EXTRA_LINK_ARGS,
    ),
    Extension("cpskRelations",
              sources=["proteus/cpskRelations.pyx"],
              depends=["proteus/pskRelations.pxd",
                       "proteus/pskRelations.h"],
              define_macros=[('PROTEUS_SUPERLU_H',PROTEUS_SUPERLU_H),
                             ('CMRVEC_BOUNDS_CHECK',1),
                             ('MV_VECTOR_BOUNDS_CHECK',1),
                             ('PETSCVEC_BOUNDS_CHECK',1),
                             ('F77_POST_UNDERSCORE',1),
                             ('USE_BLAS',1)],
              include_dirs=['proteus',
                            numpy.get_include(),
                            PROTEUS_SUPERLU_INCLUDE_DIR,
                            PROTEUS_HDF5_INCLUDE_DIR] + \
              PROTEUS_PETSC_INCLUDE_DIRS + \
              PROTEUS_MPI_INCLUDE_DIRS,
              language="c++",
              library_dirs=PROTEUS_PETSC_LIB_DIRS+PROTEUS_MPI_LIB_DIRS+PROTEUS_HDF5_LIB_DIRS,
              libraries=['hdf5','stdc++','m']+PROTEUS_PETSC_LIBS+PROTEUS_MPI_LIBS+PROTEUS_HDF5_LIBS,
              extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS + PROTEUS_PETSC_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS + PROTEUS_PETSC_EXTRA_LINK_ARGS,
    ),
    Extension("cpartitioning",
              sources=["proteus/cpartitioning.pyx",
                       "proteus/partitioning.cpp",
                       "proteus/cartesianPartitioning.cpp",
                       'proteus/mesh.cpp',
                       'proteus/meshio.cpp',],
              depends=["proteus/partitioning.h",
                       "proteus/partitioning.cpp",
                       "proteus/cartesianPartitioning.cpp",
                       "proteus/cpartitioning.pyx",
                       "proteus/cmeshTools.pxd",
                       "proteus/mesh.h",
                       "proteus/index_type.h",
                       'proteus/mesh.cpp',
                       'proteus/meshio.cpp'],
              define_macros=[('PROTEUS_SUPERLU_H',PROTEUS_SUPERLU_H),
                             ('CMRVEC_BOUNDS_CHECK',1),
                             ('MV_VECTOR_BOUNDS_CHECK',1),
                             ('PETSCVEC_BOUNDS_CHECK',1),
                             ('F77_POST_UNDERSCORE',1),
                             ('USE_BLAS',1)],
              include_dirs=['proteus',
                            numpy.get_include(),
                            str(get_pybind_include()),
                            str(get_pybind_include(user=True)),
                            PROTEUS_SUPERLU_INCLUDE_DIR,
                            PROTEUS_HDF5_INCLUDE_DIR] +
              PROTEUS_PETSC_INCLUDE_DIRS + PROTEUS_MPI_INCLUDE_DIRS,
              language="c++",
              library_dirs=PROTEUS_PETSC_LIB_DIRS+PROTEUS_MPI_LIB_DIRS+PROTEUS_HDF5_LIB_DIRS,
              libraries=['hdf5','stdc++','m']+PROTEUS_PETSC_LIBS+PROTEUS_MPI_LIBS+PROTEUS_HDF5_LIBS,
              extra_compile_args=['-std=c++20']+PROTEUS_EXTRA_COMPILE_ARGS + PROTEUS_PETSC_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
              extra_link_args=PROTEUS_EXTRA_LINK_ARGS + PROTEUS_PETSC_EXTRA_LINK_ARGS,
    ),
    # Extension("flcbdfWrappers",["proteus/flcbdfWrappers.pyx"],
    #           language="c++",
    #           depends=["proteus/flcbdfWrappers.pxd"],
    #           define_macros=[('PROTEUS_SUPERLU_H',PROTEUS_SUPERLU_H),
    #                          ('CMRVEC_BOUNDS_CHECK',1),
    #                          ('MV_VECTOR_BOUNDS_CHECK',1),
    #                          ('PETSCVEC_BOUNDS_CHECK',1),
    #                          ('F77_POST_UNDERSCORE',1),
    #                          ('USE_BLAS',1)],
    #           include_dirs=['proteus',
    #                         numpy.get_include(),
    #                         PROTEUS_SUPERLU_INCLUDE_DIR,
    #                         PROTEUS_DAETK_INCLUDE_DIR,
    #                         PROTEUS_HDF5_INCLUDE_DIR] + \
    #           PROTEUS_PETSC_INCLUDE_DIRS + \
    #           PROTEUS_MPI_INCLUDE_DIRS,
    #           library_dirs=[PROTEUS_DAETK_LIB_DIR]+PROTEUS_PETSC_LIB_DIRS+PROTEUS_MPI_LIB_DIRS+PROTEUS_HDF5_LIB_DIRS,
    #           libraries=['hdf5','stdc++','m',PROTEUS_DAETK_LIB]+PROTEUS_PETSC_LIBS+PROTEUS_MPI_LIBS+PROTEUS_HDF5_LIBS,
    #           extra_link_args=PROTEUS_EXTRA_LINK_ARGS + PROTEUS_PETSC_EXTRA_LINK_ARGS,
    #           extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS + PROTEUS_PETSC_EXTRA_COMPILE_ARGS+PROTEUS_OPT,
    # ),
    Extension(
        'mprans.cCLSVOF',
        sources=['proteus/mprans/CLSVOF.cpp'],
        depends=["proteus/mprans/CLSVOF.h", "proteus/mprans/CLSVOF.h"] + ["proteus/ModelFactory.h","proteus/CompKernel.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cNCLS',
        sources=['proteus/mprans/NCLS.cpp'],
        depends=["proteus/mprans/NCLS.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h"] + ["proteus/ModelFactory.h","proteus/CompKernel.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cMCorr',
        sources=['proteus/mprans/MCorr.cpp'],
        depends=["proteus/mprans/MCorr.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h"] + ["proteus/ModelFactory.h","proteus/CompKernel.h"] + [
            "proteus/equivalent_polynomials.h",
            "proteus/equivalent_polynomials_utils.h",
            "proteus/equivalent_polynomials_coefficients.h",
            'proteus/equivalent_polynomials_coefficients_quad.h'],
        define_macros=[('PROTEUS_LAPACK_H',PROTEUS_LAPACK_H),
                       ('PROTEUS_LAPACK_INTEGER',PROTEUS_LAPACK_INTEGER),
                       ('PROTEUS_BLAS_H',PROTEUS_BLAS_H)],
        include_dirs=get_pybind_include_dirs(),
        library_dirs=[PROTEUS_LAPACK_LIB_DIR,
                      PROTEUS_BLAS_LIB_DIR],
        libraries=['m',PROTEUS_LAPACK_LIB,PROTEUS_BLAS_LIB],
        extra_compile_args=PROTEUS_EXTRA_COMPILE_ARGS+PROTEUS_OPT+['-std=c++20'],
        extra_link_args=PROTEUS_EXTRA_LINK_ARGS,
        language='c++'),
    Extension(
        'mprans.cRANS2P',
        sources=['proteus/mprans/RANS2P.cpp'],
        depends=["proteus/mprans/RANS2P.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h"] + ["proteus/MixedModelFactory.h","proteus/CompKernel.h"] + [
            "proteus/equivalent_polynomials.h",
            "proteus/equivalent_polynomials_utils.h",
            "proteus/equivalent_polynomials_coefficients.h",
            'proteus/equivalent_polynomials_coefficients_quad.h'],
        include_dirs=get_pybind_include_dirs() + PROTEUS_MPI_INCLUDE_DIRS,
        extra_compile_args=PROTEUS_OPT+PROTEUS_MPI_LIB_DIRS+['-std=c++20'],#,'-fopenmp'],#,'-DXTENSOR_USE_OPENMP'],
        library_dirs=PROTEUS_MPI_LIB_DIRS+[PROTEUS_LAPACK_LIB_DIR,
                      PROTEUS_BLAS_LIB_DIR],
        libraries=PROTEUS_MPI_LIBS+['m',
                                    PROTEUS_LAPACK_LIB,
                                    PROTEUS_BLAS_LIB],
        extra_link_args=PROTEUS_SCOREC_EXTRA_LINK_ARGS+PROTEUS_EXTRA_LINK_ARGS,#+['-fopenmp'],
        language='c++'),
    Extension(
        'mprans.cRANS2P_IB',
        sources=['proteus/mprans/RANS2P_IB.cpp'],
        depends=["proteus/mprans/RANS2P_IB.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h"] + ["proteus/MixedModelFactory.h","proteus/CompKernel.h"] + [
            "proteus/equivalent_polynomials.h",
            "proteus/equivalent_polynomials_utils.h",
            "proteus/equivalent_polynomials_coefficients.h",
            'proteus/equivalent_polynomials_coefficients_quad.h'],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cRANS2P2D',
        sources=['proteus/mprans/RANS2P2D.cpp'],
        depends=["proteus/mprans/RANS2P2D.h"] + ["proteus/MixedModelFactory.h","proteus/CompKernel.h"] + [
            "proteus/equivalent_polynomials.h",
            "proteus/equivalent_polynomials_utils.h",
            "proteus/equivalent_polynomials_coefficients.h",
            'proteus/equivalent_polynomials_coefficients_quad.h'],
        include_dirs=get_pybind_include_dirs() + PROTEUS_MPI_INCLUDE_DIRS,
        extra_compile_args=PROTEUS_OPT+PROTEUS_MPI_LIB_DIRS+['-std=c++20'],#,'-fopenmp'],#,'-DXTENSOR_USE_OPENMP'],
        library_dirs=PROTEUS_MPI_LIB_DIRS+[PROTEUS_LAPACK_LIB_DIR,
                      PROTEUS_BLAS_LIB_DIR],
        libraries=PROTEUS_MPI_LIBS+['m',
                                    PROTEUS_LAPACK_LIB,
                                    PROTEUS_BLAS_LIB],
        extra_link_args=PROTEUS_SCOREC_EXTRA_LINK_ARGS+PROTEUS_EXTRA_LINK_ARGS,#+['-fopenmp'],
        language='c++'),
    Extension(
        'mprans.cRDLS',
        sources=['proteus/mprans/RDLS.cpp'],
        depends=["proteus/mprans/RDLS.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h"] + ["proteus/ModelFactory.h","proteus/CompKernel.h"] + [
            "proteus/equivalent_polynomials.h",
            "proteus/equivalent_polynomials_utils.h",
            "proteus/equivalent_polynomials_coefficients.h",
            'proteus/equivalent_polynomials_coefficients_quad.h'],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cVOF',
        sources=['proteus/mprans/VOF.cpp'],
        depends=["proteus/mprans/VOF.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h", "proteus/ModelFactory.h","proteus/CompKernel.h",
                 "proteus/equivalent_polynomials.h",
                 "proteus/equivalent_polynomials_utils.h",
                 "proteus/equivalent_polynomials_coefficients.h",
                 "proteus/equivalent_polynomials_coefficients_quad.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cTADR',
        sources=['proteus/mprans/TADR.cpp'],
        depends=["proteus/mprans/TADR.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h", "proteus/ModelFactory.h","proteus/CompKernel.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cMoveMesh',
        ['proteus/mprans/MoveMesh.cpp'],
        depends=["proteus/mprans/MoveMesh.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h"] + ["proteus/ModelFactory.h","proteus/CompKernel.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cMoveMesh2D',
        sources=['proteus/mprans/MoveMesh2D.cpp'],
        depends=["proteus/mprans/MoveMesh2D.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h"] + ["proteus/ModelFactory.h","proteus/CompKernel.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cSW2D',
        sources=['proteus/mprans/SW2D.cpp'],
        depends=["proteus/mprans/SW2D.h", "proteus/mprans/SW2D.h"] + ["proteus/ModelFactory.h","proteus/CompKernel.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cSW2DCV',
        sources=['proteus/mprans/SW2DCV.cpp'],
        depends=["proteus/mprans/SW2DCV.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h", "proteus/ModelFactory.h","proteus/CompKernel.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cGN_SW2DCV',
        sources=['proteus/mprans/GN_SW2DCV.cpp'],
        depends=["proteus/mprans/GN_SW2DCV.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h"] + ["proteus/ModelFactory.h","proteus/CompKernel.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cKappa',
        sources=['proteus/mprans/Kappa.cpp'],
        depends=["proteus/mprans/Kappa.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h"] + ["proteus/ModelFactory.h","proteus/CompKernel.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cKappa2D',
        sources=['proteus/mprans/Kappa2D.cpp'],
        depends=["proteus/mprans/Kappa2D.h"] + ["proteus/ModelFactory.h","proteus/CompKernel.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cDissipation',
        sources=['proteus/mprans/Dissipation.cpp'],
        depends=["proteus/mprans/Dissipation.h", "proteus/mprans/ArgumentsDict.h", "proteus/pyarray.h"] + ["proteus/ModelFactory.h","proteus/CompKernel.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
    Extension(
        'mprans.cDissipation2D',
        sources=['proteus/mprans/Dissipation2D.cpp'],
        depends=["proteus/mprans/Dissipation2D.h"] + ["proteus/ModelFactory.h","proteus/CompKernel.h"],
        include_dirs=get_pybind_include_dirs(),
        extra_compile_args=PROTEUS_OPT+['-std=c++20'],
        language='c++'),
]

import os as _os
# PUMI/SCOREC (mesh adaptation) and Chrono (rigid-body/FSI coupling) are
# unrelated optional dependencies; PROTEUS_SKIP_PUMI and PROTEUS_SKIP_CHRONO
# let a build path enable one without the other. PROTEUS_SKIP_PUMI_CHRONO
# is kept for backwards compatibility and skips both.
_skip = set()
if _os.environ.get("PROTEUS_SKIP_PUMI_CHRONO") or _os.environ.get("PROTEUS_SKIP_PUMI"):
    _skip.add("MeshAdaptPUMI.MeshAdapt")
if _os.environ.get("PROTEUS_SKIP_PUMI_CHRONO") or _os.environ.get("PROTEUS_SKIP_CHRONO"):
    _skip.add("mbd.CouplingFSI")
EXTENSIONS_TO_BUILD = [e for e in EXTENSIONS_TO_BUILD if e.name not in _skip]

def setup_given_extensions(extensions):
    # Most Extensions above list several *_LIB_DIR constants (SUPERLU, LAPACK,
    # BLAS, PETSC, HDF5, ...) in library_dirs -- in a conda/mamba dev install
    # (environment-dev.yml) these all resolve to the same single conda env
    # lib directory, so library_dirs ends up with that same path repeated
    # several times. conda-forge's `compilers` package patches distutils to
    # emit '-Wl,-rpath,<dir>' for every library_dirs entry (not just once per
    # unique directory), and setuptools/distutils separately also appends the
    # ambient $LDFLAGS (which itself already has that same rpath, courtesy of
    # the same `compilers` activation script) -- together this produces a
    # linked .so with a literal duplicate LC_RPATH load command, which dyld
    # refuses to open at import time ("... (duplicate LC_RPATH ...)").
    # Deduplicating library_dirs here (order-preserving) collapses the
    # repeats back down to one -rpath per real directory. Confirmed via a
    # real `pip install -e .` against environment-dev.yml.
    for ext in extensions:
        if getattr(ext, 'library_dirs', None):
            ext.library_dirs = list(dict.fromkeys(ext.library_dirs))
    setup(name='proteus',
          version='2.0.0.dev',
          classifiers=[
              'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Environment :: Web Environment',
              'Intended Audience :: End Users/Desktop',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3 :: Only',
              'Programming Language :: Python :: Implementation :: CPython',
              'Topic :: Scientific/Engineering :: Mathematics',
              'Topic :: Scientific/Engineering :: Physics',
              'Operating System :: MacOS :: MacOS X',
              'Operating System :: POSIX',
          ],
          description='Python tools for multiphysics modeling',
          author='The Proteus Developers',
          author_email='proteus-dev@googlegroups.com',
          url='http://proteustoolkit.org',
          python_requires='>=3.9',
          install_requires=[
              'numpy',
              'scipy',
              'mpi4py',
              'petsc4py',
              'h5py',
              # The archive of record is a .ymf file written through
              # ymf.archive (pyyaml only). The heavy front end,
              # ymf[symbolic]/ymf[frontend] (sympy, strictyaml, pint), is
              # deliberately not required. ymf is not on PyPI or
              # conda-forge yet, so install it first (the environment
              # files do, from git). Not a direct "ymf @ git+..."
              # reference: pip re-fetches those even when ymf is
              # installed, which fails on offline compute nodes and
              # replaces an editable checkout. No version floor either:
              # a --no-build-isolation build without setuptools_scm
              # reports ymf as 0.0.0.
              'ymf',
          ],
          extras_require={
              'test': ['pytest', 'pytest-cov', 'pytest-xdist', 'pytest-forked', 'matplotlib'],
          },
          packages = ['proteus',
                      'proteus.fenton',
                      'proteus.mprans',
                      'proteus.richards',
                      'proteus.m_comp_co2',
                      'proteus.elastoplastic',
                      'proteus.mbd',
                      'proteus.test_utils',
                      'proteus.config',
                      'proteus.TwoPhaseFlow',
                      'proteus.TwoPhaseFlow.utils',
                      'proteus.SWFlow',
                      'proteus.SWFlow.utils',
                      'proteus.SWFlow.models',
                      'proteus.MeshAdaptPUMI',
                      'proteus.cli',
          ],
          entry_points={'console_scripts': ['pro = proteus.cli:main']},
          cmdclass = {'build_ext':custom_build_ext},
          ext_package='proteus',
          ext_modules=extensions,
        #   ext_modules=cythonize(extensions, gdb_debug=True),
          data_files=[(proteus_install_path,
                       ['proteus/proteus_blas.h',
                        'proteus/proteus_lapack.h',
                        'proteus/proteus_superlu.h',
                        'proteus/ModelFactory.h',
                        'proteus/CompKernel.h'
                       ]),
         ],
          scripts = ['scripts/parun','scripts/gatherArchives.py','scripts/qtm','scripts/waves2xmf','scripts/povgen.py',
                     'scripts/velocity2xmf','scripts/run_script_garnet','scripts/run_script_diamond',

                     'scripts/run_script_lonestar','scripts/run_script_ranger','scripts/run_script_mpiexec','scripts/gatherTimes','scripts/clearh5.py',
                     'scripts/runSWEs.py']
    )

def setup_extensions_in_sequential():
    setup_given_extensions(EXTENSIONS_TO_BUILD)

def _setup_one_extension(extensions):
    """Pool task for setup_extensions_in_parallel: build one extension and
    report how it went instead of raising. distutils ends a failed compile
    with SystemExit, which is a BaseException, so it escapes the pool's own
    exception handling and kills the worker; a dead worker never posts its
    result and Pool.join() then waits forever (34 idle processes for 30
    minutes on a 1.9.x build with missing headers, 2026-11-02)."""
    try:
        setup_given_extensions(extensions)
        return extensions[0].name, None
    except BaseException as e:
        return extensions[0].name, '%s: %s' % (type(e).__name__, e)

def setup_extensions_in_parallel(nprocs):
    import multiprocessing, logging
    mp = multiprocessing.get_context('fork')
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)
    EXTENSIONS=[[e] for e in EXTENSIONS_TO_BUILD]
    failed = []
    # Read every result: a failure has to end the build with a non-zero
    # exit, not be dropped (imap's results were never read before, so any
    # failure that did not hang was lost and the build exited 0).
    with mp.Pool(processes=nprocs) as pool:
        for name, error in pool.imap_unordered(_setup_one_extension, EXTENSIONS):
            if error is not None:
                failed.append((name, error))
                sys.stderr.write('proteus setup.py: building %s failed: %s\n' % (name, error))
    if failed:
        sys.stderr.write('proteus setup.py: %d of %d extensions failed to build: %s\n'
                         % (len(failed), len(EXTENSIONS), ', '.join(sorted(n for n, _ in failed))))
        sys.exit(1)

import logging, multiprocessing
logging.basicConfig(force=True,level=logging.INFO)
mp = multiprocessing.get_context('fork')
logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)

# N=<processes> builds one extension per process; without it (or with a
# value that is not a positive integer) the build is sequential over the
# extensions, with setuptools' own threads inside build_ext.
try:
    _nprocs = int(os.environ.get('N', ''))
except ValueError:
    _nprocs = 0
if "build_ext" in sys.argv and _nprocs > 0:
    setup_extensions_in_parallel(_nprocs)
else:
    setup_extensions_in_sequential()
