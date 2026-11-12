# Proteus: Computational Methods and Simulation Toolkit [![Build Status](https://github.com/proteustoolkit/proteus/actions/workflows/sdist.yml/badge.svg?branch=main)](https://github.com/proteustoolkit/proteus/actions/workflows/sdist.yml) [![DOI](https://zenodo.org/badge/2212385.svg)](https://zenodo.org/badge/latestdoi/2212385)


Proteus (https://proteustoolkit.org) is a Python package for
rapidly developing computer models and numerical methods.

# Installation

## conda / mamba

```bash
mamba install proteus -c conda-forge
```

## conda / mamba for proteus development

For a development installation, you want to install Proteus's dependencies and compile Proteus from source. Two dependency files are provided, differing only in which MPI implementation they use -- see "Choosing between OpenMPI and MPICH" below. If in doubt, use the OpenMPI one:

```bash
mamba env create -f environment-openmpi-dev.yml # or environment-mpich-dev.yml; environment-dev-up.yml to try unpinned dependencies
mamba activate proteus-dev-openmpi              # or proteus-dev-mpich
# --no-build-isolation: without it pip builds in a throwaway environment and
# re-fetches build dependencies from PyPI, so proteus's extensions compile
# against PyPI numpy's headers while linking this env's numpy at runtime
# (visible as -I/tmp/pip-build-env-*/overlay/.../numpy/_core/include in the
# compile lines). The env already provides setuptools/cython/numpy/pybind11.
pip install --no-build-isolation --no-cache-dir -v -e .
```

## conda / mamba with updated/unpinned dependencies

`environment-openmpi-dev.yml`/`environment-mpich-dev.yml` pin conda-forge
builds of everything proteus links against (PETSc, MPI, HDF5,
SuperLU/SuperLU_DIST, METIS/ParMETIS, Chrono, PUMI, Triangle/TetGen,
...), so this is the path to use if you want the optional Chrono
(multibody/FSI) and PUMI (mesh adaptation) support built in. Both
intentionally exclude the `defaults` channel (`nodefaults`) -- conda-forge
alone resolves everything here, and `defaults` pulls in `repo.anaconda.com`,
a commercial channel with its own Terms of Service and registration-gated
rate limits.

### Choosing between OpenMPI and MPICH

The two files are identical except for the MPI provider (and the build-string
wildcard on every MPI-linked package: `hdf5`, `h5py`, `scorec`, `zoltan`).

- **`environment-openmpi-dev.yml` (default/recommended).** Pins OpenMPI
  5.0.10. Known caveat: OpenMPI is not fork()-safe, so running the *entire*
  test suite in one process via `pytest --forked` can produce spurious
  SIGSEGV crashes in unrelated test files (an artifact of forking after MPI
  is initialized, not a proteus bug) -- run test files individually, or with
  `pytest-xdist` using separate worker processes, if you hit this.
- **`environment-mpich-dev.yml`.** Pins MPICH 5.0.1. On Linux aarch64 this is
  currently the *only* mpich version that works at all: conda-forge's scorec
  build there requires `mpich>=5.0,<6.0a0`, and of the two aarch64 mpich
  releases `>=5.0` (`5.0.0` and `5.0.1`), `5.0.0` fails outright at compile
  time against conda-forge's aarch64 `petsc` package (`petscsys.h` hard-errors
  on the mpi.h version mismatch). MPICH 5.0.1 itself has an open, observed bug
  on aarch64 where real PETSc/hypre solves (BoomerAMG preconditioning) can
  abort with `MPI_Allreduce() function was called before MPI_INIT was
  invoked` (exit code 14) -- see the comment header in
  `environment-mpich-dev.yml` for the reproduction and verification details.
  If you hit this, switch to `environment-openmpi-dev.yml`.

  **On macOS, set `FI_PROVIDER=tcp` with MPICH.** conda-forge's MPICH runs
  over libfabric, which on macOS picks its `sockets` provider. That provider
  starts two connection threads that busy-poll at 100% CPU each from
  `MPI_Init` on, in every process, so each MPI rank holds about three cores,
  half of it system time. Its `tcp` provider has no such threads and runs
  multi-rank jobs just as well. `import proteus` sets `FI_PROVIDER=tcp` on
  macOS unless it is already set, which covers `parun` and anything that
  imports proteus before MPI starts. Set it yourself (`export FI_PROVIDER=tcp`)
  for `mpiexec` runs of other programs, or when your own script imports
  `mpi4py.MPI` or `petsc4py.PETSc` before `proteus`. Seen with MPICH 4.3.2,
  5.0.1 and 5.0.2 and libfabric 2.2.0 and 2.6.0; Open MPI is not affected.

On macOS/arm64 (Apple Silicon), `environment-openmpi-dev.yml` requires
`scorec >=4.2.1` and will not solve with anything older. conda-forge's `scorec`
had mpich-only builds on that platform until 4.2.1 build 1
(conda-forge/scorec-feedstock#32 removed the `skip:`), so an older `scorec` pin
cannot resolve against OpenMPI there. With the version floor the dependency
files already carry, both files solve on Apple Silicon and either is fine.

On x86_64 and Linux/aarch64, both files resolve equally well and neither known
issue above has been observed; OpenMPI remains the recommended default there too
for consistency.

## pip

Proteus is not a pure-Python package: most of it is C/C++/Cython/Fortran
extensions linking against PETSc, MPI, HDF5, SuperLU/SuperLU_DIST,
METIS/ParMETIS, and BLAS/LAPACK. All of that can be provisioned with pip
alone (no conda, no system package manager beyond a C/C++/Fortran compiler
and `make`) using PETSc's own `--download-x` configure options, exposed to
its PyPI package via the `PETSC_CONFIGURE_OPTIONS` environment variable.
This builds everything from source, so expect it to take a while.

The recipe below also builds Chrono (multibody/FSI) and PUMI (mesh
adaptation) via PETSc's `--download-chrono`/`--download-pumi` (plus their
own extra dependencies: `--download-eigen` for Chrono, `--download-zoltan`
for PUMI). PUMI's build additionally needs `libbz2`'s development
package on the system already (Debian/Ubuntu: `apt install libbz2-dev`;
most systems already have the runtime `libbz2` library but not its
link-time `.so` symlink, which is what's actually needed here). If you'd
rather skip one or both (e.g. to avoid that extra build time, or if you
can't install `libbz2-dev`), set `PROTEUS_SKIP_PUMI=1` and/or
`PROTEUS_SKIP_CHRONO=1` (or `PROTEUS_SKIP_PUMI_CHRONO=1` for both at once)
before the final `pip install` step below, and drop the corresponding
`--download-x` options above it.

Three pip flags matter throughout this section, and they are about
correctness rather than tidiness:

- **`--no-build-isolation`** for anything that links the native stack
  (`petsc4py`, `h5py`, `proteus` itself). Build isolation gives the build a
  throwaway environment and re-fetches build dependencies from PyPI, so it does
  not see -- and will happily rebuild -- the PETSc/HDF5/MPI you just installed.
  This is why the build dependencies are installed up front instead.
- **`--no-cache-dir`** for those same installs. pip's locally-built wheel cache
  is keyed on the requirement URL, its hash, and the interpreter name/version --
  and *nothing* about the native libraries the wheel linked against. So an
  `h5py` wheel built earlier against a different HDF5 is a cache hit here and
  gets reused silently. Note `--no-binary` does not protect against this: it
  blocks prebuilt wheels from PyPI, not pip's own cached builds of your sdists.
- **`--no-binary <pkg>`**, scoped per package -- *not* `:all:`. Only the
  packages that link this stack must be built from source (`h5py`, `petsc4py`,
  `mpi4py`); h5py's own wheels in particular bundle a private serial HDF5, which
  silently costs you parallel I/O (`can't use mpio driver`). Using `:all:` would
  also drag numpy, scipy and cmake into source builds for no correctness gain.

```bash
python -m venv proteus-env && source proteus-env/bin/activate
# setuptools: needed by the final --no-build-isolation step below, which
# uses the active venv's own packages rather than an isolated sandbox.
# scipy: several of proteus's own modules (BoundaryConditions, SpatialTools)
# import it directly.
pip install --no-cache-dir cython pybind11 wheel numpy \
    "cmake>=3.29" setuptools scipy pkgconfig
# mpi4py from source: its PyPI wheels are built against one MPI's ABI, and
# MPICH's and Open MPI's are not interchangeable (MPI_Comm is `int` on MPICH and
# a pointer on Open MPI), so a wheel can import against the wrong libmpi.
pip install --no-cache-dir --no-binary mpi4py mpi4py

# chrono/pumi aren't in upstream PETSc, so install from our fork rather than
# PyPI's own `petsc` package -- only this fork's PETSc knows about them.
# --download-cmake: PETSc's own configure can't see the cmake pip just
# installed above from inside its build-isolation sandbox.
# --download-hypre: several of proteus's own solver tests configure
# pc_type=hypre; without this they fail with "PCSetType(): Unknown type".
# --download-eigen/--download-zoltan: required by --download-chrono/
# --download-pumi respectively, not optional once those are requested.
export PETSC_CONFIGURE_OPTIONS="--download-fblaslapack --download-superlu --download-superlu_dist --download-metis --download-parmetis --download-hdf5 --download-triangle --download-triangle-build-exec=1 --download-tetgen --download-tetgen-build-exec=1 --download-cmake --download-hypre --download-eigen --download-zoltan --download-chrono --download-pumi"
pip install "petsc @ git+https://gitlab.com/cekees/petsc.git@download-proteus-support"
# --no-build-isolation is required, not optional: petsc4py declares `petsc` as a
# build backend dependency, so under isolation pip builds a SECOND petsc from
# PyPI in a throwaway environment -- discarding the fork build above -- and that
# build fails ("RuntimeError: 256" in petsc's config, then "Failed to build
# 'petsc4py' when installing backend dependencies for petsc4py").
pip install --no-build-isolation --no-cache-dir --no-binary petsc4py petsc4py

# h5py's own PyPI wheels are serial-only; build against the parallel HDF5
# PETSc just downloaded and compiled above instead:
PETSC_DIR=$(python -c "import petsc; print(petsc.get_petsc_dir())")
# Work around a packaging quirk in the `petsc` wheel: its libhdf5.so is a
# linker script (`INPUT(libhdf5.so.NNN)`), not a real file, which trips up
# h5py's own HDF5 version/config introspection at build time.
real_hdf5=$(readlink -f "$PETSC_DIR"/lib/libhdf5.so.[0-9]* 2>/dev/null | head -1)
ln -sf "$(basename "$real_hdf5")" "$PETSC_DIR/lib/libhdf5.so"
CC=mpicc HDF5_MPI=ON HDF5_DIR="$PETSC_DIR" pip install \
    --no-build-isolation --no-cache-dir --no-binary h5py h5py

export PETSC_ARCH=""
export PROTEUS_PREFIX="$PETSC_DIR"
# To skip PUMI and/or Chrono instead of building them (see above),
# uncomment as needed -- and drop the matching --download-x option(s) too:
# export PROTEUS_SKIP_PUMI=1
# export PROTEUS_SKIP_CHRONO=1
# export PROTEUS_SKIP_PUMI_CHRONO=1   # both at once
# Chrono's own Python bindings (pychrono, imported directly by
# CouplingFSI.pyx and several tests) live here, not in site-packages:
export PYTHONPATH="$PETSC_DIR/share/chrono/python"
# Only needed if your system's mpicc doesn't put mpi.h somewhere proteus's
# own build already looks (get_flags('mpi') in proteus/config/default.py
# expects $MPI_DIR/include/mpi.h specifically). Most OpenMPI/Homebrew
# installs already have mpi.h under a plain <prefix>/include, so deriving
# MPI_DIR as the parent of mpicc's own -I flag (the dirname below) works
# directly. Debian/Ubuntu's mpich package does NOT follow that layout --
# its mpi.h lives at a multiarch-nested path like
# /usr/include/x86_64-linux-gnu/mpich/mpi.h, which has no plain
# "<prefix>/include" ancestor at all; naively taking dirname() of that
# points MPI_DIR at a directory whose own include/mpi.h doesn't exist,
# and proteus's build fails with "fatal error: mpi.h: No such file or
# directory" partway through (confirmed on Ubuntu 22.04/24.04/26.04 with
# the distro's own mpich package). Detect which layout you're in and, for
# the nested case, build a small symlink shim so $MPI_DIR/include/mpi.h
# resolves correctly either way:
_mpi_inc=$(mpicc -show | grep -o '\-I[^ ]*' | head -1 | cut -c3-)
if [ "$(basename "$_mpi_inc")" = "include" ]; then
  export MPI_DIR=$(dirname "$_mpi_inc")
else
  mkdir -p "$HOME/.proteus_mpi_dir"
  ln -sf "$_mpi_inc" "$HOME/.proteus_mpi_dir/include"
  ln -sf "$(mpicc -show | grep -o '\-L[^ ]*' | head -1 | cut -c3-)" "$HOME/.proteus_mpi_dir/lib"
  export MPI_DIR="$HOME/.proteus_mpi_dir"
fi
# $PETSC_DIR/bin needs to stay on PATH after this point too, not just for
# the build below -- it's where the `triangle` CLI
# --download-triangle-build-exec=1 built lives, and some of proteus's own
# tests shell out to it.
export PATH="$PETSC_DIR/bin:$PATH"
pip install --no-build-isolation --no-cache-dir --no-deps .
```

If your PETSc is already built (e.g. by an HPC site, or you don't need
Chrono/PUMI and can use PyPI's own `petsc` package directly), skip straight to
the `petsc4py`/`h5py`/proteus steps with `PETSC_DIR` (and `PETSC_ARCH`, if
set) pointing at that install; `PETSC_CONFIGURE_OPTIONS` is only consulted
when `petsc4py`'s own install triggers a fresh PETSc build.

If PyPI does not have gmsh and you want to use or run tests with gmsh, get
it from elsewhere, (e.g. from the gmsh website or conda).

A gmsh mesh (a `use_gmsh` domain, or a `.msh` file) is read through the ymf mesh that
`ymf-import-gmsh` converts it to, so install that too:
`pip install git+https://github.com/proteustoolkit/ymf-import-gmsh.git` (the dev environment files
do). Without it, a `.msh` with no current `.ymf` beside it is an error that names the package.

This path is new and less battle-tested than the conda/mamba route above and the PETSc BuildSystem/HPC routes below -- if something here breaks, those are
the more mature fallbacks.

## Spack

[Spack](https://spack.io) builds proteus and its native dependency chain
(PETSc, MPI, HDF5, SuperLU, METIS/ParMETIS, and optionally
PUMI) from source, with no conda and no system package manager
beyond a C/C++/Fortran compiler. `py-proteus` isn't in spack-packages
`develop` yet; until it's merged, add it from the `py-proteus` branch of
https://github.com/cekees/spack-packages:

```bash
git clone https://github.com/spack/spack.git
. spack/share/spack/setup-env.sh
git clone -b py-proteus https://github.com/cekees/spack-packages.git
spack repo add spack-packages/repos/spack_repo/builtin

spack install py-proteus            # Chrono and PUMI both disabled
spack install py-proteus+pumi       # adds PUMI (SCOREC/core) mesh adaptation
spack load py-proteus
```

Chrono has no upstream Spack package yet and stays disabled either way;
`+pumi` adds `pumi`, `zoltan`, and `parmetis` for mesh adaptation. Like
the pip path above, this one is new -- fall back to conda/mamba or HPC
`--download-proteus` if something here breaks.

## HPC (PETSc BuildSystem)

For installation on high performance environments, the recommended path is
PETSc's own build system, which most HPC sites and module systems already
support: `git@gitlab.com:cekees/petsc.git`, branch `download-proteus-support`,
adds a `--download-proteus` PETSc package (plus PETSc packages for proteus's
optional native dependencies: chrono, pumi) so a single
`./configure` + `make` builds PETSc, proteus, and everything in between,
instead of the separate per-dependency manual build this section used to describe. You will need a user-installable prefix directory, which below is assumed to be a conda environment, but a Python venv should also work.

```bash
git clone git@gitlab.com:cekees/petsc.git
cd petsc
git checkout download-proteus-support
CONDA_PREFIX=/path/to/your/conda/env PREFIX=/path/to/install ./configure_macos_arm64.sh   # macOS/arm64
make PETSC_DIR="$(pwd)" PETSC_ARCH=arch-darwin-download-proteus all
make PETSC_DIR="$(pwd)" PETSC_ARCH=arch-darwin-download-proteus install
```

On Linux (x86_64 or aarch64/ARM64), use `configure_linux_unified.sh`
instead, which downloads and builds its own MPI/OpenBLAS/CMake straight
into the same prefix rather than expecting an externally-supplied conda
env to provide them — the toolchain env only needs to supply Python
(+numpy) and a C/C++/Fortran compiler:

```bash
git clone git@gitlab.com:cekees/petsc.git
cd petsc
git checkout download-proteus-support
PREFIX=/path/to/install ./setup_env_miniforge.sh   # minimal, throwaway Python env
CONDA_PREFIX=/path/to/install PREFIX=/path/to/install PETSC_PREFIX=/path/to/install \
  ./configure_linux_unified.sh
make PETSC_DIR="$(pwd)" PETSC_ARCH=arch-linux-download-proteus all
make PETSC_DIR="$(pwd)" PETSC_ARCH=arch-linux-download-proteus install
```

(A plain Python venv works too in place of `setup_env_miniforge.sh` — see
`setup_env_venv.sh` — as does an already-existing conda/mamba env, in
which case skip straight to `configure_linux_unified.sh` with
`CONDA_PREFIX`/`PREFIX`/`PETSC_PREFIX` all pointing at it.)

Validated end-to-end on macOS/arm64, Linux/x86_64, and Linux/ARM64
(aarch64) — both the plain-venv and Miniforge-Python variants of the Linux
path pass proteus's full test suite cleanly, modulo a small set of already
independently-tracked numeric-tolerance and `gmsh`-packaging issues (not
install-pathway problems). If you're building on a platform other than
these three and hit trouble, that branch's `README_PROTEUS.md` and the
comments in `config/BuildSystem/config/packages/proteus.py`/`pumi.py`
are the places to start (in particular, a couple of post-install fixups in
there are known to be macOS/dyld-specific workarounds, not applicable
elsewhere).

The package definitions themselves are plain PETSc packages and don't
depend on that specific fork -- if your site already has its own PETSc
checkout, copying `config/BuildSystem/config/packages/{proteus,chrono,pumi,numpy,h5py}.py`
into it works the same way.

See https://github.com/proteustoolkit/proteus/wiki/How-to-Build-Proteus for old information on building the entire stack by hand.

# Developer Information

Development happens on

https://github.com/proteustoolkit/proteus

which is where pull requests go, where CI runs, and where the issue tracker,
the wiki and the releases live. `RELEASE.md` describes how a release is cut.
