# Notes for Claude Code sessions

**Scope:** cloud containers and fresh Linux machines. On cekees's workstation and fleet, the workspace `CLAUDE.md`
outside this repository wins (in particular: do not `pip install -e` into its shared conda envs; build in place and
run with `PYTHONPATH=<worktree>`).

## Where changes go
Open pull requests on `cekees/proteus` against `dev` (1.9.x work against `1.9.x`), from a branch cut from `main`.
Never open a pull request on `proteustoolkit/proteus`, never merge `dev` into a branch, and sync with merges, not
rebase or force-push.

## Building proteus from source (cloud container / fresh Linux x86_64)

Verified on `main` at b9d98f5 (2026-11-17): 4 cores, 15 GB RAM, running as root.

1. **Get conda.** `micro.mamba.pm` is blocked by the cloud proxy; GitHub release downloads and `conda.anaconda.org`
   work, so use Miniforge:
   ```bash
   curl -sSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash /tmp/miniforge.sh -b -p $HOME/miniforge3
   source $HOME/miniforge3/etc/profile.d/conda.sh
   ```
2. **Create the env** (a few minutes; includes PETSc, OpenMPI, HDF5, PUMI, Chrono, ymf from git):
   ```bash
   mamba env create -y -f environment-openmpi-dev.yml
   conda activate proteus-dev-openmpi
   ```
3. **Compile in place first, then do the editable install.** In this container a plain `pip install -e .` took more
   than 30 minutes; compiling with `build_ext` first was much faster (`custom_build_ext` in `setup.py` builds the
   extensions concurrently):
   ```bash
   python setup.py build_ext -i
   pip install --no-deps --no-build-isolation -v -e .
   ```
   Always use `--no-build-isolation` (see README). Run long builds in the background with a long timeout and log to a
   file.
4. **Check it:**
   ```bash
   python -c "import proteus, proteus.superluWrappers, proteus.mprans.cRANS2P, proteus.MeshAdaptPUMI.MeshAdapt, proteus.mbd.CouplingFSI"
   ```

## Making the tests pass

- **OpenMPI refuses to run as root** (`prterun has detected an attempt to run as root`), and the cloud container runs
  as root. Every multi-rank test (the `mpiexec -n 2` ones, e.g.
  `test_cube_mms.py::test_two_ranks_give_the_serial_solution`) fails without:
  ```bash
  export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
  ```
  Also allow oversubscription on small machines:
  ```bash
  export PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe OMPI_MCA_rmaps_base_oversubscribe=1
  ```
- For multi-rank comparisons pin the partition: `PETSC_OPTIONS="-mat_partitioning_type current"`.
- Run from `test/`: `python -m pytest -q -n 4 <files>`. Use `pytest-xdist` (`-n`), not `--forked`: OpenMPI is not
  fork-safe and `--forked` gives spurious SIGSEGVs (README).
- With the env vars above, this subset gave 178 passed and 0 failed in about 2 minutes:
  `test_CompQuad.py test_analyticalSolutions.py test_boundaryconditions.py test_adr_problem.py test_cube_mms.py test_bodydynamics.py TADR`.
- **Cleanup:** xdist workers leave untracked output folders `test/gw0`, `test/gw1`, ... Delete them after a run, but
  **not `test/gwvd`**, which is a tracked test directory: use `rm -rf test/gw[0-9]*`, never `test/gw*`.
