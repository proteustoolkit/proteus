<!-- DRAFT release notes for the GitHub release of 1.9.1 (same layout as the 1.9.0 release body).
     Numbers: cekees/proteus#N = fork review PR; proteustoolkit/proteus#N = org PR (trains). Compare range: 1.9.0..1.9.x.
     Add the Zenodo DOI badge line after the release is archived. Delete this comment when publishing. -->

# proteus 1.9.1

Patch release on the `1.9.x` line (155 commits and 244 files changed since 1.9.0). It carries the code behind the
Torino flume, Hybrid CutFEM-IFEM, Richards (JNMA) and variable-density flow/transport (WRR) manuscripts, build fixes
for newer PETSc/pybind11/Cython/pytest, HPC configurations, and parallel-correctness fixes for `parun`.
No 2.0-only redesign work is included.

# Contributors

- Chris Kees @cekees
- Arnob Barua @abarua-ce
- Darsh Nathawani @nathawani

# New features

- **Shallow-water models for the Torino flume paper** (SWE/DSWE): integration-by-parts walls, canopy drag and porosity, tailwater
  condition, parallel fixes; the 1.9.0 drag hack is removed
  ([proteustoolkit/proteus#1332](https://github.com/proteustoolkit/proteus/pull/1332), cekees/proteus#143).
- **Hybrid CutFEM-IFEM, three-phase formulation** in the ADR model, with the triply-cut switches as options and a paper preset,
  pinned three-phase tests, partition-independent cut-cell face terms, P2 in parallel, owned-element L2 error
  ([#1327](https://github.com/proteustoolkit/proteus/pull/1327) = cekees/proteus#144; cekees/proteus#153, #165; train
  [#1334](https://github.com/proteustoolkit/proteus/pull/1334)). Cleanup: debug hooks and unused switches removed, varying-J_s flux
  lift fixed, H1 error reported (cekees/proteus#190, train [#1338](https://github.com/proteustoolkit/proteus/pull/1338)).
  New ADR documentation page and `ADR.Coefficients` docs (cekees/proteus#197, train
  [#1342](https://github.com/proteustoolkit/proteus/pull/1342); Darsh Nathawani).
- **Compatible variable-density flow to transport coupling** (Barua & Kees, section 2.2) (cekees/proteus#150), Henry K = 0.01 and Barua's
  edge-flux transport beside the compatible one (cekees/proteus#174), levee and Henry figure decks (cekees/proteus#169).
- **Richards**: implicit FCT, theta-scheme, L-scheme and increment tests, Hornung-Messing example (cekees/proteus#123, JNMA);
  air-entry van Genuchten closure `VG_AIR_ENTRY` and levee Case B/C on the compatible transport (cekees/proteus#170).
- **HPC**: Cray EX configurations for Carpenter, Barfoot, Blueback and Narwhal (cekees/proteus#26).
- `Context.Options` splits `-C` on any whitespace, commas and semicolons; `gatherTimes` takes the frame count from the `.h5`,
  is a no-op on a complete `.xmf`, and `-t` never duplicates frames (cekees/proteus#31).
- The numerics file's partitioning options now reach legacy p/n meshes (cekees/proteus#55).

# Issues fixed

- **SIGILL on x86_64 hosts without AVX**: AVX support is detected, not assumed (cekees/proteus#15).
- **pybind11 3.x** allowed in `pyproject.toml` (cekees/proteus#17).
- **petsc4py 3.26**: `Mat.createVecLeft` replaces the removed `getVecLeft` alias (cekees/proteus#20).
- **Cython 3**: binds `self` on class-stored compiled functions; `staticmethod` at all sites (cekees/proteus#205, train
  [#1342](https://github.com/proteustoolkit/proteus/pull/1342)).
- **`SUPERLU_DIR`** honoured and duplicated build entries removed (erdc/proteus#1283).
- **`parun`**: exits nonzero, on all ranks, when a system step fails (cekees/proteus#38); hard-coded developer `sys.path` entry removed
  (cekees/proteus#40); `parun -F` scaling, silent numbering corruption and the collective mappings reopen fixed (cekees/proteus#33).
- `argsDict` values that were 1-tuples through trailing commas fixed in twelve places (cekees/proteus#45).
- Build: each extension gets its own object directory (parallel-build race), and a parallel `build_ext` ends when an extension fails
  (cekees/proteus#154).
- `past.utils.old_div` replaced by true division (cekees/proteus#41); Sphinx warnings fixed and the `richards.cADR` extension restored
  (cekees/proteus#43).
- Tests: sibling-import guard and `--import-mode=importlib` in every invocation (cekees/proteus#49); two intermittent failures fixed
  (`test_sedclosure` rounding, `test_porosity` log parsing; cekees/proteus#66); dependent chains made self-contained so one failure
  cannot cascade into a hang ([proteustoolkit/proteus#1300](https://github.com/proteustoolkit/proteus/pull/1300)).

# CI

- Superseded PR runs are cancelled ([#1292](https://github.com/proteustoolkit/proteus/pull/1292)); pytest kept below 9.2 and Open MPI
  `ob1` on aarch64 ([#1301](https://github.com/proteustoolkit/proteus/pull/1301)).
- `sdist.yml` is green on `1.9.x` at b6c60a5d (run 42182060541 on the org, 42183185908 on the fork).
- This is the last release whose CI includes the macOS Intel legs.

# Dependency floors (unchanged since 1.9.0)

The locked environments pin PETSc/petsc4py 3.25.4 (3.26 works with this release); scorec/PUMI >= 4.2.1, pychrono >= 10, xtensor >= 0.27.1.
