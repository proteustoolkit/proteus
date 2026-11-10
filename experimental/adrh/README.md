# ADRH engine prototype

Experimental, not part of the proteus build. A header-only n-component
advection-diffusion-reaction-Hamiltonian element kernel (P1 simplices, Galerkin) with the
coefficient functor as a template parameter, to settle the interface a ymf code generator has to
target. Design: `ADRH_ENGINE_DESIGN.md` (kept beside the campaign notes).

* `adrh.h` – the `Terms<nc,nS>` interface and `elementKernel` (geometry, u and grad u, coefficient
  evaluation and the weak form, residual and Jacobian, all inside the element loop; nothing stored
  per quadrature point). P1 simplices with the affine map, the first version of the kernel.
* `spaces.h`, `engine.h` – Lagrange spaces (`P1Simplex`, `P2Simplex`, `Q1Cube`, in 2 and 3 dimensions) and the
  space-generic `elementKernelSpace`: a space gives basis values and reference gradients at a reference point,
  the map from a face's coordinates to the element's, each face's outward reference normal and vertices; the
  geometry is isoparametric through the same basis at every quadrature point (`PointGeometry`), affine for the
  simplices, trilinear for hexahedra. The masked weak form is the one of `masked.h`.
* `coefficients.h` – hand-written functors in generated form: `Poisson3D` (test/ci/poisson_3d_p)
  and `Coupled2`, a two-component system with every term nonlinear.
* `module.cpp`, `build.sh` – pybind11 module (`sh build.sh` inside the proteus environment).
* `compare_with_python.py [nn]` – checks the kernel against `OneLevelTransport`'s element residual
  and Jacobian on tets, and the coupled system's Jacobian against finite differences.

Measured (Apple M-series, one core): relative difference to the Python path 7e-16 (residual, on
elements with no exterior face: the Python element residual includes the exterior boundary-flux
terms) and 8e-16 (Jacobian); coupled Jacobian vs central differences 1e-10. Element residual and
Jacobian for 162,000 tets: 0.054 s for the kernel including geometry and coefficient evaluation,
against 0.39 s for the Python path's kernels alone with its quadrature arrays already filled.

## Sparsity masks and the advection-diffusion regression

* `masked.h` – `elementKernelMasked`: the same kernel with the functor's sparsity as `static constexpr` predicates
  (`DenseMasks` says everything is nonzero; a functor redefines what it knows). Component loops are unrolled with
  `constexpr_for` and every disallowed block is dropped with `if constexpr`. `compare_masks.py` checks it against the
  runtime-loop kernel and times both: a five-component chain with tridiagonal reaction is 23x faster, a two-component
  coupled system 2.1x, Poisson 1.3x; with all-true masks the unrolled kernel costs the same as the runtime loops, so the gain
  is all sparsity.
* `ad_regression.py` – steady advection-diffusion with the manufactured sine solution
  (`AnalyticalSolutions.LinearADR_Sine`, c = 0) on regular triangles / tets (`AD_SPACE=p1`, `p2`) or quadrilaterals /
  hexahedra (`q1`), in `AD_ND=2` or 3 dimensions: the engine, a plain Proteus model (`ad_sine_p.py`, `ad_sine_n.py`) and the
  exact solution. Engine and Proteus nodal solutions agree to solver round-off (< 5e-14) in every boundary mode, space and
  dimension (42 runs); L2 error rates 2 for P1 and Q1, 3 for P2.

## Boundary conditions (`boundary.h`)

`exteriorFaceKernel` adds the exterior-face terms per face (residual and Jacobian, scattered by the driver), for any
of the spaces (its third template parameter; default P1): segments and triangles of simplices, edges of quadrilaterals,
quadrilaterals of hexahedra, all through the space's own face map and normal. A BC functor says, per component and
boundary quadrature point, which of three kinds applies; each is the engine's version of a Proteus exterior
numerical-flux treatment, and `ad_regression.py` compares against that treatment (`AD_BC=`, in 2D and 3D, for
`AD_SPACE=p1|p2|q1`):

| kind | what it does | Proteus equivalent | `AD_BC` |
|---|---|---|---|
| `Strong` | nothing on the face; the driver replaces the row | `StrongDirichletFactory`, `numericalFluxType=None` | `strong` |
| `Flux` | prescribed diffusive flux, optionally affine in the trace (`gdiff + dgdiff u`); advective flux prescribed where given, else the interior `f(u).n` on outflow and zero on inflow | `StrongDirichletFactory({ci: 'outFlow'})` with `diffusiveFluxBoundaryConditions` / `advectiveFluxBoundaryConditions`; `RobinFactory` for `alpha u + beta q.n = g` | `mixed`, `inflow`, `robin` |
| `WeakDirichlet` | upwinded advective flux against `f(g)`, `-a grad phi . n + (C/h^p)(u - g)`, adjoint `-sigma (u - g) (a^T n).grad w` | `Advection_DiagonalUpwind_Diffusion_{IIPG,NIPG,SIPG}_exterior` (C = 2, 10, 100; sigma = 0, -1, +1) | `iipg`, `nipg`, `sipg` |

The header's comment says where the engine deliberately differs from Proteus (the adjoint's tensor contraction, da/du in
the Jacobian); neither shows with an isotropic constant diffusion tensor.

## Time integration (`ad_dirac_regression.py`)

The element kernel's mass term is backward Euler, `(m(u) - m(u_old)) / dt`, with the coefficients evaluated at the old
state. `ad_dirac_regression.py` steps the transient linear ADR equation `u_t + div(b u - kappa grad u) + c u = 0`
(`TransientLinearADR`, Proteus's `LinearVADR_ConstantCoefficients`) from the Dirac-IC Gaussian
(`AnalyticalSolutions.LinearADR_Decay_DiracIC`, verified in #90) in fixed steps, with the boundary data from the exact
solution at time t through `ExactBC<DiracExact>` in any of the seven `AD_BC` modes, against Proteus's `BackwardEuler` +
`FixedStep` (`ad_dirac_p.py`, `ad_dirac_n.py`) and against the exact solution. `regression_common.py` holds what the two
drivers share. Engine and Proteus agree to round-off (< 1e-13) in every space, dimension and mode; the error against the
exact solution converges at rate 1 in dt and saturates at the time error in h, as backward Euler should.

One Proteus behaviour had to be mimicked for that agreement (`AD_FIRST_STEP=proteus`): Proteus's old mass for the *first*
step is taken from a residual evaluation made after the strongly imposed Dirichlet dofs were already set to their
values at t_1 (`NumericalSolution.calculateSolution`: `initialize_dt_model`, then `getResidual`, then
`stepController.initializeTimeHistory`), so in step one the boundary dofs have no time derivative. The effect is
O(dt^2) and local to the boundary-adjacent rows; later steps are right. The engine's default keeps the true old state.

Design, decisions and numbers: `ADRH_ENGINE_DESIGN.md` (kept beside the campaign notes, sections 11-18).

## Kokkos port (threads and devices)

Decision (cekees, 2026-10-24): the generated kernels get on-node threads and devices in 2.0.0, in Kokkos, with element
coloring for the assembly's scatter (`ADRH_ENGINE_DESIGN.md` section 19, roster C19).

* `portable.h` – `ADRH_FN` is `KOKKOS_INLINE_FUNCTION` when `ADRH_USE_KOKKOS` is defined and `inline` otherwise, so the
  headers are the host prototype's and the device kernel's, one source. Nothing else in the kernel changed.
* `masked.h` – the body of the element loop is now `elementMaskedOne` (one element: coordinates and DOF values in, R and J
  out); `elementKernelMasked` is the loop over it. **It and the compile-time loop helpers are `always_inline`**: left to the
  compiler's heuristics, a helper (`constexpr_for` with its lambdas) stayed a real call whenever the kernel was reached through a member
  function (a Kokkos functor), which pushes every array of the element into memory. The Poisson kernel ran 3.2 times
  slower there (815 against 250 ns per element) than in a plain loop; Kokkos was not the cause. With the fix the host prototype itself is
  faster: masked coupled system 3.4 to 1.3 microseconds per element, masked five-component chain 6.6 to 2.5, the same chain
  with all-true masks 146 to 52 (`compare_masks.py`).
* `kokkos_kernel.h` – `elementKernelKokkos` (one Kokkos iteration per element, R and J stored, the prototype's interface) and
  the fused `assembleColored`: a greedy node-sharing coloring (`colorElements`, 30 colors on the cube's tetrahedra), a block-CSR
  Jacobian pattern (`buildBlockCsr`), and one launch per color in which a thread computes its element and adds it to the global residual and
  Jacobian. Elements of one color share no node, so there are no atomics and the sums happen in the same order every run.
* `bench_kokkos.cpp` – the four functors on a structured tetrahedral mesh (`nn` nodes per edge; 31 is 162,000 tets): host prototype,
  Kokkos element kernel, colored assembly, with the differences and a checksum of the assembled bits.
* `module_kokkos.cpp` – `adrh_kokkos`, the host module's calling convention over the Kokkos kernel (masked kernel only), so the
  existing checks run on it: `ADRH_MODULE=adrh_kokkos python compare_with_python.py | ad_regression.py`; `golden.py` compares
  any two builds (`golden.py against adrh_kokkos`; `golden.py save|compare` brackets a change to the host kernel).
* `build_kokkos.sh` – CPU build in a conda environment with `kokkos`; the CUDA recipe is in the design note.
* `CMakeLists.txt` – the same modules as a CMake project, with the tests as `ctest` (`cmake -S . -B build && cmake --build build && ctest --test-dir build`; CUDA/HIP Kokkos: configure with its compiler, see the design note). `.github/workflows/adrh.yml` runs it on linux-64, linux-aarch64 and macOS arm64 in a conda environment of `cmake compilers pybind11 numpy scipy kokkos`; the tests import `proteus.Quadrature` from the source tree, which needs no compiled proteus.
* `build_kokkos_cuda.sh ARCH` – Kokkos 5.2.2 from source (OpenMP + CUDA, `ARCH` = TURING75, AMPERE80, ADA89, HOPPER90 ...) and these modules and tests against it, in a conda environment of `cmake ninja gxx_linux-64=14 pybind11 numpy scipy cuda-nvcc=12.9 cuda-cudart-dev cuda-driver-dev cuda-version=12.9` (about 2 min on a many-core machine, all 8 tests pass on an RTX 6000 Ada). The `gpu` job in `adrh.yml` runs it nightly on a GPU runner once the organization has one (set the Actions variable `ADRH_GPU_RUNNER` to its label; `ADRH_GPU_ARCH` if it is not a T4); a `kokkos_default_space` test fails the job if the kernels ran on the host instead of the device. conda-forge's CUDA Kokkos packages are not used: conda cannot solve them (they need the `__cuda_arch` virtual package) and there are no sm86/sm89 builds.

Results (`reports/adrh_kokkos_2026-10-24/` beside the campaign notes; nn = 31, ns per element, element kernel / colored assembly):

| | Poisson | AD sine | coupled 2 | chain 5 |
|---|---|---|---|---|
| Mac M4 Max, host prototype, 1 thread | 238 | 277 | 1,322 | 2,408 |
| Mac, Kokkos OpenMP, 1 thread | 230 / 251 | 273 / 291 | 1,313 / 1,448 | 2,404 / 2,863 |
| Mac, Kokkos OpenMP, 12 threads | 21 / 35 | 24 / 37 | 118 / 136 | 213 / 272 |
| ceds-model Zen 4, host prototype, 1 thread (g++ -O3) | 578 | 554 | 3,274 | 5,382 |
| ceds-model, Kokkos OpenMP, 96 threads (g++ -O3) | 11 / 16 | 10 / 16 | 52 / 60 | 90 / 105 |
| ceds-model, Kokkos OpenMP, 96 threads (-march=native), nn = 61 | 5.7 / 8.0 | 8.3 / 10.4 | 31 / 34 | 52 / 64 |
| ceds-model, one RTX 6000 Ada, nn = 61 | 8.8 / 10.8 | 8.3 / 10.3 | 41 / 53 | 158 / 200 |

* **The port is the prototype.** The Kokkos element kernel reproduces the host `elementKernelMasked` bit for bit on the CPU, for all
  four functors. `adrh_kokkos` against `adrh_proto` (`golden.py against adrh_kokkos`): 14 of 18 arrays bitwise equal, the other
  four within 3.1e-16: the unmasked runtime-loop kernel, which the Kokkos module aliases to the masked one, and the 2-D
  advection-diffusion arrays, which the host module now computes with the spaces kernel (`engine.h`) rather than the masked
  P1 kernel. Against Proteus: element residual and Jacobian 8.6e-16 and 6.0e-16 (`compare_with_python.py`); the advection-diffusion
  regression (`ADRH_MODULE=adrh_kokkos AD_SPACE=p1 python ad_regression.py`) in 2D and 3D and all seven boundary treatments agrees with the
  Proteus model to 1.2e-14 or better, with the L2 rates of the host module (1.85 to 2.00).
* **Scope.** The Kokkos port is of the P1 simplex kernel (`masked.h`). The Lagrange-space kernel (`engine.h`, P2 and Q1,
  isoparametric geometry at every point) and the transient driver are not ported; the module raises `NotImplementedError` for
  `AD_SPACE=p2|q1`. Its per-element body would be extracted the same way (`elementMaskedOne` is the template).
* **Colored assembly is the same bits at every thread count**: 1, 4, 12, 16 on the Mac; 1, 24, 96 on ceds-model; repeated GPU runs
  and both GPUs. It matches a serial scatter to 7e-16 (summation order).
* **Scaling is near-linear in cores** for the element kernel: 11 to 11.6x at 12 threads (the M4 Max has 12 performance cores;
  16 threads, adding the efficiency cores, does not help), 52 to 63x at 96 threads on the Zen 4 node. The fused scatter costs
  7 to 19% more on one thread and 14 to 66% more on 12 to 96 threads, most for the small kernels, where the 30 launches and the
  scattered writes are a larger share.
* **The GPU is not faster than the CPU node here.** One RTX 6000 Ada matches 96 Zen 4 cores on Poisson and the coupled system and is 3x
  slower on the five-component chain (the per-thread arrays are large). That card's double precision rate is a small fraction of its
  single precision rate (spec sheet, not measured here); it is a functional test platform for the device path, not a guide to a data-center GPU.
* **Bitwise results across platforms depend on fused multiply-add.** With contraction on, the CPU native build, the CPU generic
  build and the GPU give three different answers. With it off (`-ffp-contract=off`, `nvcc --fmad=false`) Apple arm64, x86-64 scalar and AVX2,
  and the GPU give **identical bits for the Poisson and five-component chain kernels** (residual and Jacobian), and identical Jacobians for
  all four. The residuals of the two kernels that call `sin` differ in the last bit between the C libraries and CUDA's. So
  `ymf_run --check`-style reproducibility needs the generated kernels built with contraction off, and either no transcendental functions
  in the reproducible path or one implementation of them.

## Building the CUDA benchmark

```
nvcc_wrapper -O3 -std=c++20 -arch=sm_89 -fopenmp --extended-lambda --expt-relaxed-constexpr \
  -I$KOKKOS/include bench_kokkos.cpp -o bench_kokkos_cuda -L$KOKKOS/lib -lkokkoscore -lkokkossimd -lhwloc -lcuda
```

with Kokkos built from source for the card (`-DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_ADA89=ON`,
17 seconds): conda-forge's CUDA builds of Kokkos have no sm86 or sm89. The headers compile unchanged under nvcc 13.4; no
device-specific code was added.

## Stage-1 device assembly, measured end to end (`ad_solve_device.py`)

Roster C19 decision (6): assemble on the device and solve on the host, and see what the copies cost. `adrh_kokkos.AdSineAssembler3D`
holds a tetrahedral mesh in the Kokkos execution space (coloring and block-CSR pattern built once on the host) and returns the
global residual and the CSR Jacobian of the 3-D steady advection-diffusion problem, timing the host-to-device copy of the nodal
vector, the assembly (best of 5) and the device-to-host copy. `ad_solve_device.py` imposes the Dirichlet rows on the host and solves
with PETSc (GMRES, GAMG, rtol 1e-10, one rank), against the manufactured solution of `ad_regression.py`. `--check` compares the
device system with the host engine's element kernel and a numpy scatter (residual 2.6e-16, Jacobian 5.6e-16 at nn=9, 3.7e-16 at nn=17); the maximum
error against the exact solution reproduces the regression's (6.240e-02 at nn=9, 1.575e-02 at nn=17) and is the same on the CPU and the GPU.

`ceds-model`, seconds, 3-D cube split into Kuhn tetrahedra, one PETSc rank for the host side. The first three rows ran with the machine
quiet (load about 2); the GPU row was repeated with one host thread (`KOKKOS_NUM_THREADS=1`) at load 25 (see below). `host` is Dirichlet rows, PETSc matrix
creation, GAMG setup and solve.

| tets | assembly on | assemble | copies | host | total | vs 1 thread |
|---|---|---|---|---|---|---|
| 1.3 M | 1 thread | 0.792 | 0.001 | 0.876 | 1.669 | 1.00x |
| | 12 threads | 0.064 | 0.000 | 0.944 | 1.008 | 1.66x |
| | 96 threads | 0.016 | 0.000 | 0.995 | 1.011 | 1.65x |
| | one RTX 6000 Ada | 0.014 | 0.002 | 0.904 | 0.919 | 1.82x |
| 6.0 M | 1 thread | 3.865 | 0.007 | 4.770 | 8.642 | 1.00x |
| | 12 threads | 0.505 | 0.002 | 4.959 | 5.465 | 1.58x |
| | 96 threads | 0.085 | 0.001 | 5.125 | 5.211 | 1.66x |
| | one RTX 6000 Ada | 0.053 | 0.006 | 4.852 | 4.911 | 1.76x |

* **The copies are not the cost.** The 15.2 M-entry matrix and the residual come back in 5.4 ms (about 23 GB/s); sending the nodal vector takes 0.4 ms.
  On the device side they are 10% of the time, and about 0.1% of the total.
* **The ceiling is the host side, as the baseline predicted.** Assembly is 45% of a one-thread solve at 6 M tets (the baseline's 19 to 51%). Taking it to zero bounds the speedup at
  8.64 / 4.77 = 1.81x; the GPU reaches 1.76x and 96 CPU threads 1.66x. **Device assembly with a host solve is not better than 96 threads of
  Kokkos assembly by more than 6%**, although the GPU is 1.4x faster on assembly alone (0.053 + 0.006 against 0.085 s). On this card (weak double precision) the GPU is not
  where the gain is; a data-center GPU would assemble faster but cannot move the ceiling.
* **What dominates afterwards is host work that is not the solver's arithmetic**: GAMG setup (2.7 s) and solve (2.0 s), and 0.17 s of Dirichlet rows plus
  copying the CSR into PETSc, which is 30 times the device transfer. Wrapping the arrays without a copy, and imposing the rows on the device, are cheap wins but
  worth 3.5% of the total.
* **Verdict for decision (6): stage 1 is not worth building as a product.** It is a measurement that bounds the gain at 1.8x and ties a threaded CPU. The device
  pays only when the matrix stays on the device: stage 2 (PETSc CUDA vectors and matrices, `MatSetValuesCOO` fed with the device values, a device
  preconditioner), which needs a CUDA PETSc.
* **Caveat, and it matters for a node.** This is one rank. A node with 96 cores and 2 GPUs running a rank per GPU leaves 94 cores idle for the solve; the CPU path
  uses all 96 for both. Stage 1 therefore compares against the threaded assembly only when the solve is also threaded or the host has cores to spare per GPU.

**A trap in the Python module with a device build.** Kokkos initializes its OpenMP host space with every hardware thread by default, next to CUDA. Those idle threads
slowed the single-core PETSc solve on the same machine by 45% (2.63 s against 1.83 s at 6 M tets, 0.54 against 0.31 s at 1.3 M): `KOKKOS_NUM_THREADS=1` restores it (0.315 s).
A threaded CPU build also slowed the host solve by 8 to 21% (0.349 and 0.374 s against 0.309 s at 1.3 M tets; 1.98 and 2.10 against 1.83 s at 6 M), so the 12- and 96-thread rows above carry that;
`OMP_WAIT_POLICY=passive` trimmed the 96-thread solve from 0.370 to 0.349 s at 1.3 M tets (load 25). Set both in any run that mixes a threaded Kokkos phase with a serial host phase.

Building the Python module for CUDA needs Kokkos as shared libraries (`-DBUILD_SHARED_LIBS=ON`; the static install cannot be linked into an extension) and the same
nvcc flags as the benchmark plus `-shared -Xcompiler -fPIC` with pybind11's and Python's include paths; at run time `libcudart` must be on `LD_LIBRARY_PATH`.
Raw results: `reports/adrh_kokkos_solve_2026-10-25/` beside the campaign notes.

## Whole solve at a fixed core count: ranks against threads (`ad_solve_mpi.py`)

Roster C19 decision (3). The same problem, distributed: each MPI rank owns a brick of the Kuhn-tetrahedra mesh, assembles it with the
Kokkos colored kernel on `T` threads, and sends the matrix to PETSc through `MatSetValuesCOO` (ghost rows are added into their owners) and the
residual through `VecSetValues`; PETSc then solves with GMRES + GAMG on all `R` ranks. `R x T` is held at the 96 cores of `ceds-model`
(`mpiexec -n R -bind-to core:T`, `KOKKOS_NUM_THREADS=T`, `OMP_NUM_THREADS=T`, `OMP_PROC_BIND=spread OMP_PLACES=cores`), the machine idle (the campaign script waits on measured CPU
use between runs), best of two runs of each layout (the two agree within 1 to 3%), assembly best of 5. Every layout reproduces the serial answer: max error 4.051e-04
(6 M tets) and 1.583e-04 (24.6 M), 24 or 25 GMRES iterations. Seconds, maximum over ranks:

| tets | R x T | assemble | matrix exchange | GAMG setup | solve | total | vs 96 x 1 |
|---|---|---|---|---|---|---|---|
| 6.0 M | 96 x 1 | 0.054 | 0.037 | 0.125 | 0.062 | 0.279 | 1.00x |
| | 48 x 2 | 0.054 | 0.037 | 0.162 | 0.080 | 0.334 | 1.20x |
| | 24 x 4 | 0.053 | 0.043 | 0.237 | 0.107 | 0.442 | 1.58x |
| | 12 x 8 | 0.053 | 0.066 | 0.368 | 0.164 | 0.653 | 2.34x |
| | 4 x 24 | 0.054 | 0.249 | 0.899 | 0.517 | 1.725 | 6.2x |
| | 1 x 96 | 0.065 | 0.419 | 2.751 | 2.096 | 5.344 | 19.2x |
| 24.6 M | 96 x 1 | 0.254 | 0.132 | 0.473 | 0.611 | 1.477 | 1.00x |
| | 48 x 2 | 0.249 | 0.143 | 0.581 | 0.644 | 1.625 | 1.10x |
| | 24 x 4 | 0.253 | 0.207 | 0.850 | 0.736 | 2.053 | 1.39x |
| | 12 x 8 | 0.255 | 0.303 | 1.529 | 0.952 | 3.051 | 2.07x |
| | 4 x 24 | 0.261 | 0.640 | 4.072 | 2.299 | 7.297 | 4.9x |
| | 1 x 96 | 0.259 | 1.112 | 11.122 | 8.154 | 20.697 | 14.0x |

* **Threads do as well as ranks for the assembly and for nothing else.** Assembly time does not depend on the layout (0.054 s at 6 M tets from 96 x 1 to 4 x 24; the 1 x 96 layout is 20% slower), as in
  the assembly benchmark. Everything that follows it is one core per rank: the matrix exchange (building and inserting the COO arrays), the GAMG setup and the
  solve. With `T` threads per rank the other `T - 1` cores idle through them, so the total grows with `T`. At the layouts in the table the assembly is 19% of the total at 96 x 1, 3% at 4 x 24 and 1.2% at 1 x 96.
* **For this solver stack the answer to decision (3) is a rank per core.** The cost of `T = 2` is 10 to 20%, of `T = 4` 39 to 58%, of `T = 8` about a factor 2. This is a statement about PETSc's MPI-only
  solvers and a serial matrix exchange. It would change with a threaded or device-resident solve, which is the case for stage 2, not for threads on a CPU node.
* **Memory is not a reason, on this evidence.** Total resident memory falls from 13.9 GB (96 x 1) to 1.8 GB (1 x 96) at 6 M tets, but 7.8 GB of the 13.9 is the 81 MB each rank carries after importing
  Python, numpy, PETSc and Kokkos. The ghost layer is 11.3% of the nodes at 96 ranks (7.0% at 24.6 M tets); the rest of the difference is driver temporaries.
  A production run's per-rank state is larger than this driver's, so the halo and duplicated-setup argument is worth measuring on a real model before it is made.
* **GAMG iteration counts do not depend on the layout** (24 to 25), so the solve times are rates, not artefacts of a worse preconditioner.

Raw results and the campaign script: `reports/adrh_kokkos_ranks_2026-10-26/` beside the campaign notes.

## The Lagrange-space kernel under Kokkos (`engine.h`, P1 and P2 simplices, Q1 cubes)

The spaces kernel is ported the way the P1 kernel was: the body of the element loop is `elementSpaceOne` (`engine.h`), `elementKernelSpace` is the loop over it, `spaces.h`
is `ADRH_FN` throughout (the P2 edge table moved into the function: a device cannot index a host-side `static constexpr` array), and `kokkos_kernel.h` takes an element
policy (`P1Element`, `SpaceElement`) so that one launch path, one colored assembly and one block-CSR builder serve both: `elementKernelKokkosSpace` and
`assembleColoredSpace` beside the P1 `elementKernelKokkos` and `assembleColored`. `bench_kokkos.cpp` builds P2 tetrahedra (edge midpoints shared between elements, proteus's
numbering) and Q1 hexahedra and runs `ad_sine` and `coupled2` on them; `adrh_kokkos.advection_diffusion_sine(nd, space, ...)` takes `p1` (the affine kernel), `p2` and `q1`, so
`ADRH_MODULE=adrh_kokkos AD_SPACE=p2 python ad_regression.py` runs the whole Proteus regression on it. The exterior-face kernels and the transient driver were ported next (the section below).

* **Correct.** The Kokkos kernel is bitwise the host `elementKernelSpace` on the CPU in every case (R and J, 5 cases, 3 thread counts); `golden.py against adrh_kokkos`: all 12 arrays of P2 and Q1,
  in 2 and 3 dimensions, bitwise equal. The regression against the Proteus model, every dimension, space (p1, p2, q1) and the seven boundary treatments, nn = 5, 9, 17: worst
  difference 1.2e-14 (2D P1), 9.7e-15 (2D P2), 2.3e-15 (2D Q1), 7.1e-15 (3D P1), 5.9e-14 (3D P2, Robin; the host engine gives the same digits), 1.2e-14 (3D Q1).
* **Same bits at every thread count** (1, 4, 12 on the Mac; 96 on `ceds-model` agrees with the serial scatter to 5e-16). Hexahedra colour with 8 colours (by cell parity); P2 tetrahedra with the same 30 as P1, since two elements that
  share an edge DOF share its vertices.
* **Speed**, ns per element, element kernel / colored assembly, nn = 31 (162,000 tets, 27,000 hexes):

| kernel | Mac 12 threads | host, 1 core of `ceds-model` | `ceds-model` 96 threads | one RTX 6000 Ada |
|---|---|---|---|---|
| ad_sine P1 (space kernel) | 40 / 53 | 485 | 10 / 16 | 8 / 28 |
| ad_sine P2 | 122 / 174 | 2198 | 46 / 56 | 50 / 172 |
| ad_sine Q1 hex | 164 / 183 | 3406 | 63 / 70 | 99 / 400 |
| coupled2 P2 (mass) | 806 / 905 | 9792 | 199 / 241 | 256 / 695 |
| coupled2 Q1 hex (mass) | 913 / 949 | 9874 | 241 / 234 | 393 / 1476 |

  The element kernel scales 11 to 11.7x on 12 M4 threads and 41 to 54x on 96 Zen 4 threads. At 1.3 M tets (nn = 61) the GPU element kernel is 48 ns for P2 and 74 ns for Q1 and the colored assembly 69 and 97 ns, close to the 96-thread CPU (45 and 63; 66 and 68).
  **The GPU's colored assembly is 2.7 to 4.1 times its element kernel at 162,000 elements** (the 30 or 8 launches are too small to fill the card: 5,400 tets or 3,400 hexes per colour) and
  closes to 1.2 to 1.5x at 1.3 M tets.
* **Across platforms**, as for P1: with contraction on, the GPU differs from the host by up to 2.7e-15 (P2) in the element kernel; with it off (`-ffp-contract=off`, `nvcc --fmad=false`) the **Jacobians are bitwise equal CPU to GPU in all five new
  cases**, and the residuals differ only through `sin` in the coefficient.
* **Compiler.** nvcc 13.4 accepts the spaces headers unchanged. Forcing the geometry helpers (`PointGeometry::compute`, the basis functions) to inline made P2 and Q1 about 6% faster in both
  the host loop and the Kokkos kernel and put them at parity; the P1 case through the general kernel is slower in both (292 to 461 ns on the Mac) and is not a production path, since P1 has the affine kernel.

## Exterior faces and the transient driver under Kokkos

The last host-only pieces. `boundary.h`: the loop body is `exteriorFaceOne` (always inline), `exteriorFaceKernel` is the loop over it, and the BC functors (`SineBC`, `DiracExact`, `ExactBC`,
`FaceBC`) and `TransientLinearADR` are `ADRH_FN`. `kokkos_kernel.h`: `exteriorFaceKernelKokkos` (one thread per face, per-face outputs; the BC functor is copied to the device with the launch, so
the per-face kind array it points to must live there) and a **colored face pass** for the fused assembly, `ColoredFaces` and `assembleFacesColored`. Faces of elements of one color share no node, but a
corner element can own several exterior faces, so a color runs in passes, the r-th face of every element in pass r (60 passes for the cube's tetrahedra: 30 colors, at most 2 faces per element; 24 for hexahedra: 8 colors,
3 faces); the pass adds to the system `assembleColoredSpace` built. The transient driver needs nothing new in the element kernel (the mass term and the old-time values were already there, with `Old` a compile-time choice); the module gets the
host's entry points: `advection_diffusion_sine_faces`, `transient_adr` and `transient_adr_faces`, in any space, so `ADRH_MODULE=adrh_kokkos python ad_regression.py` and `ad_dirac_regression.py` run entirely on Kokkos kernels.

* **Bitwise the host kernel.** Face kernel, transient element kernel and transient faces: all 36 new arrays in `golden.py against adrh_kokkos` (P1, P2, Q1; 2D and 3D; the four face kinds) bit for bit. The host module is bit-identical to its saved
  output on all 64 golden arrays.
* **Proteus regressions on Kokkos kernels**, nn = 5, 9, 17 (steady) and 5, 9 (transient), 2 and 3 dimensions, p1/p2/q1, seven boundary treatments (84 cases each): steady worst 5.9e-14 (as before); **transient worst 1.2e-13**
  (3D P2 Robin, 1.1e-13 for 3D Q1; the host engine gives the same digits) with `AD_FIRST_STEP=proteus` in the strongly imposed modes, as the transient driver requires; without it the strong modes differ from Proteus by 1e-6 to 7e-6, the
  first-step behaviour documented above.
* **Same bits at every thread count** for the colored face scatter (1, 4, 12 on the Mac).
* **Speed**, ns per face, face kernel / colored face pass, nn = 31 (10,800 tetrahedron faces, 5,400 hexahedron faces):

| kernel | Mac 12 threads | host, 1 core of `ceds-model` | `ceds-model` 96 threads | one RTX 6000 Ada |
|---|---|---|---|---|
| faces P1 | 13 / 154 | 109 | 4.8 / 81 | 3.3 / 112 |
| faces P2 | 22 / 187 | 282 | 9.4 / 131 | 13 / 437 |
| faces Q1 hex | 39 / 209 | 820 | 24 / 113 | 52 / 799 |

  The face kernel scales 7.7 to 9.4x on 12 M4 threads and 23 to 34x on 96 Zen 4 threads. **The colored face pass is dominated by launch overhead**: 24 to 62 small passes (180 to 600 faces each) cost 5 to 34 times
  the face kernel at this size, and still 8 to 15 times on the GPU at 43,200 faces. Surface terms are a small share of an assembly, but the pass count should shrink: coloring the faces directly
  instead of inheriting the element colors would cut 60 passes to the number of colors of the boundary-element graph.
* **Across platforms.** With multiply-add fusing on, the GPU differs from the host by up to 5.3e-15 over all 62 arrays. With it off, the **face Jacobians are bitwise equal CPU to GPU** in all three spaces (and
  the element Jacobians as before); the residuals differ only through `sin` and `cos` in the boundary data. **A trap this found:** the face Jacobians were *not* equal at first, by 4e-16, because the penalty is `C / pow(h, p)` and
  CUDA's `pow` does not return `h` exactly for `p = 1`; taking the power-one case out (`pow(h, 1)` is `h` in a conforming libm, so the host is unchanged) fixed it. Any libm call in the Jacobian is a cross-platform
  bit difference; in the residual, the boundary data's `sin` and `cos` are the only ones left.

## Richards' equation (`richards.h`, `richards_regression.py`, `richards_layers.py`)

A nonlinear, strongly heterogeneous transport problem for the engine: Richards' equation in pressure-head form with the Mualem-van Genuchten
relations, `m(psi)_t + div(f(psi) - a(psi) grad psi) = 0`, `m = rho theta(psi)`, `a = rho K_r(psi) K_s`, `f = (rho^2/rho0) K_r(psi) K_s g`. `richards.h` ports
proteus's `vgm_wetting` (`proteus/pskRelations.h`) and the coefficients of `proteus/richards/Richards.h` into the engine's `Terms`;
`RichardsMaterials` holds up to 8 parameter sets and a material number per element and reaches the kernels through the new opt-in `bindElement(eN)` hook
(`engine.h`, `boundary.h`): a coefficient functor that defines it is asked for the functor to use on element `eN`, and one that does not is used as it is, so
the other functors pay nothing. Segment (1D) spaces were added to `spaces.h` for the decks' one-dimensional columns (a face of a segment is a point; as proteus's mesh gives
it, its diameter is 1). `FaceBC::advInterior` selects the weak Dirichlet variant of `Richards_IIPG_exterior`: the advective flux from the interior state, no upwinding against the
boundary value.

* **Against proteus's richards model.** `richards_regression.py` runs `test/richards/test_1` (the Celia et al. 1990 column, 1000 backward Euler steps, 40 elements) and
  compares with the profiles proteus saved for its Galerkin scheme (`STABILIZATION_TYPE=0`). The deck's boundary condition is **weak** (`Richards_IIPG_exterior`, penalty 100): with the Dirichlet
  rows replaced instead (`--bc strong`) the fronts differ by 0.06 to 0.08 m of pressure head. With the weak treatment and proteus's own Newton stopping rule (2-norm of the residual below
  1e-8) the engine reproduces the saved profiles to **2.5e-13** at 12 h and 24 h, on the host and the Kokkos modules alike; converged to 1e-12 it differs from them by 8.7e-8 and 4.2e-6, which is
  proteus's Newton tolerance error at the dry nodes (a tiny capacity there), not the discretization. 3.0 Newton iterations per step.
* **Heterogeneity, against no finite elements.** `richards_layers.py`: a two-layer column (a sand over a soil with `K_s` 160 times smaller, different `alpha` and `n`), steady, strong
  Dirichlet rows. Steady 1D flow has a constant flux, so each layer obeys an ODE, `psi' = -Q/K(psi) - 1`, with `psi` continuous at the interface, and the flux `Q` that reaches the top value
  follows by shooting; the ODE is integrated to 1e-12. The engine's nodal error falls at **exactly the P1 rate, 2.00, at every refinement from 11 to 321 nodes** (7.2e-4 to 7.1e-7), identically on the host and Kokkos modules.
* **Not covered here.** The other two decks (HYDRUS, Szymkiewicz) need a stabilized scheme and, for Szymkiewicz, the Brooks-Corey closure: see the section on the K-upwinded scheme below. Faces through `bindElement` (a layered boundary element) are implemented but not exercised by a test.

**Richards as a kernel benchmark** (`bench_kokkos.cpp`: unsaturated, backward Euler mass term, four materials in stripes by element number, so `bindElement` runs on every element of every backend with no data pointer): the van Genuchten-Mualem law
calls `pow` and `sqrt` at every quadrature point, so it costs about three times the advection-diffusion kernel per element on one core (P1: 1,490 ns against 485 for `ad_sine`) and the GPU's element kernel is 1.2 times faster than the 96-core node on P1 and level with it on P2 (its colored assembly is behind the node at 162,000 elements and level at 1.3 M, where the launches fill the card).
ns per element, element kernel / colored assembly, `ceds-model` (one machine, a few percent of the cores busy with another session's job during these runs):

| kernel | M4, 12 threads | host, 1 core | 96 threads, nn = 31 | RTX 6000 Ada, nn = 31 | 96 threads, nn = 61 | RTX 6000 Ada, nn = 61 |
|---|---|---|---|---|---|---|
| Richards P1 | 185 / 226 | 1,490 | 30.5 / 36.2 | 24.7 / 81.7 | 28.8 / 32.2 | 24.5 / 28.8 |
| Richards P2 | 497 / 726 | 4,138 | 86.4 / 101.1 | 74.5 / 252.6 | 69.7 / 83.5 | 68.8 / 109.8 |

The Kokkos kernel is bitwise the host's on the CPU for both; the GPU differs from the host by up to 2e-14 (multiply-add fusing and the device `pow`). Unlike the advection-diffusion kernels, **with contraction off the
Richards residual and Jacobian are not bitwise equal CPU to GPU** (`pow` and `sqrt` of the retention curve: any libm call in a Jacobian is a cross-platform bit difference, as found for the face penalty).
The Celia column runs on the GPU module as well and matches proteus's saved profiles to 3.3e-13, and the layered column reproduces the CPU errors exactly.

## Algebraic stabilization (`lowOrder.h`, `stabilization_test.py`, `stabilization_timing.py`)

Consistent-mass Galerkin is not monotone: a sharp front overshoots and undershoots, which is why proteus runs its HYDRUS and Szymkiewicz Richards decks only with its low-order (`STABILIZATION_TYPE=2`) or FCT scheme.
`lowOrder.h` adds the algebraic cure that needs only the assembled matrix, so it works for any coefficient functor (**but see the section on the K-upwinded scheme: on a strongly nonlinear conductivity this operator cannot carry a front into dry soil**): a **lumped mass** (`M_L,i (m(u_i) - m(u_old,i)) / dt`, `M_L,i` the integral
of the basis function; `lumpedMassKernel`, with `bindElement` for materials) and **discrete upwinding of the spatial operator**: for each pair, the symmetric artificial diffusion `d_ij = max(0, J_ij, J_ji)`
(`J` the Jacobian of the spatial residual), `r_i += sum_j d_ij (u_i - u_j)`, `J_ij -= d_ij`, `J_ii += sum_j d_ij`. The result has no positive off-diagonal, so with the lumped mass it is an M-matrix and satisfies a
discrete maximum principle. It is the low-order operator of FCT and monolithic limiting: the antidiffusive correction a later limiter adds is the difference to the Galerkin operator over this same pattern and
transpose map (`buildTransposeMap`). Row-parallel on the execution space, in two passes (`d`, then the rows), each writing only its own row: no race, the result does not depend on the thread count. `D` is
held fixed in the Jacobian (an approximate Newton step: it converges, not quadratically). One component, P1 and Q1.

* **The pass is exact.** Against a scipy implementation of the formula on the Jacobian of an advection-dominated 3D problem: 1.5e-16 (Jacobian) and 3.4e-16 (residual); 4,128 positive off-diagonals become 0; row sums change by 1e-16;
  a constant vector's artificial residual is exactly 0. Host, CPU Kokkos and GPU give the same numbers.
* **A sharp front** (`stabilization_test.py`: `u_t + u_x - 2e-4 u_xx = 0`, step data, cell Peclet number 25, CFL 0.2, backward Euler): consistent-mass Galerkin overshoots to 1.0052 (and undershoots to -1.3e-8); lumping alone is worse, 1.0659;
  **lumped mass with discrete upwinding stays inside [0, 1] exactly**. It pays in accuracy: the L1 error against the exact erfc solution is 3.9e-2 against Galerkin's 1.4e-2, the smearing of a first-order monotone scheme. Mass is conserved
  to the same digits (0.495000 against 0.495349).
* **Richards, the Celia column** (`richards_regression.py --stabilize`): the pressure head stays exactly inside the data's bounds [-10, -0.75] at 12 h and 24 h, where plain Galerkin undershoots to -11.6 and -10.8. The front (psi = -5 m):

| | 12 h | 24 h | lowest psi |
|---|---|---|---|
| engine, Galerkin (= proteus, 2.5e-13) | 0.6218 | 0.4373 | -11.60 |
| **engine, lumped mass + upwinding** | 0.6184 | 0.4324 | -10.00 |
| proteus, FCT | 0.6147 | 0.4219 | -10.00 |
| proteus, `STABILIZATION_TYPE=2` (the K-upwinded low-order scheme, below) | 0.5505 | 0.3565 | -10.00 |

  The algebraic scheme's front is within 0.15 and 0.4 of an element (0.025 m) of proteus's FCT front and lags less than proteus's `STABILIZATION_TYPE=2` scheme; 3.5 Newton iterations per step. The profiles are not compared digit for digit
  (different schemes). The HYDRUS and Szymkiewicz decks show where this operator fails: see below.
* **Cost** (`stabilization_timing.py`, 3D tetrahedra, the pass against the colored assembly that produces its input, `ceds-model`):

| rows | 1 thread | 96 threads | RTX 6000 Ada | transpose map (once, host) |
|---|---|---|---|---|
| 227 k | 7.0 ms (assembly 819 ms) | 0.3 ms (13 ms) | 0.6 ms (13.5 ms) | 9 to 19 ms |
| 1.03 M | 79 ms (3.9 s) | 14.6 ms (65 ms) | 2.3 ms (53 ms) | 41 to 85 ms |

  Two to four percent of the assembly on one thread and on the GPU. At 96 threads it is 23% at 1.03 M rows: the pass is a memory-bound gather over the CSR and scales 5x, not 50x, on the node.


## The FCT limiter (`lowOrder.h`, `fct_limit`; `stabilization_test.py`, `richards_regression.py --fct`)

The lumped-mass/upwinded scheme is monotone and first-order, so it smears a front (10-90% width 0.154 against the exact 0.031). Flux-corrected transport restores the accuracy the low-order scheme gave up while keeping its bounds.
`fctLimit` is the linearized (Zalesak) limiter over the pattern and transpose map `buildTransposeMap` already provides:

* The antidiffusive edge flux is `f_ij = Mc_ij (dm_j - dm_i)/dt + d_ij (v_j - v_i)` with `Mc` the consistent mass, `dm = m_L - m_old` the change of the **conserved variable** `m` over the low-order step, `d_ij` the dissipation of the
  upwinding pass and `v` the variable it acts on; `f_ji = -f_ij` exactly, so what leaves a node arrives at its neighbour. For a transport equation `m = v = u`. For Richards `m` is the moisture content and `v` the pressure head:
  the driver limits `theta`, then inverts it with the analytic van Genuchten inverse (`Column.fct_correct`), so mass in the sense of the PDE is what is conserved.
* The bounds of node `i` are the extrema of `m_L` and `m_old` over `i` and its neighbours; `R+_i`, `R-_i` are the ratios of the room that is available to the antidiffusion that is asked for, and the edge gets
  `alpha_ij = min(R+_i, R-_j)` (or the other pair, by sign), the same `alpha` on both sides of the edge. Three row-parallel passes (fluxes, bounds, apply), each writing only its own row: no race, the result does not
  depend on the thread count. `consistent_mass` supplies `Mc` (`massMatrixOne`, any space); `dissipation_values` the `d_ij`. One component; nc > 1 and P2 are not done.

* **Against a numpy implementation of Zalesak's formula**, random bounded data on 343 nodes and 1,854 edges: 1.1e-16 (max abs); the local bounds are respected to 1e-17; the conserved sum `M_L m` changes by 3e-18 (relative).
  Host, CPU Kokkos and the RTX 6000 Ada agree.
* **The sharp front** (the `stabilization_test.py` problem; width = distance between the 10% and 90% levels):

| scheme | min u | max u | L1 error | width |
|---|---|---|---|---|
| Galerkin, consistent mass | -1.3e-8 | 1.0052 | 1.39e-2 | 0.0687 |
| lumped mass only | 0 | 1.0659 | 2.13e-2 | 0.0761 |
| lumped mass + upwinding | 0 | 1.0000 | 3.94e-2 | 0.1537 |
| **FCT** | **0** | **1.0000** | **1.57e-2** | **0.0747** |

  (exact width 0.0310). The limiter halves the width of the low-order front, keeps the profile in [0, 1] exactly and has a smaller L1 error than the lumped-mass Galerkin scheme; the mass is that of the low-order scheme (0.495000).
* **Richards, the Celia column** (`richards_regression.py --fct`): the pressure head stays exactly in [-10, -0.75] at 3.74 Newton iterations per step. Position of psi = -5 m:

| | 12 h | 24 h | lowest psi |
|---|---|---|---|
| engine, Galerkin (= proteus, 2.5e-13) | 0.6218 | 0.4373 | -11.60 |
| engine, lumped mass + upwinding | 0.6184 | 0.4324 | -10.00 |
| **engine, FCT** | 0.6188 | 0.4339 | -10.00 |
| proteus, FCT | 0.6147 | 0.4219 | -10.00 |
| proteus, `STABILIZATION_TYPE=2` (the K-upwinded low-order scheme, below) | 0.5505 | 0.3565 | -10.00 |

  The engine's FCT front is 0.0041 and 0.012 m (a sixth and a half of an element) ahead of proteus's FCT front. The two limiters are not the same algorithm (here the correction is applied after a Newton solve of the low-order scheme
  rather than inside the time step; the saved profiles are not expected to agree digit for digit), and the engine's FCT moves the front back only a little toward the Galerkin one: on this smooth, strongly
  diffusive problem the upwinding costs little. The HYDRUS and Szymkiewicz decks, where FCT matters, are in the next section.
* **Cost** (`stabilization_timing.py`, 3D tetrahedra, 31^3 nodes, 29,791 rows, 424 k nonzeros): the three passes take 0.3 ms on the RTX 6000 Ada (7% of the 4.7 ms colored assembly) and 0.6 ms with 12 threads on the Mac (8%).
  The limiter needs `Mc` and `d` as well; both are assembled once per step with the Jacobian's pattern.
* **Caveats.** The limiter is linearized and applied once after the low-order solve, as in Zalesak's original; nonlinear limiting (iterating the correction, or inside the Newton loop) is not done. It does not make the Galerkin
  solution bit-for-bit reproducible across platforms (the Richards moisture inversion uses `pow`).


## The K-upwinded low-order scheme, and the HYDRUS and Szymkiewicz decks (`richards_decks.py`, `upwind_test.py`)

`richards_regression.py --case celia|hydrus|szymkiewicz` runs the three one-dimensional decks of `test/richards` (the parameters are the decks', computed as they compute them) and compares with the
profiles proteus saved for each. The Szymkiewicz (2009) deck is Brooks-Corey with Mualem conductivity (`PSK_type=BC_MUALEM`), so `richards.h` gained `bcWetting`, a port of `bc_wetting` in `pskRelations.h`
(`psk` argument of the Richards entry points: 0 van Genuchten, 1 Brooks-Corey-Burdine, 2 Brooks-Corey-Mualem). The Celia result is unchanged (2.5e-13).

**What the decks showed about the Jacobian-based upwinding of the previous section.** On HYDRUS (a 20 m sand column, hydrostatic about a water table, 0 m head at the top, 101 nodes) the relative conductivity
falls from 1 to 1e-10 over the top 2 m, and the Galerkin quadrature takes `K_r` from its points, where the dry side dominates: the flux into the first dry element is about 1e-10 of the true one. Plain Galerkin
undershoots to -38.7 m and is stuck; **the Jacobian-based upwinding of `lowOrder.h` is stuck too** (its dissipation is the size of the Jacobian's off-diagonals, which carry that same tiny `K_r`): the front sits
at the top of the column (19.7 m) for the whole two days, where the reference front is at 16.07 m after 5 h and 11.4 m after 11 h. On Szymkiewicz (11 nodes) Newton does not converge for it at all. It is not a bug of
the pass; the operator cannot move a front into a medium where the conductivity ratio is that large. The right low-order operator for Richards is not "Galerkin plus symmetric dissipation" but proteus's.

**The K-upwinded low-order operator** (proteus's `STABILIZATION_TYPE=2`, which, in spite of the name `EntropyViscosity`, is this scheme: it matches the engine to 3e-13 on Celia and Szymkiewicz): write the
flux on the total head `phi = psi - g.x`, so gravity is part of the gradient, and take the relative permeability at the **upwind node** of each edge,

    flux into i from j:  F_ij = kr_up c_ij (phi_j - phi_i),   c_ij = max(0, -S_ij),   up = i if -S_ij (phi_j - phi_i) <= 0, else j,

with `S` the stiffness `K_s grad phi_i . grad phi_j` and the lumped mass. The same node is upwind from both sides of an edge, so it is conservative; the residual sums to zero and so does every column of the Jacobian.
The matrices come from the element kernel itself: `mode` 1 of `RichardsVGM` is `S` (relative permeability 1, no storage or gravity), `mode` 2 is the **frozen** Galerkin stiffness `C` (`K_r` at the quadrature points, no derivative
of `K_r`); `richards_kr_nodes` gives the nodal `kr` and `dkr/du`, `upwind_flux` the residual and the Jacobian on the pattern (row-parallel, `lowOrder.h` `upwindFluxRow`). The FCT correction on it is
`g_ij = -M_C,ij (dm_j - dm_i)/dt + A_ij` with the antidiffusive flux `A_ij = (-C_ij - kr_up c_ij)(phi_j - phi_i)` (Galerkin minus upwinded; `antidiffusive_flux`), through the same bounds and apply passes
(`fct_limit_flux`, optionally with proteus's bounds). Proteus imposes the Dirichlet values on this scheme strongly, so `--upwind` requires `--bc strong` and sets them before each Newton iteration, as proteus does. The conductivity is taken per node
from one material and the density factor assumes `beta = 0` (true of every deck).

* **Tests** (`upwind_test.py`, 3D tetrahedra, both retention models): `S` and `C` symmetric with zero row sums to 2e-16; `upwind_flux` against a numpy edge loop 0 (residual) and 1.5e-16 (Jacobian); conservation to 1e-15; the
  Jacobian against central finite differences of the residual to 1e-5 or better; the antidiffusive flux antisymmetric; FCT on it keeps the local bounds exactly and the mass to 5e-18, and with `A_ij = d_ij (v_i - v_j)` it
  gives the bits of the Jacobian-based `fct_limit`. The Kokkos module agrees with the host (bitwise on the Mac; 6e-14 at most on `ceds-model`, CPU and RTX 6000 Ada alike).
* **Against proteus** (`richards_regression.py --bc strong --upwind [--fct]`, proteus's Newton rule): the largest pressure-head differences from the saved columns are

| | low-order vs `stab_2` | FCT vs `FCT`, bounds over our two states | FCT vs `FCT`, `--proteus-limiter` |
|---|---|---|---|
| Celia (12 h, 24 h) | 3.1e-13, 2.7e-13 | 4.2e-4, 1.3e-4 | **3.0e-8, 5.6e-9** |
| Szymkiewicz (1.5 h, 3 h) | 6.3e-14, 1.1e-13 | 3.1e-10, 7.9e-7 | **6.2e-10, 1.9e-12** |
| HYDRUS (5 h, 11 h, 48 h) | 4.6e-2, 2.3e-2, 6.0e-9 | 2.2, 1.6, 2.1e-7 | 6.1e-2, 3.6e-2, 2.1e-7 |

  **The FCT difference is the limiter's bounds, not a second limiter.** Proteus's `FCTStep` takes the local extrema (the `Qposi`, `Qnegi` of Zalesak's limiter) over the low-order moisture content `mLow` of a node and its
  neighbours alone; the engine's default takes them over `mLow` and the old moisture content, a wider interval. `fct_limit_flux(..., bounds_old=False)` / `--proteus-limiter` takes them over `mLow` alone, and then the
  engine reproduces proteus's FCT column to 3e-8 (Celia; Newton-tolerance level) and 2e-12 (Szymkiewicz). **The `alpha_dot` limiter in `FCTStep` (a limiter on the difference of the nodal time derivatives) is not used by
  these decks and is not ported:** it sits in the `MONOLITHIC` branch, and `Richards.py` passes `MONOLITHIC = 0` to `FCTStep` (a `cek hack` comment) whatever the deck says; in the branch that runs, `alpha_dot` is computed and discarded.
  (An earlier version of this section attributed the Celia difference to it; that was wrong.) The monolithic variant, with the correction inside the Newton loop, has no proteus reference to validate against and the code
  carries the comment that it is probably not right for Richards. Using the nodal time derivative `mDotLow` that proteus's residual stores (the fluxes over `M_L`) in the mass-matrix term, instead of `(mLow - mold)/dt`, makes the
  agreement much worse (2.0 on Celia), so that is not what the limiter sees. The HYDRUS difference (a single node at the edge of the front, in the low-order scheme as well) does not depend on the Newton tolerance of the engine
  between 1e-4 and 1e-12 and is not explained; the fronts agree to 3e-4 m.
* **Against a reference** (`richards_decks.py`: the engine's own K-upwinded FCT on a mesh 8 times finer in space and in time (16 for Szymkiewicz), with one twice finer still to show it has converged: the moisture
  content differs between the two by an integral of 1.7e-5 (Celia), 3.7e-3 (HYDRUS) and 5.3e-3 (Szymkiewicz), a small fraction of the errors below). Front = where the moisture content falls through the middle between the wet
  boundary value and the driest initial value; `L1` is the integral of |theta - theta_ref| over the column (m).

| deck, time | scheme | L1 | front (m) | lowest psi |
|---|---|---|---|---|
| **Celia**, 12 h (reference front 0.6736) | Galerkin = proteus | 1.6e-4 | 0.6743 | -11.60 |
| | Jacobian upwinding | 5.1e-4 | 0.6687 | -10.00 |
| | K-upwind (= proteus `stab_2`) | 3.2e-3 | 0.6429 | -10.00 |
| | K-upwind + FCT (= proteus FCT, 3e-8 with `--proteus-limiter`) | 7.2e-4 | 0.6681 | -10.00 |
| **HYDRUS**, 5 h (reference 16.07) | Galerkin | 1.5 | 19.90 | -38.7 |
| | Jacobian upwinding | 1.4 | 19.70 | -19.4 |
| | K-upwind (= proteus `stab_2`) | 6.0e-2 | 15.94 | -15.40 |
| | K-upwind + FCT (= proteus FCT) | 5.2e-2 | 15.95 | -16.12 |
| **HYDRUS**, 11 h (reference 11.44) | K-upwind | 7.0e-2 | 11.32 | -10.80 |
| | K-upwind + FCT | 6.3e-2 | 11.32 | -11.27 |
| **Szymkiewicz**, 3 h (reference 3.39) | Galerkin | 0.36 | 4.25 | -10.87 |
| | Jacobian upwinding | Newton fails at 0.0355 d | | |
| | K-upwind (= proteus `stab_2`) | 0.11 | 2.99 | -7.50 |
| | K-upwind + FCT (= proteus FCT) | 7.6e-2 | 3.06 | -7.50 |

  HYDRUS at 48 h is wet through and every K-upwinded scheme gives psi = 0 to 1e-15 (the saved columns agree). Reading: on a sharp front the K-upwinded low-order scheme lags by about half an element, and FCT
  recovers part of it (a third of the L1 error on Szymkiewicz, 13% and 10% on HYDRUS, where the front is narrower than one element and the local bounds cannot admit a sharper one); on Celia, the smooth case,
  the Jacobian-based upwinding is the most accurate of the stabilized schemes (L1 5.1e-4 against 3.2e-3 for the K-upwind scheme, which lags by a third of an element). The engine reproduces proteus's two stabilized schemes on all three decks, and plain Galerkin
  cannot run two of them.
* **On the GPU**, Celia, Szymkiewicz and HYDRUS with `--upwind --fct` print the same differences from the saved profiles as the host (RTX 6000 Ada, `adrh_kokkos` CUDA module).
* **Not done.** One material per run for the nodal `kr` (the node's material number is an argument, but the decks use one); `beta != 0`; 2D and 3D decks; the cost of the new passes (they are the same
  row-parallel gathers as the dissipation pass, 2 to 4% of the assembly there); FCT iterated inside the Newton loop.

## Heterogeneous media and 2D: `richards_nd.py`, `richards_nd_test.py`, `richards_2d.py`

The K-upwinded scheme on a general P1 mesh (`KUpwindRichards`, 1, 2 or 3 dimensions), built from the same passes as the 1D `Column` (which stays pinned to proteus's saved profiles): the element kernel in
mode 1 and 2, `richards_kr_nodes`, the new `richards_m_nodes` (the nodal storage `m` and `dm/du` with each node's own retention curve, in both modules), `upwind_flux`, `antidiffusive_flux` and the FCT limiter.

* **Nodal materials.** `K_s` is the element's. The storage and the upwind relative permeability are nodal, so a node has one retention curve, the first incident element's, which is what proteus does
  (`Richards.py`, `nodeMaterialTypes_n`). `first_incident_material` does it.
* **`interface='element'`** takes the upwind `K_r` of an element's flux from that element's own material: the flux is split by material, `F = sum_m kr_m(psi_up) c_m (phi_j - phi_i)`, `c_m` the upwinding weight of the stiffness of
  the elements of material `m` alone (psi is continuous across the interface). For one material it is the same scheme. Measured on the two-layer steady column of `richards_layers.py` against the ODE (shooting) solution:

  | nodes | proteus's rule (`interface='node'`) | `interface='element'` |
  |---|---|---|
  | 21 | 5.8e-1 | 6.3e-3 |
  | 41 | 4.7e-1 (rate 0.32) | 3.3e-3 (0.91) |
  | 81 | 3.4e-1 (0.44) | 1.7e-3 (0.96) |
  | 161 | 2.3e-1 (0.57) | 8.7e-4 (0.98) |
  | 321 | 1.4e-1 (0.68) | 4.4e-4 (0.99) |

  Proteus's rule converges, slowly, because the one element above the interface takes `K_r` from the soil below it. The storage stays nodal in both so that FCT can invert the moisture content analytically.
* **FCT bounds on the head** (`fct_correct(bounds='head')`, the default; `'moisture'` is proteus's). The moisture contents of two soils at one head are not comparable, so a moisture bound across an interface lets a node dry
  below every head around it: in the lens deck (`richards_2d.py --lens --fct`, 41 x 41 nodes) the head falls to -11.3 m after 70 steps and to -15.7 m after 280, from an initial minimum of -10. Bounding the head
  (`head_extrema` over a node's row, mapped to moisture content with the node's own curve, `fct_limit_flux_bounded`) keeps it at -10 and is the moisture bound for one material (equal to 1e-12). The moisture bound is
  proteus's; whether proteus has the same undershoot on a lens was not run.
* **Tests** (`richards_nd_test.py`, both modules, in CTest): the 1D Celia deck against `Column`, low-order 8e-14 and FCT 8e-13; the Celia column as a strip of triangles (x-uniform) equals the 1D solution to 3e-9 (low-order),
  and FCT keeps the mass exactly and stays in the data's bounds; the layered column and its strip; a transient layered FCT run; the head bounds against the moisture bounds; `golden.py` has the three new passes
  (bitwise equal host and Kokkos).
* **The 2D deck** (`richards_2d.py`): proteus's `re_vgm_sand_10x10m_2d` (10 m x 10 m sand, ponded on the middle third of the top, 0.35 d) with an optional lens of a finer soil. There is no saved reference for the deck; it
  checks the bounds, and host and Kokkos agree (570 Newton iterations, the same mass, on the lens with FCT). Not done: a comparison with a proteus run of the same deck, and 3D.

Compressibility (`beta != 0`) is in the K-upwinded scheme through the edge form of the flux (section below).

### Cost of the K-upwinded passes (`upwind_timing.py`, `upwind_passes_timed`)

3D tetrahedra (Kuhn cube, six per cell), device-resident, best of 5, against the colored assembly of the advection-diffusion element kernel on the same mesh as a yardstick (the Richards element kernel costs about
three times that per element, above; that factor was not re-measured here). K_r varies by orders of magnitude in the state (a wet layer on a dry column). Times in ms:

| pass | RTX 6000 Ada, 29,791 rows (424 k nnz) | Ada, 132,651 rows (1.9 M nnz) | Mac, 12 OpenMP threads, 29,791 rows | Mac, 1 thread, 29,791 rows |
|---|---|---|---|---|
| `kr_nodes` | 0.02 | 0.06 | 0.12 | 0.47 |
| `upwind_flux` (residual and Jacobian) | 0.02 | 0.07 | 0.14 | 0.71 |
| `antidiffusive_flux` | 0.02 | 0.08 | 0.10 | 0.50 |
| FCT limiter, moisture bounds (3 passes) | 0.21 | 0.58 | 0.62 | 1.68 |
| `head_extrema` | 0.01 | 0.02 | 0.08 | 0.38 |
| FCT limiter, bounds given (3 passes) | 0.20 | 0.65 | 0.55 | 1.29 |
| **head-bounded FCT step** (antidiffusive flux + head extrema + limiter) | **0.23** | **0.76** | **0.73** | **2.17** |
| yardstick: colored AD assembly | 7.3 | 13.5 | 17.0 | 121 |

The whole FCT step is 3-6% of the yardstick on the Ada and 4% on the Mac with 12 threads: the limiter's three row-parallel passes dominate (0.2-0.65 ms on the device, launch-bound and gather-heavy), the flux passes
are a few percent of that. Upload of the CSR arrays and the state is 2-6 ms and is paid once, not per pass. Not measured: a T4 (the CI job runs the tests, not this script), and a whole Newton iteration with the
Richards element kernel in the denominator.

## Implicit FCT, the correction inside the solve (`implicit_fct_step`; `richards_implicit_fct.py`)

proteus's `FCT_TYPE='implicit'` (Dielmann, Kuzmin and Kees, JNMA), the first FCT in this directory that is not applied after the solve. One step: (1) a Galerkin predictor `p^H` (consistent mass, K at the quadrature
points; if its Newton iteration fails within `predictor_maxit` = 20 iterations the low-order solution stands in, as proteus's `FCT_PREDICTOR_FALLBACK`), (2) the raw antidiffusive fluxes at `p^H`,
`g_ij = M_C,ij (sDot_i - sDot_j) + A_ij` with `sDot = (s(p^H) - s^n)/dt` and `A` the Galerkin flux minus the K-upwinded one, multiplied by the nodal-variation prelimiter
`alpha_nv,i = 1 - (|sum_j (h_i - h_j)| / sum_j |h_i - h_j|)^3` (`j` over the nodes coupled by the mass matrix, `h` the pressure head; `min(alpha_i, alpha_j)` per edge), limited by Zalesak's ratios with **global**
bounds (each node's `theta_r`..`theta_s`) taken from `s^n`, giving the datum `s^{n,FCT} = s^n + (dt/M_L) sum_j alpha_ij g_ij`, and (3) the low-order solve
`M_L (s(p) - s^{n,FCT})/dt + A~(p) u(p) = 0` started from `p^H`. New passes in `lowOrder.h` (both modules): `nodalVariationRow`, `fctFluxRowAlpha`, `implicitFctLimit` (it reuses `fctBoundsRowGiven` and
`fctApplyRow`, with `s^n` as the base point); `golden.py` has them (bitwise equal host and Kokkos). The limiter agrees with an independent numpy version to 0 (random fluxes, two materials), and the datum conserves
`sum M_L s` to 4e-19. proteus keeps `s^n` at the Dirichlet nodes (`bc_mask`); so does `implicit_datum` (`mask=False` shows the exact conservation).

Measured (`richards_implicit_fct.py`, 1D decks at proteus's mesh and step, first saved time, against the post-step K-upwinded FCT on a mesh 8 times finer in space and time; L1 of the moisture content; the predictor never failed):

| deck | K-upwind, lumped | post-step FCT | implicit FCT | Newton iterations per step (low / post / implicit) |
|---|---|---|---|---|
| Celia, 12 h | 3.2e-3 (front 0.6429) | 7.2e-4 (0.6681) | 1.1e-3 (0.6656) | 3.0 / 3.4 / 5.1 |
| HYDRUS, 5 h | 6.0e-2 (15.94) | 5.2e-2 (15.95) | 5.6e-2 (15.95) | 7.3 / 8.4 / 10.3 |
| Szymkiewicz, 1.5 h | 0.168 (3.78) | 0.153 (3.79) | 0.161 (3.79) | 3.0 / 3.1 / 5.9 |

(reference fronts 0.6736, 16.07, 4.14 m). On these decks it is **less accurate than the post-step FCT and better than the low-order scheme**, at 1.5 to 2 times the Newton iterations (two solves). Its head stays at the
data's bounds (HYDRUS lowest psi -15.40 against the post-step FCT's -16.12). The differences from proteus's saved profiles are scheme differences: the saved `stab_2` and `FCT` columns are matched exactly by the
low-order and post-step schemes (0.000), and the implicit one is a third scheme.

**Where it fails: the lens deck** (`richards_2d.py --lens`, 21 x 21 nodes, 35 steps): with proteus's global bounds (`bounds='global'`) the head falls to **-174 m** (initial minimum -10; 7 nodes below it; -75 m at 41 x 41 nodes and 70 steps), because
the global lower bound lets a dry node's datum fall to `theta_r`, where a tiny moisture change is an enormous head change; the prelimiter does not prevent it. `bounds='local'`, now the default (the head's extrema over a node and its
neighbours at the predictor and the old state, mapped to moisture content with the node's own curve) holds it at -10.3 m, still a small undershoot (3 nodes), not zero. The Galerkin predictor fails often at the deck's
coarse step (6 of 35 steps without the lens, 11 of 35 with it, at 21 x 21 nodes; 1 of 280 at 41 x 41) and each failure costs `predictor_maxit` iterations before the fallback. Whether proteus's implicit FCT
does the same on a lens was not run. The post-step FCT with head bounds stays inside [-10, 0.1] on the same deck.

**Which default.** `bounds='local'` is the default of `implicit_fct_step`, `implicit_datum` and `run(implicit=True)`; `bounds='global'` is proteus's algorithm and stays available (`proteus_reference_2d.py` and the proteus comparisons below pass it, and the 1e-10 agreement with proteus's implicit FCT is for it). The two cost nothing on the benchmark decks: on Celia, HYDRUS and Szymkiewicz (`richards_implicit_fct.py`, a row for each) the L1 errors, fronts and lowest heads equal each other to the digits printed (1.10e-3, 5.56e-2, 1.61e-1; HYDRUS 10.42 against 10.26 iterations per step), and only local bounds hold the lens (-10.3 m against -174 m). This is a departure from proteus's FCT, made for this branch (main, 2.0); the 1.9.x line and the papers that use it are unaffected.
(`richards_implicit_fct.py`, a row for each) its L1 errors, fronts and lowest heads equal the global-bounds ones to the digits printed (1.10e-3, 5.56e-2, 1.61e-1; HYDRUS 10.42 against 10.26 iterations per step), and it is the only one of the two that
holds the lens (-10.3 m against -174 m). Use it wherever the data have a dry region next to a wet one; making it the default would be a deviation from proteus that the PR should propose, not assume.

## Compressibility, `beta != 0` (`KUpwindRichards(..., rho=, beta=, rho0=)`)

`rho = rho_t exp(beta psi)`, with `rho0` as in proteus (`rho0 != 1` enters only through `f = (rho^2 / rho0) K_r K_s g = rho^2 K_r K_s (g / rho0)`, so the engine's bindings, which fix `rho0 = 1`, are used with `g / rho0`;
`beta >= 0` for FCT, see below). The total-head form of the K-upwinded flux needs the gravity term to be the gradient of a nodal head, which it is not when `rho / rho0 != 1`: the Galerkin flux is
`rho K_r K_s (grad psi - (rho / rho0) g)`. The edge flux is therefore `F_ij = (rho K_r)_up c_ij dphi_ij` with `c_ij` from the unit-density stiffness `S`, the upwinded coefficient `rho K_r` taken at the upwind node, and
`dphi_ij = u_j - u_i - rhoEdge g.(x_j - x_i)`, `rhoEdge` the mean of `rho` over the edge (what proteus's edge loop does with its salinity density). New passes `upwindFluxRowE` and
`antidiffusiveFluxRowE` (`upwind_flux_edges`, `antidiffusive_flux_edges`; both modules, bitwise equal in `golden.py`) take `dphi` and its partials in `u_i`, `u_j` on the pattern, so the Jacobian includes the density's
dependence on `u`; with `rho = 1` and `beta = 0` they are the nodal-head passes (1.8e-15). The storage is `rho theta` (`richards_m_nodes` has had `rho`, `beta` all along), the Galerkin predictor and the stiffness
of the antidiffusive flux (mode 2) carry `rho` at the quadrature points, and FCT's inversion of `m = rho_t exp(beta u) theta(u)` is a bracketed, safeguarded Newton iteration per node: `m` falls below `rho_t theta_r` at very negative
heads, where the analytic inverse does not reach.

Checked against an exact solution (`richards_nd_test.py`, a steady column, `psi = -1` below and `-3` above, sand with `K_s = 7.967`): with `rho = exp(beta psi)` and gravity down the flux `Q = -rho K (psi' + rho)` is constant, so
`psi' = -Q exp(-beta psi) / K(psi) - exp(beta psi)` and `Q` follows by shooting. The Galerkin discretization of the same equation converges to it at rate 2.00 for `beta = 0.05` and `0.5` (so the ODE is right), and the K-upwinded
scheme converges to the same solution:

| beta | nodes 21, 41, 81, 161: max error of psi | rate (finest pair) |
|---|---|---|
| 0.05 | 2.9e-2, 1.6e-2, 8.4e-3, 4.3e-3 | 0.96 |
| 0.5 | 1.3e-1, 7.8e-2, 4.6e-2, 2.5e-2 | 0.87 |

The flux Jacobian agrees with central differences (6e-10, beta = 0.4); the x-uniform strip equals the column (2e-12, beta = 0.05); FCT keeps `sum M_L rho theta` exactly; the implicit step stays in the data's bounds.

**`rho_t != 1` and `rho0 != 1`.** The ODE with them is `psi' = -Q / (rho_t exp(beta psi) K) - rho_t exp(beta psi) / rho0`; at `beta = 0.3, rho_t = 1.2, rho0 = 1.5` the Galerkin solution converges to it at rate 2.00 and the K-upwinded one at 0.91
(test in `richards_nd_test.py`).

**Negative `beta`.** The steady column converges the same way (`beta = -0.05` and `-0.3`: Galerkin rate 2.00, K-upwind 0.97 and 0.99), and so does the low-order transient (`-0.05`, below). FCT does not work: it inverts
`m = rho_t exp(beta u) theta(u)` for the head, and with `beta < 0` that `m` is not monotone (it rises again as `u -> -infinity`, because `exp(beta u)` grows faster than `theta` falls, and falls beyond saturation), so the head is not
determined. The inversion returned heads of order 1e18 without complaint; it now raises `ValueError`. At `beta = -0.3` and `psi = -3` `d(rho theta)/dpsi` is negative, so the transient problem there is ill-posed (the reference
solver cannot integrate it), which is the physical reason: a fluid that gets lighter under pressure.

**The transient** (`compressible_transient` in `richards_nd_test.py`): a wetting front, `psi = -3` initially and at the top, `-1` at the bottom, `t = 1`, against a method-of-lines solution written without the engine (node-centred
conservative differences, 1601 nodes, BDF to 1e-10; its own error 9e-4 at `beta = 0.3` and 6e-5 at `-0.05`). Backward Euler, 50, 100 and 200 steps on nodes 41, 81 and 161:

| case | K-upwind errors | rate | Galerkin errors | rate |
|---|---|---|---|---|
| beta 0.3, rho_t 1.2, rho0 1.5 | 5.9e-1, 3.9e-1, 2.3e-1 | 0.76 | 4.1e-2, 1.5e-2, 5.9e-3 | 1.32 |
| beta -0.05 | 1.3e-1, 7.1e-2, 3.8e-2 | 0.92 | 7.8e-3, 3.9e-3, 1.9e-3 | 1.00 |

The Galerkin transient converging to the independent solution is what shows the reference is the right equation; the K-upwinded scheme converges to it more slowly, as at `beta = 0` (first order at a sharp front). Not done: a comparison
with proteus's own low-order scheme at `beta != 0`, whose edge loop uses a density ratio from its salinity density only (no `exp(beta psi)`) and was not reproduced here; at the deck value `beta = 1e-4` and heads within 10 m the factor
`exp(beta psi)` differs from 1 by 1e-3.

## Against proteus's own code (`proteus_reference_2d.py`)

The JNMA branch of proteus (cekees/proteus#123 and #124) has the implicit FCT; its 2D sand deck and 1D Celia deck were run beside `richards_nd.py` on the same mesh, step for step (`proteus_reference_2d.py` copies and patches
the decks in a scratch directory; it needs that proteus built, `PROTEUS_REF_BIN` and `PROTEUS_REF_DECKS`). Setup that matters, each found by a comparison that did not match: **the deck's element quadrature is order 4**, not 3 (the Galerkin
step depends on it, the low-order scheme does not); **Dirichlet conditions must be strong** (`forceStrongConditions=True`; the default is a weak boundary flux that leaves the boundary nodes 5e-2 m off the data); **the pond edges
must fall on mesh nodes**: on the deck's Triangle mesh the edge at x = 10/3 lies inside a boundary face, and proteus's weak boundary flux then acts on a face that is part pond and part not (my residual at proteus's step-1
solution was 1e-13 at every free node but the two beside the pond); with the pond limits given a 1e-9 tolerance and a structured 31 x 31 grid the two agree. All numbers below: 31 x 31 nodes, 35 steps of 0.01 d, Newton to 1e-12.

| case | max abs dpsi | max abs dtheta | proteus psi min |
|---|---|---|---|
| Galerkin | 5.6e-10 | 8e-17 | -10.0325 |
| Galerkin, beta 1e-4 / 0.05 | 1e-11 / 3e-14 | 1e-16 | |
| Galerkin, with the lens (144 elements of a second soil) | 4.2e-10 | 2e-16 | -10.0325 |
| K-upwind low-order | 5.2e-11 | 2e-16 | -10.0000 |
| K-upwind low-order, with the lens, nodal rule (`interface='node'`) | 7.1e-11 | 3e-16 | |
| K-upwind low-order, with the lens, `interface='element'` | 0.24 | 1.4e-2 | (a different scheme) |
| K-upwind low-order, beta 1e-4 / 0.01 / 0.05 | 1.4e-3 / 0.145 / 0.51 | 2.9e-6 / 3.0e-4 / 1.9e-3 | |
| implicit FCT | 1.8e-2 | 8e-4 | -1760.47 (mine) / -1760.45 |
| implicit FCT, with the lens | 3.5e2 | 3.3e-3 | -1586.3 / -1585.9 |

Reading: **the engine's Galerkin, low-order and nodal-material passes are proteus's to round-off in 2D, with and without the lens, and the Galerkin path with compressibility is proteus's to 1e-11 up to beta = 0.05.** The nodal-material rule
(`first_incident_material`) is the same as proteus's, and the `interface='element'` rule is a different scheme (0.24 m, 1.4e-2 in moisture content on this lens). **The implicit FCT agrees with proteus's**: on the 1D Celia deck at proteus's step to
1.4e-3 m in the head and 4.4e-6 in moisture content (an ad hoc run, not in the script); on this 2D deck at 350 steps of 0.001 d step by step (also ad hoc): 3e-17 in moisture content after one step, at most 1e-10 through step 20 (including
intermediate dry-node heads of -137 m and -90 m that both codes produce), growing afterwards to 1.6e-5 at step 50 and 5e-4 at step 350 (the limiter clips at the bounds, so tiny differences amplify; the Newton tolerance is not the
cause: 1e-8 and 1e-12 give the same). **Proteus's implicit FCT has the same undershoot as mine**: -1760 m on the plain 2D sand deck at 0.01 d, -35 m on the lens at 0.0025 d, -10.3 m to -13 m at 0.005 to 0.001 d. It comes from the global
bounds in the JNMA algorithm (and the Galerkin predictor failing at the coarse step), not from this implementation; `bounds='local'` is an addition here, and the default.

**Compressibility: proteus's low-order scheme is not consistent for beta != 0.** The low-order schemes differ by an amount proportional to beta (table). Against the exact steady compressible column of `richards_nd_test.py`
(`proteus_reference_2d.py --steady`, the 1D Celia deck run to a steady state, psi = -1 below and -3 above, T = 200 d; nodes 21, 41, 81; max |psi - exact|, convergence rate in brackets):

| beta | proteus Galerkin | proteus low-order | richards_nd K-upwind |
|---|---|---|---|
| 0.05 | 2.5e-4, 6.5e-5 (1.96), 1.6e-5 (2.01) | 5.6e-2, 4.6e-2 (0.28), 4.1e-2 (0.18) | 2.9e-2, 1.6e-2 (0.85), 8.4e-3 (0.93) |
| 0.5 | 9.1e-4, 2.5e-4 (1.87), 6.2e-5 (2.01) | 0.33, 0.32 (0.06), 0.31 (0.04) | 0.13, 7.8e-2 (0.69), 4.6e-2 (0.78) |

Proteus's Galerkin scheme converges to the exact solution at rate 2 (and to my Galerkin's errors); its low-order scheme stops converging at an error that grows with beta, 4e-2 m at 0.05 and 0.31 m at 0.5, because the density enters the
gravity term of its edge flux only through the salinity density (`rho_edge / rho`) and not through `exp(beta psi)` (read in `Richards.h`'s edge loop; not changed here). The edge form of `richards_nd.py` converges to the exact solution. At the deck's
beta = 1e-4 the effect is 1.4e-3 m after 35 steps, so the decks' results are not affected.

**The post-step FCT (proteus's semi-implicit FCT, `FCT=True`, `FCT_TYPE='semi-implicit'`) in 2D.** On the deck it returned the low-order solution to the last digit, and the reason is a veto in `Richards.FCTStep`: on the
Newton-invert path (`nd > 1`, so every 2D case) the limited step is thrown away and the low-order head kept whenever the correction moves any node's head by more than 1 m (`Richards.Coefficients(FCT_DU_MAX=...)`, default 1), and a dry node's moisture
content moves its head by metres. `diagnostics=('fct',)` logs each decision; `proteus_reference_2d.py` sets `FCT_DU_MAX=1e30`. With the veto lifted, the limiter differs from `fct_correct(bounds='moisture')` in two ways, both found by
dumping proteus's step-1 arrays and replaying its limiter in numpy (the replica reproduces proteus's limited moisture content to 6e-17):
* **`FCT_SATURATED_LOW_ORDER`**: where the low-order head is >= 0 the room to the bounds is set to zero, so every flux touching the node is limited to zero. `fct_correct(..., sat_low=True)` does the same (`richards_nd_test.py`).
* **The Dirichlet rows of the high-order flux matrix.** Proteus's `TransportMatrixConsistent` also holds the weak exterior-boundary terms (`Richards_IIPG_exterior`), which are not symmetric: in the row of a Dirichlet node they add an antidiffusive flux
  that the transposed entry does not have (1.45e-2 against 0 at the pond nodes after one step). Every interior edge agrees with `antidiffusive()` to 1e-17 (4918 edges), and with proteus's fluxes in the Dirichlet rows and mine everywhere else the
  replayed limiter reproduces proteus to 2e-16; with mine in the Dirichlet rows it is off by 6e-3. The rows are masked (`bc_mask`), so no Dirichlet value changes, but the Zalesak ratios there limit the fluxes of their free neighbours: five nodes
  beside the Dirichlet corners and pond edges differ by up to 3e-3 in moisture content after one step (35 steps: max |dtheta| 2.5e-2, mean |dpsi| 4e-3). It is a property of the weak boundary treatment, not an error in the limiter, and `fct_correct`
  does not model it.

**The deck's Triangle mesh** (`--mesh triangle`, 1496 nodes, 2861 elements): the Galerkin scheme does not run in proteus at the deck-copy's fixed 0.01 d step (the nonlinear solver fails at step 12; the original deck uses its adaptive controller), so there is no
Galerkin comparison, and the K-upwinded low-order scheme differs by 5.8e-2 in moisture content (max |dpsi| 6.4 m at a dry node, mean 0.13 m). The difference is the pond: no mesh node lies at x = 10/3 or 20/3 (the top nodes are 3.125, 3.4375, ...),
so the Dirichlet nodes cover 3.44 to 6.56 here, and in proteus the weak flux also acts on the two faces that are part pond. Its wetting front lies between mine with the pond on nodes only and mine with the pond widened by one node at each end
(mean |dtheta| 4.4e-3 and 7.8e-3 for those two; proteus's is the one in between), the largest differences being the nodes just outside the pond (x = 3.125 and 6.875, y = 10: -0.169 against -0.102). Matching it needs the weak boundary flux of
`Richards_IIPG_exterior`, which this driver does not model.

## Two-phase level set, slice 1: VOF (`vof.h`, `vof_test.py`)

The first model of the two-phase family (C19, decided 2026-11-01). `vof.h` splits proteus's VOF
(`mprans/VOF.h`, `STABILIZATION_TYPE=0`, as `test/TwoPhaseFlow/damBreakLevelSet.py` runs it) into the part a
generator writes, `VOFCoefficients` (m = porosity u, f = porosity v u), and two kernels that do not know it is VOF:
`scalarSUPGElement` (BDF1 with the mass history as a point field, metric SUPG tau, the residual-based shock-capturing
viscosity computed and returned, the lagged one used) and `scalarUpwindFace` (upwind advective flux with Dirichlet
inflow and prescribed-flux points). Fields from other models (RANS2P's velocity, the porosity) come in per quadrature
point through the functor's `nFields`.

`vof_capture.py` runs the damBreak case in proteus and keeps one VOF Newton iteration exactly as its kernel saw it;
`vof_damBreak_he0.05.npz` (733 triangles, t = 0.063, dt = 0.032) is its output. `vof_test.py` gives the engine the same
arrays and checks against proteus: residual 1.9e-15, Jacobian 2.4e-16, CFL, boundary flux and upwinded trace bitwise,
the next step's shock-capturing viscosity 5.5e-12 of its maximum (round-off amplified by the formula where u is flat;
the test explains the tolerance), and one Newton step reproduces proteus's solution to 3.3e-16.

## Two-phase level set, slice 2: NCLS (`ncls.h`, `ncls_test.py`)

NCLS (`mprans/NCLS.h`, `STABILIZATION_TYPE=0`) is the same element kernel with a Hamiltonian in place of the advective
flux. `ScalarTerms` now carries H and dH next to f and df, and a functor declares which it has (`hasF`, `hasH`). The
kernel compiles only those terms and uses a = df + dH as the characteristic velocity of the strong residual, the
adjoint, tau and the CFL. `NCLSCoefficients` is m = u, H = v . grad u (fields: the velocity only, no porosity). VOF's
fixture still passes unchanged (`vof_test.py`).

The faces are NCLS's own kind, `scalarInflowPenaltyFace`: there is no flux to upwind. The face term is
F = -(v . n)(u - u_b) on inflow and 0 on outflow. u_b is the Dirichlet value where a point has one, and RDLS's
redistanced trace (`ebqe_rd_u_ext`) elsewhere. The face kernel also returns the boundary trace and its normalized
gradient, which proteus hands to the other models (`ebqe_u`, `ebqe_grad_u`). `signChangeElement` marks the elements
where the level set changes sign. Their nodes are the `interface_locator` that RDLS freezes.

`vof_capture.py OUT model=ncls` records NCLS from the same run at the same step. `ncls_damBreak_he0.05.npz` (733
triangles, t = 0.063) holds 18 of 79 boundary faces with a nonzero inflow term and 33 interface nodes. The deck has no
Dirichlet points, so that branch is not exercised. `ncls_test.py` checks the engine against proteus:

| check | max difference, relative |
|---|---|
| residual | 1.7e-16 |
| Jacobian | 2.5e-16 |
| next step's shock-capturing viscosity | 2.4e-16 |
| CFL, boundary trace, interface locator | bitwise |
| grad u at the points (q_n), normalized boundary gradient | 2.4e-15 |
| solution after one Newton step | 1.9e-16 |

The viscosity is not amplified here as it is for VOF: the level set is not flat away from the interface.

## Two-phase level set, slice 3: RDLS (`rdls.h`, `rdls_test.py`)

RDLS (`mprans/RDLS.h` with `ELLIPTIC_REDISTANCING=0`, `freezeLevelSet=1`, `useExact=0`, the damBreakLevelSet
configuration) is the same element kernel again, with the template's first nonlinear Hamiltonian:

- `RDLSCoefficients`: S = H_eps(phi) - (1 - H_eps(phi)) with eps = epsFact h_phi, H = S |grad u|,
  dH = S grad u / (|grad u| + 1e-12), and a reaction r = -S. Its fields are NCLS's level set at the point, h_phi (the
  node diameters interpolated to the point) and the element diameter.
- What the kernel gained, each behind a functor trait or a parameter so VOF and NCLS compile as before (their fixtures
  pass with the same numbers):
  - a reaction term r (`hasR`), in the weak form and the strong residual;
  - RDLS's tau, 1/(sqrt(dH . G dH) + 1e-8) (`tauForm = 1`). There is no time term (dm_t = 0), and VOF's form
    1/sqrt(4 dm_t^2 + a . G a + 1e-8) would differ at 1e-8, not at round-off;
  - shock capturing at the current u (`SUPGParameters::lag_sc = 0`), plus a background diffusion the functor returns
    (`nuExtra`: backgroundDiffusionFactor h_e where |phi| > 2 eps).
- The Jacobian is proteus's, not the exact one. The H term is exact (dH/d(grad u) = dH). The subgrid term keeps dH and
  tau frozen, and nu is not differentiated.
- `rdlsPinElement` is the interface pin. After an element's quadrature, every node whose starting value is within eps
  of zero has its element residual row replaced by (u - u_last) wDF h_e and its Jacobian row by wDF h_e on the
  diagonal. eps there is the element's value at its last quadrature point, because RDLS.h reuses the loop variable.
  A node shared by several elements can therefore be pinned in some of them and not in others.
- No boundary terms (`NumericalFlux.DoNothing`). `rdlsTraceFace` returns the trace and its normalized gradient
  (`ebqe_u`, `ebqe_n`).
- RDLS.h writes `q[('grad(u)',0)]` from an uninitialized local (`dir`, whose assignment is commented out). That output
  is not reproduced.

`vof_capture.py OUT model=rdls` records the first RDLS Newton iteration at or after T. RDLS's kernel gets only
element-local CSR offsets, so the capture takes the Jacobian's CSR structure from the matrix. `rdls_damBreak_he0.05.npz`
(733 triangles, t = 0.062) has 30 pinned nodes (165 pinned element rows). `rdls_test.py`:

| check | max difference, relative |
|---|---|
| residual | 4.1e-14 of its max; 4.5e-17 absolute, 3e-15 of a node's H and r terms (O(sum w dV) = 1.5e-2), which cancel where \|grad u\| = 1 |
| Jacobian | 3.9e-17 |
| shock-capturing viscosity (current u) | 4.7e-15 |
| CFL | 3.2e-16 |
| boundary trace | bitwise |
| normalized boundary gradient | 2.4e-15 |
| next Newton iterate (one step with the engine's R and J) | 1.9e-16 |
| residual at the next iterate (pinned rows nonzero there) | 6.9e-14 of its max |

## Two-phase level set, slice 4: MCorr (`mcorr.h`, `mcorr_test.py`)

MCorr (`mprans/MCorr.h` with `useExact=0`, `useExact_s=0`, the damBreakLevelSet configuration) is not SUPG. It is a
steady nonlinear reaction-diffusion problem for the level-set correction u, so it runs on the generic element kernel
(`engine.h`, `elementKernelSpace`), the one Richards uses, not on the VOF/NCLS/RDLS template:

- `MCorrCoefficients`: r = porosity (H_epsH(phi + u) - H_vof), dr = porosity delta_epsD(phi + u), a = eps_diff I,
  phi = u, all scaled by H_s = H_0(phi_s) (1 in the fluid). eps_X = epsFact_X (useMetrics h_phi + (1 - useMetrics)
  h_e). No mass, advection or Hamiltonian; the masks say so, so the kernel skips those blocks. The Jacobian is exact.
- Point data: phi (NCLS's level set), H_vof (VOF's solution), the porosity, h_phi, h_e and phi_s come in as point fields.
  The generic kernel gained one hook for this, `bindPoint(q)` beside `bindElement(eN)` (`engine.h`): a functor that has
  it is evaluated as `C.bindElement(eN).bindPoint(q)`. A functor without it compiles to what it did before (the Richards
  tests print the same digits with the module built before and after the hook).
- `mcorrPointOutputs` writes what MCorr hands to the other models at the points: q_u, q_r (r without H_s) and
  q_n = grad u / sqrt(1e-8 + |grad u|^2).
- Active DOFs: with `useExact_s = 0` every element is active, so the marks are all 1 and MCorr.py's masking of the
  residual and of the Jacobian rows is the identity. The cut-cell path (`useExact_s = 1`: ghost penalty on cut faces,
  inactive rows) is not reproduced.

`vof_capture.py OUT model=mcorr` records the first MCorr Newton iteration at or after T, plus q_u, q_r and q_n at the next
iterate (MCorr starts from u = 0, where they are trivial). `mcorr_damBreak_he0.05.npz` (733 triangles, t = 0.063, 125
elements within eps of the interface) gives:

| check | max difference, relative |
|---|---|
| residual | 6.9e-15 |
| Jacobian | 3.3e-16 |
| q_r | 5.4e-15 (q_u, q_n are 0 at u = 0, and the engine gives 0) |
| every DOF and element active | yes |
| next Newton iterate (one step with the engine's R and J; MCorr has no line search) | 2.5e-14 |
| residual at the next iterate | 3.2e-15 of the first residual (4.6e-19 absolute; Newton cut the residual 1e4, so its own max is 2.6e-8) |
| q_u, q_r, q_n at the next iterate | 1.1e-14 |

## Two-phase flow, slice 5: RANS2P2D (`rans2p.h`, `rans2p_test.py`)

RANS2P (`mprans/RANS2P2D.h`, the damBreakLevelSet configuration: non-conservative form, `useVF = 1`, equal-order P1, no
turbulence closure, drag, particles, moving mesh or RBLES) is the first vector model: unknowns (p, u_0, .., u_{nS-1}),
nSpace-generic, checked in 2D.

- `navierStokesVMSElement`: the element integrals, the residual-based VMS stabilization with the lagged subgrid velocity
  and the lagged shock-capturing viscosity (round 4, e433fece).
- `navierStokesExteriorFace`: the weak boundary conditions on the exterior faces (round 5). The continuity flux
  porosity u . n; the momentum flux upwinded on the normal flow speed s = rho porosity u . n (inflow: (g - u) s, with
  g = 0 where a component has no Dirichlet value; outflow: 0) plus n (p_bc - p) where p has a Dirichlet value; prescribed
  fluxes where the advective-flux flags say so; for each entry of the stress, the SIPG flux -(a grad u_d) . n +
  max(a) C_b/h_n (u_d - g_d) with h_n = (n . G n)^(-1/2), and the symmetric adjoint term (u_d - g_d)(a n) . grad w. Two
  quirks kept: the adjoint is switched off by the row's own diffusive-flux flag (proteus passes `isDiffusiveFluxBoundary_u`
  for the (u, v) entry), and the diffusive-flux flag wins over the Dirichlet flag (the deck's top boundary has both for v).
  The Jacobian is exact, including d s/d u in the inflow upwinding.

`vof_capture.py OUT model=rans2p` records the first RANS2P Newton iteration at or after T, the face arrays and boundary
conditions, the element integrals alone (the kernel called again with no exterior faces), and the next Newton iterate with
proteus's residual there (the deck solves with SuperLU). `rans2p_damBreak_he0.05.npz` (733 triangles, 79 exterior faces,
t = 0.063; the faces are 8.2e-5 of the residual and 3.4e-2 of the Jacobian) gives:

| check | max difference, relative |
|---|---|
| residual, element integrals / elements + faces / the faces' share | 3.3e-16 / 3.3e-16 / 3.3e-16 |
| Jacobian, element integrals / elements + faces / the faces' share | 2.9e-16 / 2.9e-16 / 4.6e-15 |
| q_velocity, q_numDiff_u, q_cfl | 1.6e-16, 5.3e-16, 8.4e-17 |
| boundary mass flux, ebqe_velocity | 2.2e-16, 1.5e-16 |
| next Newton iterate (one step with the engine's R and J) | 1.1e-13: the solve's floor (cond(J) = 7.7e8; proteus's own J and r through three LU orderings give 1.3e-13 to 9.1e-13) |
| residual at the next iterate | 1.1e-14 |
