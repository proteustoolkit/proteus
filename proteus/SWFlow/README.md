# Proteus -- SWFlow API

The SWFlow API contains solvers for the the 2D nonlinear Shallow Water Equations, and the 2D dispersive Serre-Green-Naghdi equations (also known as Green-Naghdi equations, nonlinear Boussinesq, etc.). Both models are
numerically solved using continuous, linear finite elements as seen in
[Guermond, *et al* 2018](https://doi.org/10.1137/17M1156162), [Guermond *et al* 2019](https://doi.org/10.1016/j.jcp.2019.108917) and [Guermond *et al* 2020].

<!-- The Shallow Water equations are a set of partial differential equations that form a hyperbolic system. They can be used to describe a body of water evolving under the action of gravity under the assumption that the deformations of the free surface are small compared to the water height.

The relaxed dispersive Serre--Saint-Venant equations are a set of partial differential equations that form a
hyperbolic system and are an O(h) approximation to the traditional Green-Naghdi equations (where h is the mesh size). -->

<!-- The different tests demonstrate the respective models abilities to handle wet/dry states, reciprocate "real world" problems (see malpasset, colorado_river),
propagate solitary waves, etc. -->

## Running a case from a ymf spec

A `.ymf` spec runs through `pro run SPEC.ymf` (proteus.Dispatch maps its sub-problems onto SW2DCV or GN_SW2DCV and
proteus.DispatchRun builds an `SWFlowProblem` with these p and n modules). Nothing here is deprecated yet: the decks are
the oracle the specs are compared with, bitwise, in `test/dispatch`. The cases a spec equals today: `circular_damBreak.py`
(`sw_model` 0 and 1, structured mesh), `dam3Bumps.py` (structured with `mesh.cells` per axis, unstructured, a stored mesh,
open boundaries). Decks that impose a state on part of the boundary, wave generation, `atan` in the bed, or a hot start
still need a ymf word the spec does not have, and keep running as decks.

## Getting Started

These instructions will help you run and create new SWFlow tests on your local machine.

### Prerequisites

The only prerequisite that you need is a working copy of Proteus. If you have yet to build Proteus, go to your Proteus directory and run

```
make develop
```

## Running the tests

Each test is located in `proteus/tests/SWFlow/` and is a python file named something like `test_problem.py` (examples are `dam3Bumps.py`, `solitary_reef.py`, etc.). This is the file that sets up the computational domain, initial conditions,
boundary conditions, and other problem specific options.

Each `test_problem.py` file has its own set of default context options. These include choosing which PDE model to use, final run time of the simulation, refinement level of mesh, etc. For example, in `proteus/tests/dam3Bumps.py`, a dam break problem with 3 conical obstacles, we see the following set of default options:

```
opts = Context.Options([
    ('sw_model', 0, "sw_model = {0,1} for {SWEs,DSWEs}"),
    ("final_time", 20.0, "Final time for simulation"),
    ("dt_output", 0.1, "Time interval to output solution"),
    ("refinement", 4, "Level of refinement"),
    ("structured", True, "Structured or unstructured mesh"),
    ("he", 0.5, "Mesh size for unstructured mesh"),
    ("cfl", 0.33, "Desired CFL restriction"),
    ("reflecting_BCs", True, "Use reflecting BCs"),
    ("mannings", 0.02, "Mannings roughness coefficient")
])

```
Here `sw_model` refers to which PDE model is chosen: ``0`` for the Shallow Water Equations and ``1`` for the dispersive Serre--Saint-Venant (or "Dispersive SWEs"). The rest of the options are straight forward.

To run a test on a single processor, one can use the following commands:

```
parun --SWEs -l1 -v dam3Bumps.py
```
or
```
runSWEs.py -l1 -v -f dam3Bumps.py
```

which will execute the solver with the above default context options. The `-l1` flag controls the amount of output during the execution; you can choose any number between 1 and 10. The `-v` flag stands for verbose. (See the `proteus/parun.py` or `proteus/runSWEs.py` files for more details on the different flag options for `parun` and `runSWEs.py`.)

To use more than one processor, execute the following commands:

```
mpiexec -np 4 parun --SWEs -l1 -v dam3Bumps.py
```
or
```
mpiexec -np 4 runSWEs.py -l1 -v -f dam3Bumps.py
```
which runs the `dam3Bumps.py` problem on 4 processors.

To use options other than the default options, one can add these in the run command as follows:

```
parun --SWEs -l1 -v dam3Bumps.py -C "sw_model=1 final_time=100"
```

which switches the model to the dispersive model and increases the final time to 100 seconds.

## Creating a test

Creating a new test is as simple as copying one of the existing files,
and modifying appropriately to fit your needs.

Currently, most of the tests are set up to use a rectangular domain with triangle elements created by a function in `proteus/SpatialTool.py`. To see examples with slightly more complicated domains, take a look at `mach_flow.py` and `obstacle_flow.py`. The SWFlow API can be used with meshes created by Triangle and Gmsh. The "real world" tests such as the Malpesset problem use meshes created by the software SMS (https://www.aquaveo.com/software/sms-surface-water-modeling-system-introduction) in the ADH format. If you would like to use your own mesh, but aren't sure how to use it, please contact us and we can gladly help.

<!-- ## Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
``` -->

<!-- ## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds -->

## Authors

* **Dr. Manuel Quezada De Luna**
* **Dr. Chris Kees**
* **Eric Tovar(*)**

## Contributing

Please read [CONTRIBUTING.md](https://github.com/proteustoolkit/proteus/blob/master/CONTRIBUTING.md) for details on the Proteus code of conduct, and the process for submitting pull requests to us.

See also the list of Proteus [contributors](https://github.com/proteustoolkit/proteus/blob/master/CONTRIBUTORS.md) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## References

* J.-L. Guermond, M. Quezada de Luna, B. Popov, C. Kees, and M. Farthing. Well-balanced second-order finite element approximation of the shallow water equations with friction. SIAM Journal on Scientific Computing, 40(6):A3873– A3901, 2018.
* J.-L. Guermond, B. Popov, E. Tovar, C. Kees. Robust explicit relaxation technique for solving the Green-Naghdi equations. Journal of Computational Physics,
Volume 399, 2019, 108917.
* J.-L. Guermond, B. Popov, E. Tovar, C. Kees. Hyperbolic relaxation technique for solving the dispersive Saint-Venant shallow water equations with topography. Submitted, 2020.

<!-- ## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc -->
