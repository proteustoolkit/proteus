Long runs: checkpoints and hot start
====================================

A run archives *frames*, one per output time, and a hot start (``parun -H``) continues the archive from its last
frame. What a frame holds decides what the continuation can be.

What a frame carries
--------------------

* **The solution** (every frame): the dofs, and the fields the deck asks to archive. A hot start from a frame
  restarts from this solution as an initial condition: the history the time integrators, shock capturing and
  subgrid error keep from step to step is rebuilt from it, so the run differs slightly from the uninterrupted one
  (``--hotStartMode restart`` is that, by request).
* **The solver state** (some frames): that history itself, written in place, one copy for the whole archive,
  stamped with the frame it belongs to. A hot start from the frame it is stamped with
  (``--hotStartMode continue``, the default) is the uninterrupted run bit for bit, on the same number of ranks and
  the same partition. It is some 20 times the size of one frame's solution (Torino S0H30 at 1,536 ranks: 34 GB, 19
  GB of it RANS2P's quadrature-point lags), so it is not written every frame.

The solver state is written

* with the final frame of the run;
* with a checkpoint frame (below);
* with every N-th frame, the initial one included, with ``--state-every N`` (``pro run --state-every N``,
  ``RunOptions.stateEvery``). The default, 0, is the two cases above. As the state is stored in place, only the
  latest write is in the archive: after a crash, ``parun -H -t <the time of that frame>`` continues exactly from
  it, and the run loses at most N frames.

A hot start from a frame that carries no state says so on the log and in one ``RuntimeWarning``: ``frame 4 carries
the solution only: the archive holds the solver state of step 6 only``.

Stopping a run before its wall time: ``--wall-limit``
-----------------------------------------------------

A batch job is killed when its wall time runs out, in the middle of a step or of an archive write.
``parun --wall-limit HH:MM:SS`` (also ``D-HH:MM:SS``; ``pro run --wall-limit``) makes the run stop by itself
instead, at a system step boundary, with room left to write a checkpoint:

* the budget counts from the start of parun (``import proteus`` included). ``--wall-limit scheduler`` reads the end
  of the SLURM or PBS job instead (``SLURM_JOB_END_TIME``, else ``squeue -o %L``; ``qstat -f``'s walltime less the
  walltime used), and a duration given inside a batch job is cut at the job's end. Leave slack for the job's own
  start and exit: ``--wall-limit 23:45:00`` in a 24 h job.
* at the end of each system step rank 0 decides whether the next step and a checkpoint still fit:
  ``now + margin > limit``, with ``margin`` = the longest system step so far (archive writes not counted) + the
  time one checkpoint takes to write (for each model, its longest frame and solver-state write so far, measured by
  the run itself). One scalar broadcast hands the answer to every rank, so all stop at the same step. Nothing relies
  on scheduler signals. A run with a limit writes each model's solver state once with its first frame, to time it;
  the writes are on the log at ``-l2`` and above (``Archive: frame 11 of model mcorr at t=0.05 written in ... s,
  solver state in ... s``).
* the stop writes **one frame at that time, with the solver state**, closes the archives and exits with status
  **75** (``WallLimit.CHECKPOINT_EXIT_STATUS``, sysexits' ``EX_TEMPFAIL``). The frame is off the ``dt_output`` grid
  when the stop is mid-interval; that is the frame the next leg reads. The end-of-simulation work (``runAtEOS``) is
  not run, as the run is not over.
* the next leg is ``parun -H`` of the same case on the **same number of ranks and partition** (pin it as for any
  continue hot start). It continues from the checkpoint exactly: the steps, dofs and force and moment histories
  are those of the run that was never stopped (tested at 1 and 2 ranks, mid-interval and at an output time:
  ``test/test_timed_checkpoint.py``).

====== =========================================================================================
status meaning
====== =========================================================================================
0      the run reached its final time
1      the run failed (a system step failed after its retries, or a time went non-finite)
2      a usage error, a malformed ``--wall-limit`` among them (stops before anything runs)
75     stopped for the wall limit with a checkpoint: continue it with ``parun -H``
====== =========================================================================================

Not supported: PUMI mesh adaptivity (it has its own checkpoint frequency; the limit is dropped with a warning) and
runs with no archive (there is nothing to continue from; a warning says so).

Chaining the legs in a job script
---------------------------------

A SLURM script that runs a leg and, on exit 75, submits the next one (``rc`` is made 0 so that
``--dependency=afterok`` is satisfied; use ``qsub -W depend=afterok:$PBS_JOBID`` under PBS):

.. code-block:: bash

    #SBATCH --time=24:00:00
    HOT=""; [ -f flume.h5 ] && HOT="-H"
    srun parun $HOT --wall-limit 23:45:00 flume_so.py; rc=$?
    if [ $rc -eq 75 ]; then sbatch --dependency=afterok:$SLURM_JOB_ID "$0"; rc=0; fi; exit $rc
