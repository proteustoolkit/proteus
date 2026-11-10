#!/usr/bin/env bash
# usage: run.sh
#
# Coupled Richards-TADR manufactured solution: every run behind the tables and
# figures in paper_figures/ (the 2026-11-02 Phase 4 ladder), then the tables and
# figures themselves.  One run is run_mms.sh <run dir> MMS_*=...; the case
# lists are runs/jobs_*.txt, one run per line (MMS_PLAN.md defines T1-T3):
#   jobs_phase4.txt    T1, T2, T3, T3disp, T3bubble, low order and FCT,
#                      path (a), N = 10..160; coupling sweep s_* at N = 40
#   jobs_n320.txt      T2 and T3 at N = 320
#   jobs_dt2.txt, jobs_dt2_n320.txt (via queue_dt2_all.sh)
#                      T3 with dt/2 on each mesh, for the time/space split
# Exact solution, van Genuchten parameters and the sources f_w (flow) and g
# (transport): mms_exact.py.  MMS_JOBS sets the number of concurrent runs (10).
# Detach, which otherwise dies with the session:
#   setsid nohup ./run.sh > run.log 2>&1 < /dev/null & disown
set -u
DECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DECK/runs"
./run_jobs.sh jobs_phase4.txt "${MMS_JOBS:-10}" > phase4.log 2>&1
./run_jobs.sh jobs_n320.txt 4 > n320.log 2>&1
./queue_dt2_all.sh

cd "$DECK"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate tadr-check
T3=$(for N in 10 20 40 80 160 320; do echo runs/a_T3_fct0_N$N runs/a_T3_fct1_N$N; done)
mkdir -p figures
python mms_table.py blocks psi $T3 > figures/tab_mms_t3_psi.tex
python mms_table.py blocks c $T3 > figures/tab_mms_t3_c.tex
python mms_table.py dt2 $T3 > figures/tab_mms_dt2.tex
python mms_table.py sweep runs/s_* > figures/tab_mms_sweep.txt
python mms_plot.py runs    # figures/mms_*.png, .pdf
