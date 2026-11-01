#!/usr/bin/env bash
#
# Hornung-Messing convergence study in the layout of Dielmann, Kuzmin & Kees
# (JNMA), Tables 1-2: h = 1/2 ... 1/64 on the unit square, T = 0.05,
#   theta = 1   : dt = 0.2 h^2  (low order, Galerkin, FCT)
#   theta = 1/2 : dt = 0.1 h    (Galerkin, FCT; the low order is always BE)
#
#   ./run_hm_convergence.sh                    # everything, then the tables
#   SCHEMES="FCT" THETAS="1.0" ./run_hm_convergence.sh
#   NNS="3 5 9" ./run_hm_convergence.sh        # coarse levels only
#   TABLE_ONLY=1 ./run_hm_convergence.sh
#   MESH=q1 ./run_hm_convergence.sh            # Q1 on squares (the JNMA element)
#
# P1 (default): runs/theta<theta>/<scheme>/nn<nn>/, tables convergence_theta<theta>.txt.
# Q1:           runs_q1/...,                       tables convergence_q1_theta<theta>.txt.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNS="${NNS:-3 5 9 17 33 65}"
THETAS="${THETAS:-1.0 0.5}"
TABLE_ONLY="${TABLE_ONLY:-0}"
MESH="${MESH:-p1}"
if [ "$MESH" = "q1" ]; then RUNS="runs_q1"; TAG="convergence_q1"; MOPT="quad=True"
else RUNS="runs"; TAG="convergence"; MOPT="quad=False"; fi
T=0.05

scheme_opts() {
    case "$1" in
        LO)       echo "STABILIZATION_TYPE=2 FCT=False" ;;
        Galerkin) echo "STABILIZATION_TYPE=0 FCT=False" ;;
        FCT)      echo "STABILIZATION_TYPE=2 FCT=True" ;;
        IFCT)     echo "STABILIZATION_TYPE=2 FCT=True fct_type='implicit'" ;;
        *) echo "unknown scheme '$1'" >&2; return 1 ;;
    esac
}
ndt_for() {  # $1 = theta, $2 = nn  ->  number of steps
    python3 -c "h=1.0/($2-1); print(max(1, round($T/(0.2*h*h) if float('$1')==1.0 else $T/(0.1*h))))"
}

for th in $THETAS; do
    if [ "$th" = "1.0" ]; then dflt="LO Galerkin FCT IFCT"; else dflt="Galerkin FCT IFCT"; fi
    for s in ${SCHEMES:-$dflt}; do
        [ "$s" = "LO" ] && [ "$th" != "1.0" ] && continue
        dirs=()
        for nn in $NNS; do
            d="$HERE/$RUNS/theta$th/$s/nn$nn"
            dirs+=("$d")
            [ "$TABLE_ONLY" = "1" ] && continue
            ndt=$(ndt_for "$th" "$nn")
            ctx="$(scheme_opts "$s") nn=$nn nDT=$ndt T=$T theta_time=$th $MOPT"
            mkdir -p "$d"
            cp "$HERE/hm_2d_p.py" "$HERE/hm_2d_n.py" "$d/"
            echo "nn=$nn nDT=$ndt T=$T theta=$th scheme=$s mesh=$MESH" > "$d/context.txt"
            echo "=== theta=$th $s nn=$nn nDT=$ndt"
            ( cd "$d" && parun hm_2d_p.py hm_2d_n.py -l 3 -v -C "$ctx" > run.log 2>&1 ) \
                || echo "!!! FAILED: $d/run.log" >&2
        done
        {
            echo "# theta=$th scheme=$s :  h  dt  ||e||_L2  order"
            python "$HERE/hm_l2_error.py" "${dirs[@]}"
        } | tee -a "$HERE/${TAG}_theta$th.txt.new"
    done
    mv "$HERE/${TAG}_theta$th.txt.new" "$HERE/${TAG}_theta$th.txt"
done
