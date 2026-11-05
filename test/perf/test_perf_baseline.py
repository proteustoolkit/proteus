"""The on-node baseline scripts keep running: the ceilings in quick mode, the kernel baseline on
one rank and a tiny mesh. They are measurement tools, not regression tests; what this pins is that
they still start, account for every pass, and write the JSON the campaign documents read."""
import json
import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, args, cwd):
    proc = subprocess.run([sys.executable, os.path.join(HERE, script)] + args, cwd=str(cwd),
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    return proc.stdout


def test_machine_ceilings_quick(tmp_path):
    out = run('machine_ceilings.py', ['quick=1', 'out=ceilings.json'], tmp_path)
    assert 'triad GB/s' in out and 'FMA GFLOP/s' in out
    with open(tmp_path / 'ceilings.json') as f:
        c = json.load(f)
    m = c['measurements']['proteus']
    assert m['triad']['1'][0]['per_proc_min_gbps'] > 0
    assert m['fma']['1']['per_proc_min_gflops'] > 0


@pytest.mark.parametrize('kind', ['hex', 'tet'])
def test_kernel_baseline_one_rank(kind, tmp_path):
    run('machine_ceilings.py', ['quick=1', 'out=ceilings.json'], tmp_path)
    out = run('kernel_baseline_mpi.py', [kind, '5', 'repeat=2', 'ceilings=ceilings.json', 'out=baseline.json'],
              tmp_path)
    result, = [line for line in out.splitlines() if line.startswith('RESULT ')]
    assert 'ns_per_element_residual=' in result
    assert 'CEILING stream_per_rank=' in out
    with open(tmp_path / 'baseline.json') as f:
        r = json.load(f)['results'][0]
    assert r['kind'] == kind and r['ranks'] == 1
    assert r['t_residual'] > 0 and r['t_jacobian'] > 0
    passes = r['kernel']['passes']
    # the assembly made C passes, attributed to the phases they ran in
    assert any(k.startswith('residual/element residual | femIntegrals.') for k in passes)
    assert any(k.startswith('jacobian/element jacobian | femIntegrals.') for k in passes)
    assert all(v['bytes'] >= 0 and v['calls'] > 0 for v in passes.values())
    # a phase's C time never exceeds its inclusive time
    phases = r['kernel']['phases']
    for path, v in phases.items():
        c_time = sum(p['seconds'] for k, p in passes.items()
                     if k.split(' | ')[0] == path or k.split(' | ')[0].startswith(path + '/'))
        assert c_time <= v['seconds'] * 1.01 + 1e-6, path


def test_model_baseline_one_rank(tmp_path):
    """A TwoPhaseFlow case, kernel by kernel: damBreak on its stored mesh, one rank."""
    ci = os.path.join(HERE, os.pardir, "TwoPhaseFlow")
    for name in os.listdir(ci):
        if name.startswith("meshDambreak"):
            shutil.copy(os.path.join(ci, name), tmp_path)
    run('machine_ceilings.py', ['quick=1', 'out=ceilings.json'], tmp_path)
    out = run('model_baseline_mpi.py',
              [os.path.join(ci, 'damBreak.py'), 'context=final_time=0.02 dt_output=0.02 he=0.1',
               'ceilings=ceilings.json', 'out=model.json'], tmp_path)
    assert 'RESULT case=damBreak.py' in out and 'CEILING stream_per_rank=' in out
    assert any(line.startswith('MODEL rans3p') and 'jacobian' in line for line in out.splitlines())
    with open(tmp_path / 'model.json') as f:
        r = json.load(f)
    assert r['models'][:2] == ['clsvof', 'rans3p']
    keys = list(r['passes'])
    assert any(k.endswith('rans3p0.rans3pf.calculateResidual') for k in keys)
    assert any(k.endswith('rans3p0.rans3pf.calculateJacobian') for k in keys)
    # the bytes counted are those of the arrays the ArgumentsDict carried
    residual = [v for k, v in r['passes'].items() if k.endswith('rans3pf.calculateResidual')]
    assert all(v['bytes'] > 0 and v['calls'] > 0 for v in residual)


def test_model_baseline_level_set_iterative(tmp_path):
    """damBreakLevelSet with TwoPhaseFlow's iterative solvers (superlu=0), one rank: RANS2P's defaults
    (right-preconditioned GMRES, ASM with PETSc's LU per block, one setup per Newton solve) run, the KSP
    iterations are counted per Newton iteration, and solution= writes the fields and the VOF volume."""
    ci = os.path.join(HERE, os.pardir, "TwoPhaseFlow")
    for name in os.listdir(ci):
        if name.startswith("meshDambreak"):
            shutil.copy(os.path.join(ci, name), tmp_path)
    out = run('model_baseline_mpi.py',
              [os.path.join(ci, 'damBreakLevelSet.py'), 'context=final_time=0.03 dt_output=0.03 he=0.1',
               'warmup=1', 'superlu=0', 'solution=sol', 'out=model.json'], tmp_path)
    ksp = [line for line in out.splitlines() if line.startswith('KSP rans2p')]
    assert len(ksp) == 1 and float(ksp[0].split()[2]) > 0, out[-2000:]
    assert any(line.startswith('INTEGRAL vof') for line in out.splitlines())
    with open(tmp_path / 'model.json') as f:
        r = json.load(f)
    x = r['newton']['rans2p']['per_iteration_extras_mean']
    assert x['ksp its'] > 0 and x['convergence test'] == 0.0  # PETSc's test, not the Python r-true callback
    assert (tmp_path / 'sol.0.npz').exists()


def _gate(monkeypatch, ctl, ack, pattern='femIntegrals.update'):
    sys.path.insert(0, HERE)
    try:
        import accounting
    finally:
        sys.path.remove(HERE)
    monkeypatch.setenv('PROTEUS_PERF_PASS', pattern)
    monkeypatch.setenv('PROTEUS_PERF_CTL', ctl)
    monkeypatch.setenv('PROTEUS_PERF_ACK', ack)
    return accounting.PerfGate()


def _fake_perf(ctl, ack, words):
    """What `perf stat --control fifo:CTL,ACK` does with the fifos: read a word, answer ack."""
    import threading

    def serve():
        with open(ctl) as c, open(ack, 'w', buffering=1) as a:
            for line in c:
                words.append(line.strip())
                a.write('ack\n')
                a.flush()
    t = threading.Thread(target=serve, daemon=True)
    t.start()
    return t


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='needs named pipes')
def test_perf_gate_one_perf_per_rank(tmp_path, monkeypatch):
    """With {rank} in the fifo paths every rank gates its own perf, and counts its own calls."""
    gates, words = [], {}
    for rank in (0, 1):
        for name in ('ctl', 'ack'):
            os.mkfifo(str(tmp_path / ('%s.%d' % (name, rank))))
        words[rank] = []
        _fake_perf(str(tmp_path / ('ctl.%d' % rank)), str(tmp_path / ('ack.%d' % rank)), words[rank])
        gate = _gate(monkeypatch, str(tmp_path / 'ctl.{rank}'), str(tmp_path / 'ack.{rank}'))
        gate.open(rank, 2)
        gates.append(gate)
    for rank, gate in enumerate(gates):
        for _ in range(rank + 1):
            gate.enable()
            gate.disable()
    assert [g.calls for g in gates] == [1, 2]
    assert words == {0: ['enable', 'disable'], 1: ['enable', 'disable'] * 2}


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='needs named pipes')
def test_perf_gate_shared_fifos_gate_on_rank_0_only(tmp_path, monkeypatch, capsys):
    """One fifo pair on 2 ranks: rank 1 never touches it (its disable would end rank 0's window), and
    rank 0 says that perf must be counting rank 0 alone."""
    ctl, ack = str(tmp_path / 'ctl'), str(tmp_path / 'ack')
    os.mkfifo(ctl)
    os.mkfifo(ack)
    words = []
    _fake_perf(ctl, ack, words)
    rank1 = _gate(monkeypatch, ctl, ack)
    rank1.open(1, 2)
    rank1.enable()
    rank1.disable()
    assert (rank1.ctl, rank1.calls) == (None, 0)
    rank0 = _gate(monkeypatch, ctl, ack)
    rank0.open(0, 2)
    assert 'only rank 0 gates' in capsys.readouterr().err
    rank0.enable()
    rank0.disable()
    assert rank0.calls == 1 and words == ['enable', 'disable']
    assert rank0.matches('femIntegrals.updateDiffusion_weak_sd') and not rank0.matches('cfemIntegrals.other')


def test_perf_gate_is_off_without_all_three_variables(monkeypatch):
    for name in ('PROTEUS_PERF_PASS', 'PROTEUS_PERF_CTL', 'PROTEUS_PERF_ACK'):
        monkeypatch.delenv(name, raising=False)
    sys.path.insert(0, HERE)
    try:
        import accounting
    finally:
        sys.path.remove(HERE)
    gate = accounting.PerfGate()
    assert not gate.on and not gate.matches('femIntegrals.updateDiffusion_weak_sd')
