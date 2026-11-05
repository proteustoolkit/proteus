"""TwoStageNewton: a failed first stage fails the solve (the step controller then cuts the step) instead of being
reset by the second stage's solve."""
from proteus import NonlinearSolvers
from proteus.NonlinearSolvers import Newton, TwoStageNewton


class _Model(object):
    useTwoStageNewton = True
    stage = 0


def _solver(monkeypatch, failing_stages):
    stages = []

    def solve(self, u, r=None, b=None, par_u=None, par_r=None):
        stages.append(self.F.stage)
        self.failedFlag = self.F.stage in failing_stages
        return self.failedFlag

    monkeypatch.setattr(Newton, "solve", solve)
    solver = TwoStageNewton.__new__(TwoStageNewton)
    solver.F = _Model()
    solver.failedFlag = False
    return solver, stages


def test_a_failed_first_stage_fails_the_solve_and_skips_the_second(monkeypatch):
    solver, stages = _solver(monkeypatch, failing_stages=(1,))
    assert solver.solve(None) is True
    assert stages == [1]


def test_a_failed_second_stage_fails_the_solve(monkeypatch):
    solver, stages = _solver(monkeypatch, failing_stages=(2,))
    assert solver.solve(None) is True
    assert stages == [1, 2]


def test_two_converged_stages(monkeypatch):
    solver, stages = _solver(monkeypatch, failing_stages=())
    assert solver.solve(None) is False
    assert stages == [1, 2]
