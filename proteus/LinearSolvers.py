r"""
A hierarchy of classes for linear algebraic system solvers.

.. inheritance-diagram:: proteus.LinearSolvers
   :parts: 1
"""
from .LinearAlgebraTools import *
from . import LinearAlgebraTools as LAT
from . import FemTools
from . import clapack
from . import superluWrappers
from . import TransportCoefficients
from . import cfemIntegrals
from . import Quadrature
from .petsc_optional import PETSc as p4pyPETSc, HAVE_PETSC4PY
from math import *
import math
from .Profiling import logEvent
from .mprans import cArgumentsDict

class LinearSolver(object):
    """ The base class for linear solvers.

    Arugments
    ---------
    L : :class:`proteus.superluWrappers.SparseMatrix`
        This is the system matrix.
    """
    def __init__(self,
                 L,
                 rtol_r  = 1.0e-4,
                 atol_r  = 1.0e-16,
                 rtol_du = 1.0e-4,
                 atol_du = 1.0e-16,
                 maxIts  = 100,
                 norm = l2Norm,
                 convergenceTest = 'r',
                 computeRates = True,
                 printInfo = False):
        self.solverName = "Base Class"
        self.par_u_current = None
        self.L = L
        self.n = L.shape[0]
        self.du = Vec(self.n)
        self.rtol_r=rtol_r
        self.atol_r=atol_r
        self.rtol_du=rtol_du
        self.atol_du=atol_du
        self.maxIts=maxIts
        self.its=0
        self.solveCalls = 0
        self.recordedIts=0
        self.solveCalls_failed = 0
        self.recordedIts_failed=0
        self.rReductionFactor=0.0
        self.duReductionFactor=0.0
        self.rReductionFactor_avg=0.0
        self.duReductionFactor_avg=0.0
        self.rReductionOrder=0.0
        self.rReductionOrder_avg=0.0
        self.duReductionOrder=0.0
        self.duReductionOrder_avg=0.0
        self.ratio_r_current = 1.0
        self.ratio_r_solve = 1.0
        self.ratio_du_solve = 1.0
        self.last_log_ratio_r = 1.0
        self.last_log_ratior_du = 1.0
        self.convergenceTest = convergenceTest
        self.computeRates = computeRates
        self.computeEigenvalues=False
        self.printInfo = printInfo
        self.norm = l2Norm
        self.convergenceHistoryIsCorrupt=False
        self.norm_r0=0.0
        self.norm_r=0.0
        self.norm_du=0.0
        self.r=None
        self.leftEigenvectors=None
        self.rightEigenvectors=None
        self.eigenvalues_r=None
        self.eigenvalues_i=None
        self.work=None
        #need some information about parallel setup?
        self.par_fullOverlap = True #whether or not partitioning has full overlap
        #for petsc interface
        self.xGhosted = None
        self.b=None
    def setResTol(self,rtol,atol):
        self.rtol_r = rtol
        self.atol_r = atol
    def prepare(self,b=None,newton_its=None):
        pass
    def solve(self,u,r=None,b=None,par_u=None,par_b=None):
        pass
    def calculateEigenvalues(self):
        pass
    def refreshGhosts(self,par_u):
        """Make the ghost entries of u agree with their owners again.

        The smoothers compute a correction for the unknowns this rank owns and
        subtract it from the whole local vector, ghosts included -- but a ghost
        entry's correction belongs to whichever rank owns it, so the values
        left there are wrong. The next residual reads them (the subdomain
        matrix couples owned rows to ghost columns), so they have to be
        replaced by the owners' values before computeResidual runs.

        Serial runs pass par_u=None and this does nothing.
        """
        #Transport builds par vectors several ways, and not all of them are
        #ghosted -- a vector with no ghost entries has nothing to refresh, and
        #asking PETSc to scatter it raises "Vector is not ghosted".
        if par_u is not None and getattr(par_u,'nghosts',None) is not None:
            par_u.scatter_forward_insert()
    def refreshResidualGhosts(self,par_u,r):
        """Give the residual's ghost entries their owners' values.

        computeResidual forms r = L u - b with the subdomain matrix, which is
        correct only for the rows this rank owns; a ghost row is assembled from
        a row the subdomain holds partially, so r there is meaningless.

        Jacobi and Gauss-Seidel never read it -- their update for an owned
        unknown uses that unknown's own residual entry. A Schwarz smoother
        does: the star it solves around an owned node takes its right-hand side
        from every node in the star, ghosts included. So those entries have to
        come from their owners before the local solve, or the iteration is
        driven by noise.
        """
        if par_u is None or getattr(par_u,'nghosts',None) is None:
            return
        par_r = getattr(self,'par_r_work',None)
        if par_r is None or par_r.proteus_array is not r:
            if r.size != par_u.proteus_array.size:
                return
            par_r = self.par_r_work = par_u.duplicate_with_array(r)
        par_r.scatter_forward_insert()
    def computeResidual(self,u,r,b,initialGuessIsZero=False):
        if initialGuessIsZero:
            r[:]=b
            r*=(-1.0)
        else:
            if type(self.L).__name__ == 'ndarray':
                r[:] = numpy.dot(u,self.L)
            elif type(self.L).__name__ == 'SparseMatrix':
                self.L.matvec(u,r)
            r-=b
    def solveNorm(self,r):
        """The norm the convergence test should use.

        self.norm runs over the whole local vector, which in parallel includes
        the ghost rows -- rows the subdomain holds only partially, where the
        residual is meaningless. Counting them makes the reported norm both
        wrong and rank-dependent, so the test either never fires or fires at
        the wrong point. Over the owned entries, reduced across ranks, it is
        the same number every decomposition computes and the same one a serial
        run computes.
        """
        par_u = self.par_u_current
        if par_u is None or getattr(par_u,'nghosts',None) is None:
            return self.norm(r)
        owned = par_u.dim_proc
        return sqrt(globalSum(float(numpy.dot(r[:owned],r[:owned]))))
    def solveInitialize(self,u,r,b,initialGuessIsZero=True,par_u=None):
        self.par_u_current = par_u
        if r is None:
            if self.r is None:
                self.r = Vec(self.n)
            r=self.r
        else:
            self.r=r
        if b is None:
            if self.b is None:
                self.b = Vec(self.n)
            b=self.b
        else:
            self.b=b
        self.computeResidual(u,r,b,initialGuessIsZero)
        self.its = 0
        self.norm_r0 = self.norm(r)
        self.norm_r = self.norm_r0
        self.ratio_r_solve = 1.0
        self.ratio_du_solve = 1.0
        self.last_log_ratio_r = 1.0
        self.last_log_ratior_du = 1.0
        self.convergenceHistoryIsCorrupt=False
        return (r,b)
    def computeConvergenceRates(self):
        if self.convergenceHistoryIsCorrupt:
            return
        else:
            if self.its > 0:
                if self.norm_r < self.lastNorm_r:
                    self.ratio_r_current = self.norm_r/self.lastNorm_r
                else:
                    self.convergenceHistoryIsCorrupt=True
                    return
                if self.ratio_r_current > 1.0e-100:
                    log_ratio_r_current = log(self.ratio_r_current)
                else:
                    self.convergenceHistoryIsCorrupt
                    return
                self.ratio_r_solve *= self.ratio_r_current
                self.rReductionFactor = pow(self.ratio_r_solve, 1.0/self.its)
                if self.its > 1:
                    self.rReductionOrder = log_ratio_r_current/ \
                                           self.last_log_ratio_r
                    if self.norm_du < self.lastNorm_du:
                        ratio_du_current = self.norm_du/self.lastNorm_du
                    else:
                        self.convergenceHistoryIsCorrupt=True
                        return
                    if ratio_du_current > 1.0e-100:
                        log_ratio_du_current = log(ratio_du_current)
                    else:
                        self.convergenceHistoryIsCorrupt=True
                        return
                    self.ratio_du_solve *= ratio_du_current
                    self.duReductionFactor = pow(self.ratio_du_solve,
                                                 1.0/(self.its-1))
                    if self.its > 2:
                        self.duReductionOrder = log_ratio_du_current/ \
                                                self.last_log_ratio_du
                    self.last_log_ratio_du = log_ratio_du_current
                self.last_log_ratio_r = log_ratio_r_current
                self.lastNorm_du = self.norm_du
            self.lastNorm_r = self.norm_r
    def converged(self,r):
        convergedFlag = False
        self.norm_r = self.solveNorm(r)
        self.norm_du = self.norm(self.du)
        if self.computeRates ==  True:
            self.computeConvergenceRates()
        if self.convergenceTest == 'its':
            if self.its == self.maxIts:
                convergedFlag = True
        elif self.convergenceTest == 'r':
            if (self.its != 0 and
                self.norm_r < self.rtol_r*self.norm_r0 + self.atol_r):
                convergedFlag = True
        if convergedFlag == True and self.computeRates == True:
            self.computeAverages()
        if self.printInfo == True:
            print(self.info())
        return convergedFlag
    def failed(self):
        failedFlag = False
        if self.its == self.maxIts:
            self.solveCalls_failed+=1
            self.recordedIts_failed+=self.its
            failedFlag = True
        self.its+=1
        return failedFlag
    def computeAverages(self):
        self.recordedIts+=self.its
        if self.solveCalls == 0:
            self.rReductionFactor_avg = self.rReductionFactor
            self.duReductionFactor_avg = self.duReductionFactor
            self.rReductionOrder_avg = self.rReductionOrder
            self.duReductionOrder_avg = self.duReductionOrder
            self.solveCalls+=1
        else:
            self.rReductionFactor_avg*=self.solveCalls
            self.rReductionFactor_avg+=self.rReductionFactor
            self.duReductionFactor_avg*=self.solveCalls
            self.duReductionFactor_avg+=self.duReductionFactor
            self.rReductionOrder_avg*=self.solveCalls
            self.rReductionOrder_avg+=self.rReductionOrder
            self.duReductionOrder_avg*=self.solveCalls
            self.duReductionOrder_avg+=self.duReductionOrder
            self.solveCalls +=1
            self.rReductionFactor_avg/=self.solveCalls
            self.duReductionFactor_avg/=self.solveCalls
            self.rReductionOrder_avg/=self.solveCalls
            self.duReductionOrder_avg/=self.solveCalls
    def info(self):
        self.infoString  = "************Start Linear Solver Info************\n"
        self.infoString += "its                   =  %i \n" % self.its
        self.infoString += "r reduction factor    = %12.5e\n" % self.rReductionFactor
        self.infoString += "du reduction factor   = %12.5e\n" % self.duReductionFactor
        self.infoString += "r reduction order     = %12.5e\n" % self.rReductionOrder
        self.infoString += "du reduction order    = %12.5e\n" % self.duReductionOrder
        self.infoString += "<r reduction factor>  = %12.5e\n" % self.rReductionFactor_avg
        self.infoString += "<du reduction factor> = %12.5e\n" % self.duReductionFactor_avg
        self.infoString += "<r reduction order>   = %12.5e\n" % self.rReductionOrder_avg
        self.infoString += "<du reduction order>  = %12.5e\n" % self.duReductionOrder_avg
        self.infoString += "total its             =  %i \n" % self.recordedIts
        self.infoString += "total its             =  %i \n" % self.recordedIts
        self.infoString += "solver calls          =  %i \n" % self.solveCalls
        self.infoString += "failures              =  %i \n" % self.solveCalls_failed
        self.infoString += "failed its            =  %i \n" % self.recordedIts_failed
        self.infoString += "maxIts                =  %i \n" % self.maxIts
        self.infoString += "convergenceTest       =  %s \n" % self.convergenceTest
        self.infoString += "atol_r                = %12.5e \n" % self.atol_r
        self.infoString += "rtol_r                = %12.5e \n" % self.rtol_r
        self.infoString += "norm(r0)              = %12.5e \n" % self.norm_r0
        self.infoString += "norm(r)               = %12.5e \n" % self.norm_r
        self.infoString += "norm(du)              = %12.5e \n" % self.norm_du
        if self.convergenceHistoryIsCorrupt:
            self.infoString += "HISTORY IS CORRUPT!!! \n"
        self.infoString += "************End Linear Solver Info************\n"
        return self.infoString
    def printPerformance(self):
        pass

    #petsc preconditioner interface
    def setUp(self, pc):
        self.prepare()
    def apply(self,pc,x,y):
        if self.xGhosted is None:
            self.xGhosted = self.par_b.duplicate()
            self.yGhosted = self.par_b.duplicate()
        self.xGhosted.setArray(x.getArray())
        self.xGhosted.ghostUpdateBegin(p4pyPETSc.InsertMode.INSERT,p4pyPETSc.ScatterMode.FORWARD)
        self.xGhosted.ghostUpdateEnd(p4pyPETSc.InsertMode.INSERT,p4pyPETSc.ScatterMode.FORWARD)
        self.yGhosted.zeroEntries()
        with self.yGhosted.localForm() as ylf,self.xGhosted.localForm() as xlf:
            self.solve(u=ylf.getArray(),b=xlf.getArray(),initialGuessIsZero=True)
        y.setArray(self.yGhosted.getArray())

class LU(LinearSolver):
    """
    A wrapper for pysparse's wrapper for superlu.
    """
    def __init__(self,L,computeEigenvalues=False,computeEigenvectors=None):
        import copy
        LinearSolver.__init__(self,L)
        if type(L).__name__ == 'SparseMatrix':
            self.sparseFactor = superluWrappers.SparseFactor(self.n)
        elif type(L).__name__ == 'ndarray':#mwf was array
            self.denseFactor = clapack.DenseFactor(self.n)
        self.solverName = "LU"
        self.computeEigenvalues = computeEigenvalues or (computeEigenvectors is not None)
        if computeEigenvectors in ['left','both']:
            self.leftEigenvectors=numpy.zeros((self.n,self.n),'d')
            self.JOBVL='V'
        else:
            self.JOBVL='N'
        if computeEigenvectors in ['right','both']:
            self.rightEigenvectors=numpy.zeros((self.n,self.n),'d')
            self.JOBVR='V'
        else:
            self.JOBVR='N'
        if computeEigenvalues or computeEigenvectors is not None:
            self.Leig=copy.deepcopy(L)
            self.work=numpy.zeros((self.n*5,),'d')
            self.eigenvalues_r = numpy.zeros((self.n,),'d')
            self.eigenvalues_i = numpy.zeros((self.n,),'d')
    def prepare(self,
                b=None,
                newton_its=None):
        if type(self.L).__name__ == 'SparseMatrix':
            superluWrappers.sparseFactorPrepare(self.L,self.sparseFactor)
        elif type(self.L).__name__ == 'ndarray':
            if self.computeEigenvalues:
                self.Leig[:]=self.L
                self.calculateEigenvalues()
            clapack.denseFactorPrepare(self.n,
                                       self.L,
                                       self.denseFactor)
        #
    def solve(self,u,r=None,b=None,par_u=None,par_b=None,initialGuessIsZero=False):
        (r,b) = self.solveInitialize(u,r,b,initialGuessIsZero,par_u=par_u)
        self.du[:]=u
        self.converged(r)
        self.failed()
        u[:]=b
        if type(self.L).__name__ == 'SparseMatrix':
            superluWrappers.sparseFactorSolve(self.sparseFactor,u)
        elif type(self.L).__name__ == 'ndarray':
            clapack.denseFactorSolve(self.n,
                                     self.L,
                                     self.denseFactor,
                                     u)
        self.computeResidual(u,r,b)
        self.du -= u
        self.converged(r)
    def calculateEigenvalues(self):
        if type(self.L).__name__ == 'ndarray':
            clapack.denseCalculateEigenvalues(self.JOBVL,
                                              self.JOBVR,
                                              self.n,
                                              self.Leig,
                                              self.n,
                                              self.eigenvalues_r,
                                              self.eigenvalues_i,
                                              self.leftEigenvectors,
                                              self.n,
                                              self.rightEigenvectors,
                                              self.n,
                                              self.work,
                                              5*self.n)
            eigen_mags = numpy.sqrt(self.eigenvalues_r**2 + self.eigenvalues_i**2)
            logEvent("Minimum eigenvalue magnitude"+repr(eigen_mags.min()))
            logEvent("Maximum eigenvalue magnitude"+repr(eigen_mags.max()))
            logEvent("Minimum real part of eigenvalue "+repr(self.eigenvalues_r.min()))
            logEvent("Maximum real part of eigenvalue "+repr(self.eigenvalues_r.max()))
            logEvent("Minimum complex part of eigenvalue "+repr(self.eigenvalues_i.min()))
            logEvent("Maximum complex part of eigenvalue "+repr(self.eigenvalues_i.max()))

class RedundantLU(LinearSolver):
    """An exact solve of the global system, factored on every rank, with no PETSc.

    The coarse level of a parallel multigrid cycle needs the global coarse
    system solved exactly; factoring each rank's subdomain matrix (LU) is
    block-Jacobi instead. This does what PETSc's redundant preconditioner does
    for the PETSc path (see mgmCoarseSolver): every rank contributes its owned
    rows, in global numbers, every rank assembles and factors the whole matrix
    with SuperLU, and every rank solves the whole system with the gathered
    right-hand side. The solution is then known everywhere, so the ghost
    entries come out right with no exchange. That is cheap only because the
    coarse level is small, which is what it is for.

    par_L carries the layout (LinearAlgebraTools.ParMat_mpi4py): owned row
    count, block size, global size and the subdomain-to-global map. Only the
    plain layout (owned entries first, components interleaved) is supported.
    """
    def __init__(self,L,par_L):
        LinearSolver.__init__(self,L)
        from . import Comm
        if getattr(par_L,'nzval_proteus2petsc',None) is not None:
            raise NotImplementedError(
                "RedundantLU needs the plain parallel layout (owned entries first, components "
                "interleaved); mixed spaces are numbered by a permutation it does not apply. "
                "Use petsc4py for this problem.")
        self.mpi = Comm.get().comm.tompi4py()
        bs = max(1,par_L.blockSize)
        self.nOwned = bs*par_L.par_n
        self.N = bs*par_L.par_N
        s2g = numpy.asarray(par_L.subdomain2global)
        if bs > 1:
            s2g = (bs*numpy.repeat(s2g,bs) + numpy.tile(numpy.arange(bs),s2g.size))
        assert s2g.size == self.n, (
            "the subdomain matrix has %i rows but the layout describes %i" % (self.n,s2g.size))
        self.localToGlobal = s2g.astype(numpy.int64)
        self.sparseFactor = superluWrappers.SparseFactor(self.N)
        self.solverName = "RedundantLU"

    def prepare(self,b=None,newton_its=None):
        import scipy.sparse
        rowptr,colind,nzval = self.L.getCSRrepresentation()
        nnzOwned = int(rowptr[self.nOwned])
        rows = numpy.repeat(self.localToGlobal[:self.nOwned],numpy.diff(rowptr[:self.nOwned+1]))
        cols = self.localToGlobal[colind[:nnzOwned]]
        pieces = self.mpi.allgather((rows,cols,numpy.array(nzval[:nnzOwned])))
        A = scipy.sparse.csr_matrix((numpy.concatenate([p[2] for p in pieces]),
                                     (numpy.concatenate([p[0] for p in pieces]),
                                      numpy.concatenate([p[1] for p in pieces]))),
                                    shape=(self.N,self.N))
        A.sum_duplicates()
        A.sort_indices()
        assert numpy.all(numpy.diff(A.indptr) > 0), "a global row of the coarse matrix is empty"
        #The factor keeps state tied to the matrix it was first prepared with
        #(SuperLU reuses the column permutation and the storage it points to),
        #so the global matrix is built once and refactored with new values in
        #place, as LU does with its own matrix. Handing it a fresh matrix on
        #every prepare gave a wrong factorization once the first matrix's
        #arrays had been freed and reused. A changed pattern starts over.
        samePattern = (getattr(self,'globalL',None) is not None and
                       numpy.array_equal(self.globalRowptr,A.indptr) and
                       numpy.array_equal(self.globalColind,A.indices))
        if samePattern:
            self.globalNzval[:] = A.data
        else:
            self.globalNzval = numpy.ascontiguousarray(A.data,'d')
            self.globalColind = numpy.ascontiguousarray(A.indices,'i')
            self.globalRowptr = numpy.ascontiguousarray(A.indptr,'i')
            self.globalL = superluWrappers.SparseMatrix(self.N,self.N,self.globalNzval.size,
                                                        self.globalNzval,self.globalColind,
                                                        self.globalRowptr)
            self.sparseFactor = superluWrappers.SparseFactor(self.N)
        superluWrappers.sparseFactorPrepare(self.globalL,self.sparseFactor)

    def solve(self,u,r=None,b=None,par_u=None,par_b=None,initialGuessIsZero=False):
        (r,b) = self.solveInitialize(u,r,b,initialGuessIsZero,par_u=par_u)
        self.du[:]=u
        self.converged(r)
        self.failed()
        pieces = self.mpi.allgather((self.localToGlobal[:self.nOwned],numpy.array(b[:self.nOwned])))
        x = numpy.zeros((self.N,),'d')
        for (indices,values) in pieces:
            x[indices] = values
        superluWrappers.sparseFactorSolve(self.sparseFactor,x)
        u[:] = x[self.localToGlobal]
        self.computeResidual(u,r,b)
        self.du -= u
        self.converged(r)

class KSP_petsc4py(LinearSolver):
    """ A class that interfaces Proteus with PETSc KSP. """
    def __init__(self,L,par_L,
                 rtol_r  = 1.0e-4,
                 atol_r  = 1.0e-16,
                 maxIts  = 100,
                 norm    = l2Norm,
                 convergenceTest = 'r',
                 computeRates = True,
                 printInfo = False,
                 prefix=None,
                 Preconditioner=None,
                 connectionList=None,
                 linearSolverLocalBlockSize=1,
                 preconditionerOptions = None):
        """ Initialize a petsc4py KSP object.

        Parameters
        -----------
        L : :class: `.superluWrappers.SparseMatrix`
        par_L :  :class: `.LinearAlgebraTools.ParMat_petsc4py`
        rtol_r : float
        atol_r : float
        maxIts : int
        norm :   norm type
        convergenceTest :
        computeRates: bool
        printInfo : bool
        prefix : bool
        Preconditioner : :class: `.LinearSolvers.KSP_Preconditioner`
        connectionList :
        linearSolverLocalBlockSize : int
        preconditionerOptions : tuple
            A list of optional preconditioner settings.
        """
        LinearSolver.__init__(self,
                              L,
                              rtol_r=rtol_r,
                              atol_r=atol_r,
                              maxIts=maxIts,
                              norm = l2Norm,
                              convergenceTest=convergenceTest,
                              computeRates=computeRates,
                              printInfo=printInfo)
        assert type(L).__name__ == 'SparseMatrix', "petsc4py PETSc can only be called with a local sparse matrix"
        assert isinstance(par_L,ParMat_petsc4py)
        self.pccontext = None
        self.preconditioner = None
        self.preconditionerOptions = preconditionerOptions
        self.pc = None
        self.solverName  = "PETSc"
        self.par_fullOverlap = True
        self._dirichlet_zero_positions = None
        self.par_firstAssembly=True
        self.par_L   = par_L
        self.petsc_L = par_L
        self.csr_rep_local = self.petsc_L.csr_rep_local
        self.csr_rep = self.petsc_L.csr_rep

        # create petsc4py KSP object and attach operators
        self.ksp = p4pyPETSc.KSP().create()
        self._setMatOperators()
        self.ksp.setOperators(self.petsc_L,self.petsc_L)

        self.setResTol(rtol_r,atol_r)

        if prefix is not None:
            self.ksp.setOptionsPrefix(prefix)
        if Preconditioner is not None:
            self._setPreconditioner(Preconditioner,par_L,prefix)
            self.ksp.setPC(self.pc)
        self.ksp.max_it = self.maxIts
        self.ksp.setFromOptions()
        # -<prefix>pc_lag_newton: set the preconditioner up on the first Newton iteration of each
        # nonlinear solve and reuse it for the rest of that solve (PETSc's own
        # -ksp_reuse_preconditioner would keep the first one for the whole run)
        self.pc_lag_newton = p4pyPETSc.Options(prefix or '').getBool('pc_lag_newton', False)
        # set null space class
        self.null_space = self._set_null_space_class()
        self.converged_on_maxit=False
        if convergenceTest in ['r-true', 'rits-true']:
            self.r_work = self.petsc_L.createVecLeft()
            self.rnorm0 = None
            self.ksp.setConvergenceTest(self._converged_trueRes)
            if convergenceTest == 'rits-true':
                self.converged_on_maxit=True
        else:
            self.r_work = None
            
    def setResTol(self,rtol,atol):
        """ Set the ksp object's residual and maximum interations. """
        self.rtol_r = rtol
        self.atol_r = atol
        self.ksp.rtol = rtol
        self.ksp.atol = atol
        logEvent("KSP atol %e rtol %e" % (self.ksp.atol,self.ksp.rtol))

    def prepare(self,
                b=None,
                newton_its=None):
        self.petsc_L.zeroEntries()
        assert self.petsc_L.getBlockSize() == 1, "petsc4py wrappers currently require 'simple' blockVec (blockSize=1) approach"
        if self.petsc_L.proteus_jacobian is not None:
            self.csr_rep[2][self.petsc_L.nzval_proteus2petsc] = self.petsc_L.proteus_csr_rep[2][:]
        if not self.par_fullOverlap and getattr(self.petsc_L, 'dirichlet_nonauthority', None) is not None:
            # strong Dirichlet dofs this rank is not the authority for: drop its copy of the row
            # (see LinearAlgebraTools.strongDirichletNonAuthorityDofs), the positions are built once
            if self._dirichlet_zero_positions is None:
                rowptr = self.csr_rep[0]
                self._dirichlet_zero_positions = numpy.concatenate(
                    [numpy.arange(rowptr[i], rowptr[i + 1]) for i in self.petsc_L.dirichlet_nonauthority]
                    + [numpy.zeros(0, dtype='i')]).astype('i')
            self.csr_rep[2][self._dirichlet_zero_positions] = 0.0
        if self.par_fullOverlap == True:
            self.petsc_L.setValuesLocalCSR(self.csr_rep_local[0],self.csr_rep_local[1],self.csr_rep_local[2],p4pyPETSc.InsertMode.INSERT_VALUES)
        else:
            if self.par_firstAssembly:
                self.petsc_L.setOption(p4pyPETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR,False)
                self.par_firstAssembly = False
            else:
                self.petsc_L.setOption(p4pyPETSc.Mat.Option.NEW_NONZERO_LOCATION_ERR,True)
            self.petsc_L.setValuesLocalCSR(self.csr_rep[0],self.csr_rep[1],self.csr_rep[2],p4pyPETSc.InsertMode.ADD_VALUES)
        self.petsc_L.assemblyBegin()
        self.petsc_L.assemblyEnd()
        self.ksp.setOperators(self.petsc_L,self.petsc_L)
        if self.pc_lag_newton:
            self.ksp.getPC().setReusePreconditioner(bool(newton_its))
        if self.pc is not None:
            self.pc.setOperators(self.petsc_L,self.petsc_L)
            self.pc.setUp()
            if self.preconditioner:
                self.preconditioner.setUp(self.ksp,newton_its)
        self.ksp.setUp()
        self.ksp.pc.setUp()

    def solve(self,u,r=None,b=None,par_u=None,par_b=None,initialGuessIsZero=True):
        if par_b.proteus2petsc_subdomain is not None:
            par_b.proteus_array[:] = par_b.proteus_array[par_b.petsc2proteus_subdomain]
            par_u.proteus_array[:] = par_u.proteus_array[par_u.petsc2proteus_subdomain]
        # if self.petsc_L.isSymmetric(tol=1.0e-12):
        #    self.petsc_L.setOption(p4pyPETSc.Mat.Option.SYMMETRIC, True)
        #    print "Matrix is symmetric"
        # else:
        #    print "MATRIX IS NONSYMMETRIC"
        logEvent("before ksp.rtol= %s ksp.atol= %s ksp.is_converged= %s ksp.its= %s ksp.norm= %s " % (self.ksp.rtol,
                                                                                                   self.ksp.atol,
                                                                                                   self.ksp.is_converged,
                                                                                                   self.ksp.its,
                                                                                                   self.ksp.norm))
        if self.pccontext is not None:
            self.pccontext.par_b = par_b
            self.pccontext.par_u = par_u
        if self.matcontext is not None:
            self.matcontext.par_b = par_b

        self.null_space.apply_ns(par_b)
        self.ksp.solve(par_b,par_u)
        logEvent("after ksp.rtol= %s ksp.atol= %s ksp.is_converged= %s ksp.its= %s ksp.norm= %s reason = %s" % (self.ksp.rtol,
                                                                                                             self.ksp.atol,
                                                                                                             self.ksp.is_converged,
                                                                                                             self.ksp.its,
                                                                                                             self.ksp.norm,
                                                                                                             self.ksp.reason))
        self.its = self.ksp.its
        #if self.printInfo:
        #    self.info()
        if par_b.proteus2petsc_subdomain is not None:
            par_b.proteus_array[:] = par_b.proteus_array[par_b.proteus2petsc_subdomain]
            par_u.proteus_array[:] = par_u.proteus_array[par_u.proteus2petsc_subdomain]

    def converged(self,r):
        """
        decide on convention to match norms, convergence criteria
        """
        return self.ksp.is_converged
    def failed(self):
        failedFlag = LinearSolver.failed(self)
        logEvent("KSPpetsc4py return flag {0}".format(failedFlag))
        logEvent("KSP converged flag {0}".format(self.ksp.is_converged))
        failedFlag = failedFlag or (not self.ksp.is_converged)
        return failedFlag

    def info(self):
        """Summarise the last solve.

        Every caller of info() either prints the result or concatenates it into
        an infoString, so it has to return a string. This used to call
        ksp.view(), which returns None and writes PETSc's full object dump to
        stdout -- so NI raised TypeError the moment it logged a KSP level
        solver, and any run that got that far was buried in dump output.
        """
        self.infoString = "************Start Linear Solver Info************\n"
        self.infoString += "ksp type              =  %s \n" % self.ksp.getType()
        self.infoString += "pc type               =  %s \n" % self.ksp.getPC().getType()
        self.infoString += "its                   =  %i \n" % self.ksp.getIterationNumber()
        self.infoString += "converged reason      =  %s \n" % self.ksp.getConvergedReason()
        self.infoString += "residual norm         = %12.5e\n" % self.ksp.getResidualNorm()
        self.infoString += "************End Linear Solver Info************\n"
        return self.infoString

    def _setMatOperators(self):
        """ Initializes python context for the ksp matrix operator """
        self.Lshell = p4pyPETSc.Mat().create()
        L_sizes = self.petsc_L.getSizes()
        L_range = self.petsc_L.getOwnershipRange()
        self.Lshell.setSizes(L_sizes)
        self.Lshell.setType('python')
        self.matcontext  = SparseMatShell(self.petsc_L.ghosted_csr_mat)
        self.Lshell.setPythonContext(self.matcontext)

    def _converged_trueRes(self,ksp,its,rnorm):
        """ Function handle to feed to ksp's setConvergenceTest  """
        ksp.buildResidual(self.r_work)
        truenorm = self.r_work.norm()
        if its == 0:
            self.rnorm0 = truenorm
            logEvent("NumericalAnalytics KSPOuterResidual: %12.5e" %(truenorm), level=7)
            if self.rnorm0 == 0.:
                logEvent("NumericalAnalytics KSPOuterResidual(relative): N/A (residual vector is zero)", level=7 )
                logEvent("        KSP it %i norm(r) = %e  norm(r)/|b| = N/A (residual vector is zero) ; atol=%e rtol=%e " % (its,
                                                                                                                             truenorm,
                                                                                                                             ksp.atol,
                                                                                                                             ksp.rtol))
            else:
                logEvent("NumericalAnalytics KSPOuterResidual(relative): %12.5e" %(truenorm/ self.rnorm0), level=7 )
                logEvent("        KSP it %i norm(r) = %e  norm(r)/|b| = %e ; atol=%e rtol=%e " % (its,
                                                                                                  truenorm,
                                                                                                  (truenorm/self.rnorm0),
                                                                                                  ksp.atol,
                                                                                                  ksp.rtol))
            return False
        else:
            logEvent("NumericalAnalytics KSPOuterResidual: %12.5e" %(truenorm), level=7)
            logEvent("NumericalAnalytics KSPOuterResidual(relative): %12.5e" %(truenorm/ self.rnorm0), level=7)
            logEvent("        KSP it %i norm(r) = %e  norm(r)/|b| = %e ; atol=%e rtol=%e " % (its,
                                                                                              truenorm,
                                                                                              (truenorm/self.rnorm0),
                                                                                              ksp.atol,
                                                                                              ksp.rtol))
            if truenorm < self.rnorm0*ksp.rtol:
                return p4pyPETSc.KSP.ConvergedReason.CONVERGED_RTOL
            if truenorm < ksp.atol:
                return p4pyPETSc.KSP.ConvergedReason.CONVERGED_ATOL
            if self.converged_on_maxit and its == ksp.max_it:
                return p4pyPETSc.KSP.ConvergedReason.CONVERGED_ITS
        return False

    def _setPreconditioner(self,
                           Preconditioner,
                           par_L,
                           prefix):
        """ Sets the preconditioner type used in the KSP object """
        if Preconditioner is not None:
            if Preconditioner == petsc_LU:
                logEvent("NAHeader Precondtioner LU")
                self.preconditioner = petsc_LU(par_L,
                                               prefix)
                self.pc = self.preconditioner.pc
            elif Preconditioner == petsc_ASM:
                logEvent("NAHead Preconditioner ASM")
                self.preconditioner = petsc_ASM(par_L,
                                                prefix)
                self.pc = self.preconditioner.pc
            elif Preconditioner == petsc_MG:
                logEvent("NAHead Preconditioner MG")
                self.preconditioner = petsc_MG(par_L,
                                               prefix,
                                               self.preconditionerOptions)
                self.pc = self.preconditioner.pc
            if Preconditioner == Jacobi:
                self.pccontext= Preconditioner(L,
                                               weight=1.0,
                                               rtol_r=rtol_r,
                                               atol_r=atol_r,
                                               maxIts=1,
                                               norm = l2Norm,
                                               convergenceTest='its',
                                               computeRates=False,
                                               printInfo=False)
                self.pc = p4pyPETSc.PC().createPython(self.pccontext)
            elif Preconditioner == GaussSeidel:
                self.pccontext= Preconditioner(connectionList,
                                               L,
                                               weight=1.0,
                                               sym=False,
                                               rtol_r=rtol_r,
                                               atol_r=atol_r,
                                               maxIts=1,
                                               norm = l2Norm,
                                               convergenceTest='its',
                                               computeRates=False,
                                               printInfo=False)
                self.pc = p4pyPETSc.PC().createPython(self.pccontext)
            elif Preconditioner == LU:
                #ARB - parallel matrices from PETSc4py don't work here.
                self.pccontext = Preconditioner(L)
                self.pc = p4pyPETSc.PC().createPython(self.pccontext)
            elif Preconditioner == StarILU:
                self.pccontext= Preconditioner(connectionList,
                                               L,
                                               weight=1.0,
                                               rtol_r=rtol_r,
                                               atol_r=atol_r,
                                               maxIts=1,
                                               norm = l2Norm,
                                               convergenceTest='its',
                                               computeRates=False,
                                               printInfo=False)
                self.pc = p4pyPETSc.PC().createPython(self.pccontext)
            elif Preconditioner == StarBILU:
                self.pccontext= Preconditioner(connectionList,
                                               L,
                                               bs=linearSolverLocalBlockSize,
                                               weight=1.0,
                                               rtol_r=rtol_r,
                                               atol_r=atol_r,
                                               maxIts=1,
                                               norm = l2Norm,
                                               convergenceTest='its',
                                               computeRates=False,
                                               printInfo=False)
                self.pc = p4pyPETSc.PC().createPython(self.pccontext)
            elif Preconditioner == SimpleNavierStokes3D:
                logEvent("NAHeader Preconditioner SimpleNavierStokes" )
                try:
                    self.preconditioner = SimpleNavierStokes3D(par_L,
                                                               prefix,
                                                               velocity_block_preconditioner=self.preconditionerOptions[0])
                except IndexError:
                    logEvent("Preconditioner options not specified, using defaults")
                    self.preconditioner = SimpleNavierStokes3D(par_L,
                                                               prefix)
                self.pc = self.preconditioner.pc
            elif Preconditioner == SimpleNavierStokes2D:
                logEvent("NAHeader Preconditioner SimpleNavierStokes" )
                try:
                    self.preconditioner = SimpleNavierStokes2D(par_L,
                                                               prefix,
                                                               velocity_block_preconditioner=self.preconditionerOptions[0])
                except IndexError:
                    logEvent("Preconditioner options not specified, using defaults")
                    self.preconditioner = SimpleNavierStokes2D(par_L,
                                                               prefix)
                self.pc = self.preconditioner.pc
            elif Preconditioner == Schur_Sp:
                logEvent("NAHeader Preconditioner selfp" )
                try:
                    self.preconditioner = Schur_Sp(par_L,
                                                   prefix,
                                                   velocity_block_preconditioner=self.preconditionerOptions[0])
                except IndexError:
                    logEvent("Preconditioner options not specified, using defaults")
                    self.preconditioner = Schur_Sp(par_L,
                                                   prefix)
                self.pc = self.preconditioner.pc
            elif Preconditioner == Schur_Qp:
                logEvent("NAHeader Preconditioner Qp" )
                self.preconditioner = Schur_Qp(par_L,
                                               prefix)
                self.pc = self.preconditioner.pc
            elif Preconditioner == NavierStokes_TwoPhasePCD:
                logEvent("NAHeader Preconditioner TwoPhasePCD")
                try:
                    self.preconditioner = NavierStokes_TwoPhasePCD(par_L,
                                                                   prefix,
                                                                   density_scaling=self.preconditionerOptions[0],
                                                                   numerical_viscosity=self.preconditionerOptions[1],
                                                                   lumped=self.preconditionerOptions[2],
                                                                   num_chebyshev_its=self.preconditionerOptions[3],
                                                                   laplace_null_space=self.preconditionerOptions[4],
                                                                   velocity_block_preconditioner=self.preconditionerOptions[5])
                except IndexError:
                    logEvent("Preconditioner options not specified, using defaults")
                    self.preconditioner = NavierStokes_TwoPhasePCD(par_L,
                                                                   prefix)
                self.pc = self.preconditioner.pc
            elif Preconditioner == Schur_LSC:
                logEvent("NAHeader Preconditioner LSC")
                self.preconditioner = Schur_LSC(par_L,
                                                prefix)
                self.pc = self.preconditioner.pc
            elif Preconditioner == SimpleDarcyFC:
                self.preconditioner = SimpleDarcyFC(par_L)
                self.pc = self.preconditioner.pc
            elif Preconditioner == NavierStokesPressureCorrection:
                self.preconditioner = NavierStokesPressureCorrection(par_L,
                                                                     prefix)
                self.pc = self.preconditioner.pc

    def _set_null_space_class(self):
        current_module = sys.modules[__name__]
        null_space_cls_name = self.par_L.pde.coefficients.nullSpace
        null_space_cls = getattr(current_module,
                                 null_space_cls_name)
        return null_space_cls(self)

class SchurOperatorConstructor(object):
    """
    Generate matrices for use in Schur complement preconditioner operators.
    """
    def __init__(self, linear_smoother, pde_type='general_saddle_point'):
        """
        Initialize a Schur Operator constructor.

        Parameters
        ----------
        linear_smoother : class
            Provides the data about the problem.
        pde_type :  str
            Currently supports Stokes and navierStokes
        """
        from proteus.mprans import RANS2P
        if linear_smoother.PCType!='schur':
            raise Exception('This function only works with the' \
                'LinearSmoothers for Schur Complements.')
        self.linear_smoother=linear_smoother
        self.L = linear_smoother.L
        self.pde_type = pde_type
        # ARB TODO : the Schur class should be refactored to avoid
        # the follow expection statement
        try:
            self.L.pde
            pass
        except AttributeError:
            return

        if isinstance(self.L.pde, RANS2P.LevelModel):
            self.opBuilder = OperatorConstructor_rans2p(self.L.pde)
        else:
            self.opBuilder = OperatorConstructor_oneLevel(self.L.pde)
            try:
                self._phase_func = self.L.pde.coefficients.which_region
            except AttributeError:
                pass

    def _initializeMat(self,jacobian):
        from . import Comm
        comm = Comm.get()
        transport = self.L.pde
        rowptr, colind, nzval = jacobian.getCSRrepresentation()
        rowptr_petsc = rowptr.copy()
        colind_petsc = colind.copy()
        nzval[:] = numpy.arange(nzval.shape[0])
        nzval_petsc = nzval.copy()
        nzval_proteus2petsc=colind.copy()
        nzval_petsc2proteus=colind.copy()
        rowptr_petsc[0] = 0
        comm.beginSequential()
        for i in range(LAT.ParInfo_petsc4py.par_n+LAT.ParInfo_petsc4py.par_nghost):
            start_proteus = rowptr[LAT.ParInfo_petsc4py.petsc2proteus_subdomain[i]]
            end_proteus = rowptr[LAT.ParInfo_petsc4py.petsc2proteus_subdomain[i]+1]
            nzrow =  end_proteus - start_proteus
            rowptr_petsc[i+1] = rowptr_petsc[i] + nzrow
            start_petsc = rowptr_petsc[i]
            end_petsc = rowptr_petsc[i+1]
            petsc_cols_i = LAT.ParInfo_petsc4py.proteus2petsc_subdomain[colind[start_proteus:end_proteus]]
            j_sorted = petsc_cols_i.argsort()
            colind_petsc[start_petsc:end_petsc] = petsc_cols_i[j_sorted]
            nzval_petsc[start_petsc:end_petsc] = nzval[start_proteus:end_proteus][j_sorted]
            for j_petsc, j_proteus in zip(numpy.arange(start_petsc,end_petsc),
                                          numpy.arange(start_proteus,end_proteus)[j_sorted]):
                nzval_petsc2proteus[j_petsc] = j_proteus
                nzval_proteus2petsc[j_proteus] = j_petsc
        comm.endSequential()
        assert(nzval_petsc.shape[0] == colind_petsc.shape[0] == rowptr_petsc[-1] - rowptr_petsc[0])
        petsc_a = {}
        proteus_a = {}
        for i in range(transport.dim):
            for j,k in zip(colind[rowptr[i]:rowptr[i+1]],list(range(rowptr[i],rowptr[i+1]))):
                nzval[k] = i*transport.dim+j
                proteus_a[i,j] = nzval[k]
                petsc_a[LAT.ParInfo_petsc4py.proteus2petsc_subdomain[i],LAT.ParInfo_petsc4py.proteus2petsc_subdomain[j]] = nzval[k]
        for i in range(transport.dim):
            for j,k in zip(colind_petsc[rowptr_petsc[i]:rowptr_petsc[i+1]],list(range(rowptr_petsc[i],rowptr_petsc[i+1]))):
                nzval_petsc[k] = petsc_a[i,j]
        return SparseMat(transport.dim,transport.dim,nzval_petsc.shape[0], nzval_petsc, colind_petsc, rowptr_petsc)

    def initializeTwoPhaseCp_rho(self):
        """Initialize a two phase scaled advection operator Cp.

        Returns
        -------
        two_phase_Cp_rho : matrix
        """
        from . import Comm
        comm = Comm.get()
        self.opBuilder.attachTPAdvectionOperator()
        par_info = self.linear_smoother.L.pde.par_info
        if comm.size() == 1:
            self.two_phase_Cp_rho = ParMat_petsc4py(self.opBuilder.TPScaledAdvectionOperator,
                                                    par_info.par_bs,
                                                    par_info.par_n,
                                                    par_info.par_N,
                                                    par_info.par_nghost,
                                                    par_info.subdomain2global)
        else:
            mixed = False
            if mixed == True:
                self.petsc_two_phase_Cp_rho = self._initializeMat(self.opBuilder.TPScaledAdvectionOperator)
                self.two_phase_Cp_rho = ParMat_petsc4py(self.petsc_two_phase_Cp_rho,
                                                        par_info.par_bs,
                                                        par_info.par_n,
                                                        par_info.par_N,
                                                        par_info.par_nghost,
                                                        par_info.petsc_subdomain2global_petsc,
                                                        pde=self.L.pde,
                                                        proteus_jacobian=self.opBuilder.TPScaledAdvectionOperator,
                                                        nzval_proteus2petsc=par_info.nzval_proteus2petsc)
            else:
                self.two_phase_Cp_rho = ParMat_petsc4py(self.opBuilder.TPScaledAdvectionOperator,
                                                        par_info.par_bs,
                                                        par_info.par_n,
                                                        par_info.par_N,
                                                        par_info.par_nghost,
                                                        par_info.subdomain2global,
                                                        pde=self.L.pde)
        self.two_phase_Cp_rho_csr_rep = self.two_phase_Cp_rho.csr_rep
        self.two_phase_Cp_rho_csr_rep_local = self.two_phase_Cp_rho.csr_rep_local
        return self.two_phase_Cp_rho

    def initializeTwoPhaseInvScaledAp(self):
        """Initialize a two phase scaled laplace operator Ap.

        Returns
        -------
        two_phase_Ap_inv : matrix
        """
        from . import Comm
        comm = Comm.get()
        self.opBuilder.attachLaplaceOperator()
        par_info = self.linear_smoother.L.pde.par_info
        if comm.size() == 1:
            self.two_phase_Ap_inv = ParMat_petsc4py(self.opBuilder.TPInvScaledLaplaceOperator,
                                                    par_info.par_bs,
                                                    par_info.par_n,
                                                    par_info.par_N,
                                                    par_info.par_nghost,
                                                    par_info.subdomain2global)
        else:
            mixed = False
            if mixed == True:
                self.petsc_two_phase_Ap_inv = self._initializeMat(self.opBuilder.TPInvScaledLaplaceOperator)
                self.two_phase_Ap_inv = ParMat_petsc4py(self.petsc_two_phase_Ap_inv,
                                                        par_info.par_bs,
                                                        par_info.par_n,
                                                        par_info.par_N,
                                                        par_info.par_nghost,
                                                        par_info.petsc_subdomain2global_petsc,
                                                        pde=self.L.pde,
                                                        proteus_jacobian=self.opBuilder.TPInvScaledLaplaceOperator,
                                                        nzval_proteus2petsc=par_info.nzval_proteus2petsc)
            else:
                self.two_phase_Ap_inv = ParMat_petsc4py(self.opBuilder.TPInvScaledLaplaceOperator,
                                                        par_info.par_bs,
                                                        par_info.par_n,
                                                        par_info.par_N,
                                                        par_info.par_nghost,
                                                        par_info.subdomain2global,
                                                        pde=self.L.pde)                
        self.two_phase_Ap_inv_csr_rep = self.two_phase_Ap_inv.csr_rep
        self.two_phase_Ap_inv_csr_rep_local = self.two_phase_Ap_inv.csr_rep_local
        return self.two_phase_Ap_inv

    def initializeTwoPhaseQp_rho(self):
        """Initialize a two phase scaled mass matrix.

        Returns
        -------
        two_phase_Ap_inv : matrix
        """
        from . import Comm
        comm = Comm.get()
        self.opBuilder.attachScaledMassOperator()
        par_info = self.linear_smoother.L.pde.par_info
        if comm.size() == 1:
            self.two_phase_Qp_scaled = ParMat_petsc4py(self.opBuilder.TPScaledMassOperator,
                                                       par_info.par_bs,
                                                       par_info.par_n,
                                                       par_info.par_N,
                                                       par_info.par_nghost,
                                                       par_info.subdomain2global)
        else:
            mixed = False
            if mixed == True:
                self.petsc_two_phase_Qp_scaled = self._initializeMat(self.opBuilder.TPScaledMassOperator)
                self.two_phase_Qp_scaled = ParMat_petsc4py(self.petsc_two_phase_Qp_scaled,
                                                           par_info.par_bs,
                                                           par_info.par_n,
                                                           par_info.par_N,
                                                           par_info.par_nghost,
                                                           par_info.petsc_subdomain2global_petsc,
                                                           pde=self.L.pde,
                                                           proteus_jacobian=self.opBuilder.TPScaledMassOperator,
                                                           nzval_proteus2petsc=par_info.nzval_proteus2petsc)
            else:
                self.two_phase_Qp_scaled = ParMat_petsc4py(self.opBuilder.TPScaledMassOperator,
                                                           par_info.par_bs,
                                                           par_info.par_n,
                                                           par_info.par_N,
                                                           par_info.par_nghost,
                                                           par_info.subdomain2global)
        self.two_phase_Qp_scaled_csr_rep = self.two_phase_Qp_scaled.csr_rep
        self.two_phase_Qp_scaled_csr_rep_local = self.two_phase_Qp_scaled.csr_rep_local
        return self.two_phase_Qp_scaled

    def initializeTwoPhaseInvScaledQp(self):
        """Initialize a two phase scaled mass operator Qp.

        Returns
        -------
        two_phase_Ap_inv : matrix
        """
        from . import Comm
        comm = Comm.get()
        self.opBuilder.attachInvScaledMassOperator()
        par_info = self.linear_smoother.L.pde.par_info
        if comm.size() == 1:
            self.two_phase_Qp_inv = ParMat_petsc4py(self.opBuilder.TPInvScaledMassOperator,
                                                    par_info.par_bs,
                                                    par_info.par_n,
                                                    par_info.par_N,
                                                    par_info.par_nghost,
                                                    par_info.subdomain2global)
        else:
            mixed = False
            if mixed == True:
                self.petsc_two_phase_Qp_inv = self._initializeMat(self.opBuilder.TPInvScaledMassOperator)
                self.two_phase_Qp_inv = ParMat_petsc4py(self.petsc_two_phase_Qp_inv,
                                                        par_info.par_bs,
                                                        par_info.par_n,
                                                        par_info.par_N,
                                                        par_info.par_nghost,
                                                        par_info.petsc_subdomain2global_petsc,
                                                        pde=self.L.pde,
                                                        proteus_jacobian=self.opBuilder.TPInvScaledMassOperator,
                                                        nzval_proteus2petsc=par_info.nzval_proteus2petsc)
            else:
                self.two_phase_Qp_inv = ParMat_petsc4py(self.opBuilder.TPInvScaledMassOperator,
                                                        par_info.par_bs,
                                                        par_info.par_n,
                                                        par_info.par_N,
                                                        par_info.par_nghost,
                                                        par_info.subdomain2global)                
        self.two_phase_Qp_inv_csr_rep = self.two_phase_Qp_inv.csr_rep
        self.two_phase_Qp_inv_csr_rep_local = self.two_phase_Qp_inv.csr_rep_local
        return self.two_phase_Qp_inv

    def initializeQ(self):
        """ Initialize a mass matrix Q.

        Returns
        -------
        Q : matrix
            The mass matrix.
        """
        from . import Comm
        comm = Comm.get()
        self.opBuilder.attachMassOperator()
        par_info = self.linear_smoother.L.pde.par_info
        if comm.size() == 1:
            if par_info.mixed is False:
                self.Q = ParMat_petsc4py(self.opBuilder.MassOperator,
                                         par_info.par_bs,
                                         par_info.par_n,
                                         par_info.par_N,
                                         par_info.par_nghost,
                                         par_info.subdomain2global)
            else:
                self.Q = ParMat_petsc4py(self.opBuilder.MassOperator,
                                         1,
                                         par_info.par_n,
                                         par_info.par_N,
                                         par_info.par_nghost,
                                         par_info.subdomain2global,
                                         pde=self.L.pde)
        else:
            if par_info.mixed is True:
                self.petsc_Q = self._initializeMat(self.opBuilder.MassOperator)
                self.Q = ParMat_petsc4py(self.opBuilder.MassOperator,
                                         1,
                                         par_info.par_n,
                                         par_info.par_N,
                                         par_info.par_nghost,
                                         par_info.petsc_subdomain2global_petsc,
                                         pde=self.L.pde,
                                         proteus_jacobian=self.opBuilder.MassOperator,
                                         nzval_proteus2petsc=par_info.nzval_proteus2petsc)
            else:
                self.Q = ParMat_petsc4py(self.petsc_Q,
                                         par_info.par_bs,
                                         par_info.par_n,
                                         par_info.par_N,
                                         par_info.par_nghost,
                                         par_info.petsc_subdomain2global_petsc,
                                         pde=self.L.pde,
                                         proteus_jacobian=self.opBuilder.MassOperator,
                                         nzval_proteus2petsc=par_info.nzval_proteus2petsc)
        self.Q_csr_rep = self.Q.csr_rep
        self.Q_csr_rep_local = self.Q.csr_rep_local
        return self.Q

    def updateQ(self,
                output_matrix = False):
        """
        Update the mass matrix operator.

        Parameters
        ----------
        output_matrix : bool
            Save updated mass operator.
        """
        self.opBuilder.updateMassOperator()
        self.Q.zeroEntries()
        if self.Q.proteus_jacobian != None:
            self.Q_csr_rep[2][self.Q.nzval_proteus2petsc] = self.Q.proteus_csr_rep[2][:]
        self.Q.setValuesLocalCSR(self.Q_csr_rep_local[0],
                                 self.Q_csr_rep_local[1],
                                 self.Q_csr_rep_local[2],
                                 p4pyPETSc.InsertMode.INSERT_VALUES)
        self.Q.assemblyBegin()
        self.Q.assemblyEnd()
        if output_matrix is True:
            self._exportMatrix(self.Q,'Q')

    def updateNp_rho(self,
                     density_scaling = True,
                     output_matrix = False):
        """
        Update the two-phase advection operator.

        Parameters
        ----------
        density_scaling : bool
            Indicates whether advection terms should be scaled with
            the density (True) or 1 (False)
        output_matrix : bool
            Save updated advection operator.
        """
        self.opBuilder.updateTPAdvectionOperator(density_scaling)
        self.two_phase_Cp_rho.zeroEntries()
        if self.two_phase_Cp_rho.proteus_jacobian != None:
            self.two_phase_Cp_rho_csr_rep[2][self.two_phase_Cp_rho.nzval_proteus2petsc] = self.two_phase_Cp_rho.proteus_csr_rep[2][:]
        self.two_phase_Cp_rho.setValuesLocalCSR(self.two_phase_Cp_rho_csr_rep_local[0],
                                                self.two_phase_Cp_rho_csr_rep_local[1],
                                                self.two_phase_Cp_rho_csr_rep_local[2],
                                                p4pyPETSc.InsertMode.INSERT_VALUES)
        self.two_phase_Cp_rho.assemblyBegin()
        self.two_phase_Cp_rho.assemblyEnd()
        if output_matrix is True:
            self._exportMatrix(self.two_phase_Cp_rho,'Cp_rho')

    def updateInvScaledAp(self,
                          output_matrix = False):
        """Update the two-phase laplace operator.

        Parameters
        ----------
        output_matrix : bool
            Save updated laplace operator.
        """
        self.opBuilder.updateTPInvScaledLaplaceOperator()
        self.two_phase_Ap_inv.zeroEntries()
        if self.two_phase_Ap_inv.proteus_jacobian != None:
            self.two_phase_Ap_inv_csr_rep[2][self.two_phase_Ap_inv.nzval_proteus2petsc] = self.two_phase_Ap_inv.proteus_csr_rep[2][:]
        self.two_phase_Ap_inv.setValuesLocalCSR(self.two_phase_Ap_inv_csr_rep_local[0],
                                                self.two_phase_Ap_inv_csr_rep_local[1],
                                                self.two_phase_Ap_inv_csr_rep_local[2],
                                                p4pyPETSc.InsertMode.INSERT_VALUES)
        self.two_phase_Ap_inv.assemblyBegin()
        self.two_phase_Ap_inv.assemblyEnd()
        if output_matrix is True:
            self._exportMatrix(self.two_phase_Ap_inv,'Cp_rho')

    def updateTwoPhaseQp_rho(self,
                             density_scaling = True,
                             lumped = True,
                             output_matrix=False):
        """Update the two-phase inverse viscosity scaled mass matrix.

        Parameters
        ----------
        density : bool
            Indicates whether the density mass matrix should
            be scaled with rho (True) or 1 (False).
        lumped : bool
            Flag indicating whether the mass operator should be
            calculated as a lumped matrix (True) or as a full
            matrix (False).
        output_matrix : bool
            Save updated mass operator.
        """
        self.opBuilder.updateTwoPhaseMassOperator_rho(density_scaling = density_scaling,
                                                      lumped = lumped)
        self.two_phase_Qp_scaled.zeroEntries()
        if self.two_phase_Qp_scaled.proteus_jacobian != None:
            self.two_phase_Qp_scaled_csr_rep[2][self.two_phase_Qp_scaled.nzval_proteus2petsc] = self.two_phase_Qp_scaled.proteus_csr_rep[2][:]
        self.two_phase_Qp_scaled.setValuesLocalCSR(self.two_phase_Qp_scaled_csr_rep_local[0],
                                                   self.two_phase_Qp_scaled_csr_rep_local[1],
                                                   self.two_phase_Qp_scaled_csr_rep_local[2],
                                                   p4pyPETSc.InsertMode.INSERT_VALUES)
        self.two_phase_Qp_scaled.assemblyBegin()
        self.two_phase_Qp_scaled.assemblyEnd()
        if output_matrix is True:
            self._exportMatrix(self.two_phase_Qp_scaled,'Qp_scaled')

    def updateTwoPhaseInvScaledQp_visc(self,
                                       numerical_viscosity = True,
                                       lumped = True,
                                       output_matrix=False):
        """
        Update the two-phase inverse viscosity scaled mass matrix.

        Parameters
        ----------
        numerical_viscosity : bool
            Indicates whether the numerical viscosity should be
            included with the mass operator (True to include,
            False to exclude)
        lumped : bool
            Flag indicating whether the mass operator should be
            calculated as a lumped matrix (True) or as a full
            matrix (False).
        output_matrix : bool
            Save updated mass operator.
        """
        self.opBuilder.updateTwoPhaseInvScaledMassOperator(numerical_viscosity = numerical_viscosity,
                                                           lumped = lumped)
        self.two_phase_Qp_inv.zeroEntries()
        if self.two_phase_Qp_inv.proteus_jacobian != None:
            self.two_phase_Qp_inv_csr_rep[2][self.two_phase_Qp_inv.nzval_proteus2petsc] = self.two_phase_Qp_inv.proteus_csr_rep[2][:]
        self.two_phase_Qp_inv.setValuesLocalCSR(self.two_phase_Qp_inv_csr_rep_local[0],
                                                self.two_phase_Qp_inv_csr_rep_local[1],
                                                self.two_phase_Qp_inv_csr_rep_local[2],
                                                p4pyPETSc.InsertMode.INSERT_VALUES)
        self.two_phase_Qp_inv.assemblyBegin()
        self.two_phase_Qp_inv.assemblyEnd()
        if output_matrix is True:
            self._exportMatrix(self.two_phase_Qp_scaled,'Qp_scaled')

    def getQv(self, output_matrix=False, recalculate=False):
        """ Return the pressure mass matrix Qp.

        Parameters
        ----------
        output_matrix : bool
            Determines whether matrix should be exported.
        recalculate : bool
            Flag indicating whether matrix should be rebuilt every iteration

        Returns
        -------
        Qp : matrix
            The pressure mass matrix.
        """
        Qsys_petsc4py = self._massMatrix(recalculate = recalculate)
        self.Qv = Qsys_petsc4py.createSubMatrix(self.linear_smoother.isv,
                                             self.linear_smoother.isv)
        if output_matrix is True:
            self._exportMatrix(self.Qv,"Qv")
        return self.Qv

    def getQp(self, output_matrix=False, recalculate=False):
        """ Return the pressure mass matrix Qp.

        Parameters
        ----------
        output_matrix : bool
            Determines whether matrix should be exported.
        recalculate : bool
            Flag indicating whether matrix should be rebuilt every iteration

        Returns
        -------
        Qp : matrix
            The pressure mass matrix.
        """
        Qsys_petsc4py = self._massMatrix(recalculate = recalculate)
        self.Qv = Qsys_petsc4py.createSubMatrix(self.linear_smoother.isv,
                                             self.linear_smoother.isv)
        if output_matrix is True:
            self._exportMatrix(self.Qv,"Qv")
        return self.Qv

    def _massMatrix(self,recalculate=False):
        """ Generates a mass matrix.

        This function generates and returns the mass matrix for the system. This
        function is internal to the class and called by public functions which
        take and return the relevant components (eg. the pressure or velcoity).

        Parameters
        ----------
        recalculate : bool
            Indicates whether matrix should be rebuilt everytime it's called.

        Returns
        -------
        Qsys : matrix
            The system's mass matrix.
        """
        self.opBuilder.attachMassOperator()

class KSP_Preconditioner(object):
    """ Base class for PETSc KSP precondtioners. """
    def __init__(self):
        pass

    def setup(self, global_ksp=None):
        pass

class petsc_ASM(KSP_Preconditioner):
    """ASM PETSc preconditioner class.

    This class provides an ASM preconditioners for PETSc4py KSP
    objects.
    """
    def __init__(self, 
                 L,
                 prefix=None):
        """
        Initializes the ASMpreconditioner for use with PETSc.

        Parameters
        ----------
        L : the global system matrix.
        prefix : str
            Prefix handle for PETSc options.
        """
        self.PCType = 'asm'
        self.L = L
        self._initializePC(prefix)
        self.pc.setFromOptions()

    def _initializePC(self,
                      prefix=None):
        """ Create the pc object. """
        self.pc = p4pyPETSc.PC().create()
        self.pc.setOptionsPrefix(prefix)
        self.pc.setType('asm')

    def setUp(self,
              global_ksp=None,
              newton_its=None):
        self.pc.setUp()

COARSE_SOLVERS = ('redundant','direct','amg')


def coarseSolverOptions(kind):
    """PETSc options (names without a prefix) that make a coarse level be solved as ``kind``.

    The coarsest level of a multigrid cycle has to be solved, not smoothed, and in parallel
    that is a choice with a scaling consequence.

    ``direct`` (the default) is a parallel sparse direct solver (MUMPS, else SuperLU_DIST):
    exact, so a cycle is the same cycle to roundoff whatever the rank count, with the factors
    distributed. Serious parallel builds have one of the two, and at a small coarse level it
    costs the same as ``redundant`` (measured: 0.16 vs 0.19 s at 64 ranks, 4 levels).

    ``redundant`` gathers the coarse matrix onto every rank, factors it there in full, and
    applies it with no iteration. It is exact and needs no external package, but it is
    *replicated*: every rank does the same work, so with weak scaling -- a coarse level that
    grows with the rank count -- the cost grows with it. Measured on a 3-level hexahedral
    hierarchy at 64 ranks it was 91% of all flops and took 8.7 s of a 12.6 s run (``direct``:
    0.15 s); a deeper hierarchy made it cheap. Keep it for builds with no parallel direct
    solver.

    ``amg`` is one algebraic multigrid V-cycle (hypre BoomerAMG, else PETSc's GAMG): scalable
    and cheap, but *not* exact -- the coarse correction is approximate, so a proteus MGM cycle
    stops being the serial cycle and agrees with it only through convergence. It suits
    elliptic coarse operators; it is a poor bet for strongly nonsymmetric ones.

    Any ``-<prefix>`` option the user sets takes precedence over these.
    """
    if kind == 'redundant':
        return (('ksp_type','preonly'),('pc_type','redundant'),('redundant_pc_type','lu'))
    if kind == 'direct':
        for package in ('mumps','superlu_dist'):
            if p4pyPETSc.Sys.hasExternalPackage(package):
                return (('ksp_type','preonly'),('pc_type','lu'),
                        ('pc_factor_mat_solver_type',package))
        raise ValueError("coarseSolver='direct' needs PETSc built with MUMPS or SuperLU_DIST")
    if kind == 'amg':
        if p4pyPETSc.Sys.hasExternalPackage('hypre'):
            return (('ksp_type','preonly'),('pc_type','hypre'),('pc_hypre_type','boomeramg'))
        return (('ksp_type','preonly'),('pc_type','gamg'))
    raise ValueError("coarseSolver must be one of %s, not %r" % (COARSE_SOLVERS,kind))


def applyCoarseSolverDefaults(options,coarsePrefix,kind):
    """Fill in the options for ``kind`` under ``coarsePrefix`` that the user has not set.

    "Not set" is an exact-key test: PETSc's lookup falls back from a numbered prefix to the
    unnumbered one, so hasName('mg_levels_0_ksp_type') is true whenever mg_levels_ksp_type
    is, and a hasName guard would never fill in the option that has to override it.
    """
    given = options.getAll()
    for name,value in coarseSolverOptions(kind):
        if coarsePrefix+name not in given:
            options[coarsePrefix+name] = value


class MGHierarchy(object):
    """The levels a petsc_MG preconditioner is assembled over.

    ``operators[l]`` is level l's system matrix and ``interpolations[l]`` the
    PETSc Mat taking level l-1 to level l, for every level *below* the one the
    preconditioner belongs to -- that level's own matrix is the KSP's. Level 0
    is the coarsest, as everywhere in proteus. ``interpolations[0]`` is unused
    and kept None so the lists index like prolongList does.
    """
    def __init__(self,operators,interpolations,coarseSolver='direct'):
        assert len(interpolations) == len(operators) + 1, (len(operators),len(interpolations))
        self.operators = operators
        self.interpolations = interpolations
        self.coarseSolver = coarseSolver

    @property
    def nLevels(self):
        return len(self.operators) + 1


class petsc_MG(KSP_Preconditioner):
    """PETSc's geometric multigrid, PCMG, over proteus's own level hierarchy.

    It uses the same two ingredients proteus's MGM does -- the transfer
    operators from the multilevel mesh and one system matrix per level -- and
    leaves everything else to PETSc: smoothers, cycle type and coarse solver
    are PETSc options (``-mg_levels_ksp_type``, ``-pc_mg_cycle_type``, ...
    under the solver's options prefix). That is what makes it a cross-check on
    MGM: the same discretization and transfers with an independent
    implementation of the cycle.

    Only the coarse solve is configured here, because PETSc's default there
    needs a parallel direct solver that a plain build lacks. As in
    mgmCoarseSolver it gathers the coarse matrix and factors it in full
    (``redundant``), and a user setting any ``mg_coarse_`` option takes
    precedence.

    The operators are the model's own re-discretized level matrices, not
    Galerkin products, matching MGM. Their values are refreshed by whichever
    solver owns each level, so this reads them as they stand at setUp.
    """
    def __init__(self,L,prefix=None,hierarchy=None):
        """
        Parameters
        ----------
        L : the global system matrix of the level this preconditioner is for.
        prefix : str
            Prefix handle for PETSc options.
        hierarchy : MGHierarchy
            The coarser levels and the transfers between them.
        """
        assert isinstance(hierarchy,MGHierarchy), (
            "petsc_MG needs the level hierarchy; pass an MGHierarchy as "
            "preconditionerOptions (multilevelLinearSolverChooser does this)")
        self.PCType = 'mg'
        self.L = L
        self.hierarchy = hierarchy
        self.pc = p4pyPETSc.PC().create()
        self.pc.setOptionsPrefix(prefix)
        self.pc.setType('mg')
        self.pc.setMGLevels(hierarchy.nLevels)
        for l in range(1,hierarchy.nLevels):
            self.pc.setMGInterpolation(l,hierarchy.interpolations[l])
        for l,A in enumerate(hierarchy.operators):
            smoother = self.pc.getMGCoarseSolve() if l == 0 else self.pc.getMGSmoother(l)
            smoother.setOperators(A,A)
        #With one level PCMG has no coarse solver: level 0 is configured as a
        #smoother, under mg_levels_0_, and a user's mg_levels_ options (richardson
        #and jacobi, say) would make the "solve" three damped sweeps. A level
        #with nothing below it has to be exact -- it is level 0 of a nested
        #iteration -- so its defaults go under the prefix PETSc actually reads.
        #
        #"Not set by the user" has to be an exact-key test. PETSc's lookup
        #falls back from a numbered prefix to the unnumbered one, so
        #hasName('mg_levels_0_ksp_type') is true whenever mg_levels_ksp_type is,
        #and a hasName guard would never fill in the very option that has to
        #override it.
        coarse = 'mg_levels_0_' if hierarchy.nLevels == 1 else 'mg_coarse_'
        applyCoarseSolverDefaults(p4pyPETSc.Options(prefix),coarse,hierarchy.coarseSolver)
        self.pc.setFromOptions()

    def setUp(self,
              global_ksp=None,
              newton_its=None):
        self.pc.setUp()

class petsc_LU(KSP_Preconditioner):
    """ LU PETSc preconditioner class.

    This class provides an LU preconditioner for PETSc4py KSP
    objects.  Provided the LU decomposition is successful, the KSP
    iterative will converge in a single step.
    """
    def __init__(self,L,prefix=None):
        """
        Initializes the LU preconditioner for use with PETSc.

        Parameters
        ----------
        L : the global system matrix.
        prefix : str
            Prefix handle for PETSc options.
        """
        self.PCType = 'lu'
        self.L = L
        self._initializePC(prefix)
        self.pc.setFromOptions()

    def _initializePC(self,
                      prefix):
        r"""
        Intiailizes the PETSc precondition.

        Parameters
        ----------
        prefix : str
            Prefix identifier for command line PETSc options.
        """
        self.pc = p4pyPETSc.PC().create()
        self.pc.setOptionsPrefix(prefix)
        self.pc.setType('lu')

    def setUp(self,
              global_ksp=None,
              newton_its=None):
        pass

class DofOrderInfo(object):
    """Base class for managing DOF ordering information

    Parameters
    ----------
    dof_order_type : str
        This describes the type of dof ordering that will
        be constructed.  Currently supports: 'blocked'
        and 'interlaced'.
    """
    def __init__(self,
                 dof_order_type,
                 model_info = 'no model info set'):
        self.dof_order_type = dof_order_type
        self.set_model_info(model_info)

    def create_DOF_lists(self,
                         ownership_range,
                         num_equations,
                         num_components):
        """Virtual function with no implementation"""
        raise NotImplementedError()

    def set_model_info(self, model_info):
        self._model_info = model_info

    def get_model_info(self, model_info):
        return self._model_info

    def create_IS(self,
                  dof_array):
        idx_set = p4pyPETSc.IS()
        idx_set.createGeneral(dof_array,comm=p4pyPETSc.COMM_WORLD)
        return idx_set

class BlockedDofOrderType(DofOrderInfo):
    """Manages the DOF for blocked velocity and pressure ordering.

    Parameters
    ----------
    n_DOF_pressure : int
        Number of pressure degrees of freedom.

    Notes
    -----
    Blocked degree of freedom ordering occurs when all the pressure
    unknowns appear first, followed by all the u-components of the
    velocity and then all the v-components of the velocity etc.
    """
    def __init__(self,
                 n_DOF_pressure,
                 model_info = 'no model info set'):
        DofOrderInfo.__init__(self,
                              'blocked',
                              model_info = 'no model info set')
        self.n_DOF_pressure = n_DOF_pressure

    def create_DOF_lists(self,
                         ownership_range,
                         num_equations,
                         num_components):
        """Build blocked velocity and pressure DOF arrays.

        Parameters
        ----------
        ownership_range: tuple
            Local ownership range of DOF
        num_equations: int
            Number of local equations
        num_components: int
            Number of pressure and velocity components

        Returns
        -------
        DOF_output : lst of arrays
            This function returns a list of arrays with the DOF
            order.  [velocityDOF, pressureDOF]
        """
        pressureDOF = numpy.arange(start=ownership_range[0],
                                   stop=ownership_range[0]+self.n_DOF_pressure,
                                   dtype="i")
        velocityDOF = numpy.arange(start=ownership_range[0]+self.n_DOF_pressure,
                                   stop=ownership_range[0]+num_equations,
                                   step=1,
                                   dtype="i")
        return [velocityDOF, pressureDOF]

class InterlacedDofOrderType(DofOrderInfo):
    """Manages the DOF for interlaced velocity and pressure ordering.

    Notes
    -----
    Interlaced degrees of occur when the degrees of freedom are
    ordered as (p[0], u[0], v[0], p[1], u[1], ..., p[n], u[n], v[n]).
    """
    def __init__(self,
                 model_info = 'no model info set'):
        DofOrderInfo.__init__(self,
                              'interlaced',
                              model_info = model_info)

    def create_DOF_lists(self,
                         ownership_range,
                         num_equations,
                         num_components):
        """Build interlaced velocity and pressure DOF arrays.

        Parameters
        ----------
        ownership_range: tuple
            Local ownership range of DOF
        num_equations: int
            Number of local equations
        num_components: int
            Number of pressure and velocity components

        Returns
        -------
        DOF_output : lst of arrays
            This function returns a list of arrays with the DOF
            order.  [velocityDOF, pressureDOF]
        """
        pressureDOF = numpy.arange(start=ownership_range[0],
                                   stop=ownership_range[0]+num_equations,
                                   step=num_components,
                                   dtype="i")
        velocityDOF = []
        for start in range(1,num_components):
            velocityDOF.append(numpy.arange(start=ownership_range[0]+start,
                                            stop=ownership_range[0]+num_equations,
                                            step=num_components,
                                            dtype="i"))
        velocityDOF = numpy.vstack(velocityDOF).transpose().flatten()
        return [velocityDOF, pressureDOF]

    def create_vel_DOF_IS(self,
                          ownership_range,
                          num_equations,
                          num_components):
        """
        Build interlaced DOF arrays for the components of the velocity.

        Parameters
        ----------
        ownership_range: tuple
            Local ownership range of DOF
        num_equations: int
            Number of local equations
        num_components: int
            Number of pressure and velocity components

        Returns
        -------
        DOF_output : lst of arrays
            Each element of this list corresponds to a component of
            the velocity.  E.g. for u, v, w : [vel_u,vel_v,vel_w].
        """
        from . import Comm
        comm = Comm.get()
        vel_comp_DOF = []
        vel_comp_DOF_vel=[]
        scaled_ownership_range = ownership_range[0] * (num_components-1) / num_components
        for i in range(1,num_components):
            vel_comp_DOF.append(self.create_IS(numpy.arange(start=ownership_range[0] + i,
                                                            stop=ownership_range[0] + num_equations,
                                                            step=num_components,
                                                            dtype="i")))
            vel_comp_DOF_vel.append(self.create_IS(numpy.arange(start=scaled_ownership_range + i - 1,
                                                                stop=scaled_ownership_range + int( num_equations * (num_components-1) / num_components ),
                                                                step=num_components-1,
                                                                dtype="i")))
        return vel_comp_DOF, vel_comp_DOF_vel

    def create_no_dirichlet_bdy_nodes_is(self,
                                         ownership_range,
                                         num_equations,
                                         num_components,
                                         bdy_nodes):
        """Build block velocity DOF arrays excluding Dirichlet bdy nodes.

        Parameters
        ----------
        bdy_nodes : lst
           This is a list of lists with the local dof index for strongly
           enforced Dirichlet boundary conditions on the velocity
           components.
        """
        strong_DOF , local_vel_DOF = self.create_vel_DOF_IS(ownership_range,
                                                            num_equations,
                                                            num_components)
        strong_DOF = [ele.array for ele in strong_DOF]
        local_vel_DOF = [ele.array for ele in local_vel_DOF]
        mask = [numpy.ones(len(var), dtype=bool) for var in strong_DOF]
        for i, bdy_node in enumerate(bdy_nodes):
            mask[i][bdy_node] = False
        strong_DOF = [strong_DOF[i][mask[i]] for i in range(len(strong_DOF))]
        total_vars = int(0)
        for var in strong_DOF:
            total_vars += int(len(var))
        strong_DOF_idx = numpy.empty((total_vars),dtype='int32')
        for i, var_dof in enumerate(strong_DOF):
            strong_DOF_idx[i::2] = var_dof
        return self.create_IS(strong_DOF_idx)

class ModelInfo(object):
    """
    This class stores the model information needed to initialize a
    Schur preconditioner class.

    Parameters
    ----------
    num_components: int
        The number of model components
    dof_order_type: str
        String variable with the dof_order_type ('blocked' or
        'interlaced')
    n_DOF_pressure: int
        Number of pressure degrees of freedom (required for blocked
        dof_order_type)
    bdy_null_space : bool
        Indicates whether boundary condition creates a global null
        space
    """
    def __init__(self,
                 dof_order_type,
                 num_components,
                 L_range = None,
                 neqns = None,
                 n_DOF_pressure = None,
                 bdy_null_space = False):
        self.set_num_components(num_components)
        self.set_dof_order_type(dof_order_type)
        self.const_null_space = bdy_null_space
        if dof_order_type=='blocked':
            assert n_DOF_pressure!=None, \
                "need num of pressure unknowns for blocked dof order type"
            self.dof_order_class = BlockedDofOrderType(n_DOF_pressure, self)
        if dof_order_type=='interlaced':
            self.dof_order_class = InterlacedDofOrderType(self)

    def set_dof_order_type(self, dof_order_type):
        self._dof_order_type = dof_order_type

    def get_dof_order_type(self):
        return self._dof_order_type

    def set_num_components(self,nc):
        self.nc = nc

    def get_num_components(self):
        return self.nc

    def get_dof_order_class(self):
        return self.dof_order_class

class SchurPrecon(KSP_Preconditioner):
    """ Base class for PETSc Schur complement preconditioners. """
    def __init__(self,
                 L,
                 prefix=None,
                 solver_info=None):
        """
        Initializes the Schur complement preconditioner for use with PETSc.

        This class creates a KSP PETSc solver object and initializes flags the
        pressure and velocity unknowns for a general saddle point problem.

        Parameters
        ----------
        L : provides the definition of the problem.
        prefix : str
            Prefix identifier for command line PETSc options.
        solver_info: :class:`ModelInfo`
        """
        self.PCType = 'schur'
        self.L = L
        self._initializePC(prefix)

        if solver_info==None:
            self._initialize_without_solver_info()
        else:
            self.model_info = solver_info

        self._initializeIS()
        self.pc.setFromOptions()
        if self.model_info.get_dof_order_type() == 'interlaced':
            ## TODO: ARB - this should be extended to blocked types
            self.set_velocity_var_names()

    def _initialize_without_solver_info(self):
        """
        Initializes the ModelInfo needed to create a Schur Complement
        preconditioner.
        """
        nc = self.L.pde.nc
        L_range = self.L.getOwnershipRanges()
        neqns = self.L.getSizes()[0][0]
        if len(self.L.pde.u[0].dof) == len(self.L.pde.u[1].dof):
            self.model_info = ModelInfo('interlaced',
                                        nc,
                                        L_range = L_range,
                                        neqns = neqns)
        else:
            self.model_info = ModelInfo('blocked',
                                        nc,
                                        L_range,
                                        neqns,
                                        self.L.pde.u[0].dof.size)
        # ARB - this will come from coefficients, not pde.

    def get_num_components(self):
        val = int(self.isv.size / self.isp.size + 1)
        return val

    def set_velocity_var_names(self):
        nc = self.get_num_components()
        var_names = ('u','v','w')
        self._var_names = [var_names[i] for i in range(nc-1)]

    def get_velocity_var_names(self):
        return self._var_names

    def setUp(self,
              global_ksp,
              newton_its=None):
        """
        Set up the NavierStokesSchur preconditioner.

        Nothing needs to be done here for a generic NSE preconditioner.
        Preconditioner arguments can be set with PETSc command line.
        """
        self._setSchurlog(global_ksp)

    def _setSchurlog(self,global_ksp):
        """ Helper function that attaches a residual log to the inner solve """
        try:
            global_ksp.pc.getFieldSplitSubKSP()[1].setConvergenceTest(self._converged_trueRes)
        except:
            logEvent('Unable to access Schur sub-blocks. Make sure petsc '\
                     'options are consistent with your preconditioner type.')
            exit(1)

    def _setSchurApproximation(self,global_ksp):
        """ Set the Schur approximation to the Schur block.

        Parameters
        ----------
        global_ksp :
        """
        assert self.matcontext_inv is not None, "no matrix context has been set."
        global_ksp.pc.getFieldSplitSubKSP()[1].pc.setType('python')
        global_ksp.pc.getFieldSplitSubKSP()[1].pc.setPythonContext(self.matcontext_inv)
        global_ksp.pc.getFieldSplitSubKSP()[1].pc.setUp()

    def _initializePC(self,
                      prefix):
        r"""
        Intiailizes the PETSc precondition.

        Parameters
        ----------
        prefix : str
            Prefix identifier for command line PETSc options.
        """
        self.pc = p4pyPETSc.PC().create()
        self.pc.setOptionsPrefix(prefix)
        self.pc.setType('fieldsplit')

    def _initializeIS(self):
        r"""Sets the index set (IP) for the pressure and velocity

        Notes
        -----
        Proteus orders unknown degrees of freedom for saddle point
        problems as blocked or end-to-end. Blocked systems are used
        for equal order finite element spaces (e.g. P1-P1).  In this
        case, the degrees of freedom are interlaced (e.g. p[0], u[0],
        v[0], p[1], u[1], v[1], ...).
        """
        L_range = self.L.getOwnershipRange()
        neqns = self.L.getSizes()[0][0]
        dof_order_cls = self.model_info.get_dof_order_class()
        dof_arrays = dof_order_cls.create_DOF_lists(L_range,
                                                    neqns,
                                                    self.model_info.nc)
        self.isp = p4pyPETSc.IS()
        self.isp.createGeneral(dof_arrays[1],comm=p4pyPETSc.COMM_WORLD)
        self.isv = p4pyPETSc.IS()
        self.isv.createGeneral(dof_arrays[0],comm=p4pyPETSc.COMM_WORLD)
        self.pc.setFieldSplitIS(('velocity',self.isv),('pressure',self.isp))

    def _converged_trueRes(self,ksp,its,rnorm):
        """ Function handle to feed to ksp's setConvergenceTest  """
        r_work = ksp.getOperators()[1].createVecLeft()
        ksp.buildResidual(r_work)
        truenorm = r_work.norm()
        if its == 0:
            self.rnorm0 = truenorm
            logEvent("NumericalAnalytics KSPSchurResidual: %12.5e" %(truenorm), level=7)
            logEvent("NumericalAnalytics KSPSchurResidual(relative): %12.5e" %(truenorm/self.rnorm0), level=7 )
            logEvent("        KSP it %i norm(r) = %e  norm(r)/|b| = %e ; atol=%e rtol=%e " % (its,
                                                                                              truenorm,
                                                                                              (truenorm/self.rnorm0),
                                                                                              ksp.atol,
                                                                                              ksp.rtol))
            return False
        else:
            logEvent("NumericalAnalytics KSPSchurResidual: %12.5e" %(truenorm), level=7)
            logEvent("NumericalAnalytics KSPSchurResidual(relative): %12.5e" %(truenorm/self.rnorm0), level=7)
            logEvent("        KSP it %i norm(r) = %e  norm(r)/|b| = %e ; atol=%e rtol=%e " % (its,
                                                                                              truenorm,
                                                                                              (truenorm/self.rnorm0),
                                                                                              ksp.atol,
                                                                                              ksp.rtol))
            if truenorm < self.rnorm0*ksp.rtol:
                return p4pyPETSc.KSP.ConvergedReason.CONVERGED_RTOL
            if truenorm < ksp.atol:
                return p4pyPETSc.KSP.ConvergedReason.CONVERGED_ATOL
        return False

    def _get_null_space_cls(self):
        current_module = sys.modules[__name__]
        null_space_cls_name = self.L.pde.coefficients.nullSpace
        null_space_cls = getattr(current_module,
                                 null_space_cls_name)
        return null_space_cls

    def _is_const_pressure_null_space(self):
        if self.model_info==None:
            if self._get_null_space_cls().get_name == 'constant_pressure':
                return True
            else:
                return False
        else:
            return self.model_info.const_null_space

class Schur_Qp(SchurPrecon) :
    """
    A Navier-Stokes (or Stokes) preconditioner which uses the
    viscosity scaled pressure mass matrix.
    """
    def __init__(self,L,prefix=None,bdyNullSpace=False):
        """
        Initializes the pressure mass matrix class.

        Parameters
        ---------
        L : petsc4py matrix
            Defines the problem's operator.
        """
        SchurPrecon.__init__(self,L,prefix,bdyNullSpace)
        self.operator_constructor = SchurOperatorConstructor(self)
        self.Q = self.operator_constructor.initializeQ()

    def setUp(self,
              global_ksp,
              newton_its=None):
        """ Attaches the pressure mass matrix to PETSc KSP preconditioner.

        Parameters
        ----------
        global_ksp : PETSc KSP object
        """
        # Create the pressure mass matrix and scaxle by the viscosity.
        self.operator_constructor.updateQ()
        self.Qp = self.Q.createSubMatrix(self.operator_constructor.linear_smoother.isp,
                                      self.operator_constructor.linear_smoother.isp)
        self.Qp.scale(1./self.L.pde.coefficients.nu)
        L_sizes = self.Qp.size

        # Setup a PETSc shell for the inverse Qp operator
        self.QpInv_shell = p4pyPETSc.Mat().create()
        self.QpInv_shell.setSizes(L_sizes)
        self.QpInv_shell.setType('python')
        self.matcontext_inv = MatrixInvShell(self.Qp)
        self.QpInv_shell.setPythonContext(self.matcontext_inv)
        self.QpInv_shell.setUp()
        # Set PETSc Schur operator
        self._setSchurApproximation(global_ksp)
        self._setSchurlog(global_ksp)

class NavierStokesSchur(SchurPrecon):
    r""" Schur complement preconditioners for Navier-Stokes problems.

    This class is derived from SchurPrecond and serves as the base
    class for all NavierStokes preconditioners which use the Schur complement
    method.
    """
    def __init__(self,
                 L,
                 prefix=None,
                 velocity_block_preconditioner=True,
                 solver_info=None):
        SchurPrecon.__init__(self,
                             L,
                             prefix=prefix,
                             solver_info=solver_info)
        self.operator_constructor = SchurOperatorConstructor(self,
                                                             pde_type='navier_stokes')
        self.velocity_block_preconditioner = velocity_block_preconditioner
        if self.velocity_block_preconditioner:
            self._initialize_velocity_idx()

    def _initialize_velocity_idx(self):
        """
        This function creates index sets so that a block
        preconditioner ca be used for the velocity solve. One index
        set (e.g. is_vel_*) describes the global dof associated
        with the * component of the velocity.  The second
        is (e.g. is*_local) describes the local dof
        indexes relative to the velocity block.
        """
        L_range = self.L.getOwnershipRange()
        neqns = self.L.getSizes()[0][0]

        vel_is_func = self.model_info.dof_order_class.create_vel_DOF_IS

        velocity_DOF_full, velocity_DOF_local = vel_is_func(L_range,
                                                            neqns,
                                                            self.model_info.nc)

        for i, var in enumerate(self.get_velocity_var_names()):
            name_1 = "is_vel_" + var
            name_2 = "is"+var+"_local"
            setattr(self,name_1, velocity_DOF_full[i])
            setattr(self,name_2, velocity_DOF_local[i])

    def _initialize_velocity_block_preconditioner(self,global_ksp):
        r""" Initialize the velocity block preconditioner.

        """
        global_ksp.pc.getFieldSplitSubKSP()[0].pc.setType('fieldsplit')
        is_lst = []
        for var in self.get_velocity_var_names():
            is_local_name = "is" + var + "_local"
            is_local = getattr(self,is_local_name)
            is_lst.append((var,is_local))
        global_ksp.pc.getFieldSplitSubKSP()[0].pc.setFieldSplitIS(*is_lst)
        # ARB - need to run some tests to see what the best option is here
#        global_ksp.pc.getFieldSplitSubKSP()[0].pc.setFieldSplitType(1)  # This is for additive (e.g. Jacobi)

    def _setup_velocity_block_preconditioner(self,global_ksp):
        r"""To improve the effiency of the velocity-block solve in the
            Schur complement preconditioner, we can apply a block
            preconditioner.  This function builds an index set to
            support this.

        Parameters
        ----------
        global_ksp : xxx
           xxx

        Notes
        -----
        This is currently only set up for interlaced DOF ordering.
        """
        self.velocity_sub_matrix = global_ksp.getOperators()[0].createSubMatrix(self.isv,self.isv)

        for i, var in enumerate(self.get_velocity_var_names()):
            name_str = "is_vel_" + var
            name_str_mat = "velocity_" + var + "_sub_matrix"
            is_set = getattr(self, name_str)
            setattr(self,name_str_mat, global_ksp.getOperators()[0].createSubMatrix(is_set,
                                                                                 is_set))
            global_ksp.pc.getFieldSplitSubKSP()[0].pc.getFieldSplitSubKSP()[i].setOperators(getattr(self,name_str_mat),
                                                                                            getattr(self,name_str_mat))
            global_ksp.pc.getFieldSplitSubKSP()[0].pc.getFieldSplitSubKSP()[i].setFromOptions()
            global_ksp.pc.getFieldSplitSubKSP()[0].pc.getFieldSplitSubKSP()[i].setUp()


        global_ksp.pc.getFieldSplitSubKSP()[0].setUp()
        global_ksp.pc.setUp()

class Schur_Sp(NavierStokesSchur):
    """
    Implements the SIMPLE Schur complement approximation proposed
    in 2009 by Rehman, Vuik and Segal.

    Parameters
    ----------
    L: :class:`p4pyPETSc.Mat`
    prefix: str
        Specifies PETSc preconditioner prefix for setting options

    Notes
    -----
    This Schur complement approximation is also avaliable in PETSc
    by the name selfp.

    One drawback of this operator is that it must be constructed from
    the component pieces.  For small problems this is okay,
    but for large problems this process may not scale well and often
    a pure Laplace operator will prove a more effective choice of
    preconditioner.
    """
    def __init__(self,
                 L,
                 prefix,
                 velocity_block_preconditioner=False,
                 solver_info=None):
        super(Schur_Sp, self).__init__(L,
                                       prefix,
                                       velocity_block_preconditioner,
                                       solver_info=solver_info)
        if self.velocity_block_preconditioner:
            self.velocity_block_preconditioner_set = False

    def setUp(self,
              global_ksp,
              newton_its=None):
        try:
            if self.velocity_block_preconditioner_set is False:
                self._initialize_velocity_block_preconditioner(global_ksp)
                self.velocity_block_preconditioner_set = True
        except AttributeError:
            pass

        self._setSchurlog(global_ksp)

        self.A00 = global_ksp.getOperators()[0].createSubMatrix(self.isv,
                                                                self.isv)
        self.A01 = global_ksp.getOperators()[0].createSubMatrix(self.isv,
                                                                self.isp)
        self.A10 = global_ksp.getOperators()[0].createSubMatrix(self.isp,
                                                                self.isv)
        self.A11 = global_ksp.getOperators()[0].createSubMatrix(self.isp,
                                                                self.isp)
        L_sizes = self.isp.sizes
        self.SpInv_shell = p4pyPETSc.Mat().create()
        self.SpInv_shell.setSizes(L_sizes)
        self.SpInv_shell.setType('python')
        constNullSpace = self._is_const_pressure_null_space()
        self.matcontext_inv = SpInv_shell(self.A00,
                                          self.A11,
                                          self.A01,
                                          self.A10,
                                          constNullSpace)
        self.SpInv_shell.setPythonContext(self.matcontext_inv)
        self.SpInv_shell.setUp()
        # Set PETSc Schur operator
        global_ksp.pc.getFieldSplitSubKSP()[1].pc.setType('python')
        global_ksp.pc.getFieldSplitSubKSP()[1].pc.setPythonContext(self.matcontext_inv)
        global_ksp.pc.getFieldSplitSubKSP()[1].pc.setUp()

class Schur_Qp(SchurPrecon) :
    """
    A Navier-Stokes (or Stokes) preconditioner which uses the
    viscosity scaled pressure mass matrix.
    """
    def __init__(self,
                 L,
                 prefix=None):
        """
        Initializes the pressure mass matrix class.

        Parameters
        ---------
        L : petsc4py matrix
            Defines the problem's operator.
        prefix : str
            Specifies PETSc preconditioner prefix for setting options
        """
        SchurPrecon.__init__(self,
                             L,
                             prefix)
        self.operator_constructor = SchurOperatorConstructor(self)
        self.Q = self.operator_constructor.initializeQ()

    def setUp(self,
              global_ksp,
              newton_its=None):
        """ Attaches the pressure mass matrix to PETSc KSP preconditioner.

        Parameters
        ----------
        global_ksp : PETSc KSP object
        """
        # Create the pressure mass matrix and scaxle by the viscosity.
        self.operator_constructor.updateQ()
        self.Qp = self.Q.createSubMatrix(self.operator_constructor.linear_smoother.isp,
                                         self.operator_constructor.linear_smoother.isp)
        self.Qp.scale(1./self.L.pde.coefficients.nu)
        L_sizes = self.Qp.size

        # Setup a PETSc shell for the inverse Qp operator
        self.QpInv_shell = p4pyPETSc.Mat().create()
        self.QpInv_shell.setSizes(L_sizes)
        self.QpInv_shell.setType('python')
        self.matcontext_inv = MatrixInvShell(self.Qp)
        self.QpInv_shell.setPythonContext(self.matcontext_inv)
        self.QpInv_shell.setUp()
        # Set PETSc Schur operator
        self._setSchurApproximation(global_ksp)
        self._setSchurlog(global_ksp)

class NavierStokesSchur(SchurPrecon):
    r""" Schur complement preconditioners for Navier-Stokes problems.

    This class is derived from SchurPrecond and serves as the base
    class for all NavierStokes preconditioners which use the Schur complement
    method.
    """
    def __init__(self,
                 L,
                 prefix=None,
                 velocity_block_preconditioner=True,
                 solver_info=None):
        """
        Initializes a base class for Navier-Stokes Schur complement
        preconditioners.

        Parameters
        ---------
        L : petsc4py matrix
            Defines the problem's operator
        prefix : str
            Specifies PETSc preconditioner prefix for setting options
        velocity_block_preconditioner : Bool
            Indicates whether the velocity block should be solved as
            a block preconditioner
        """
        SchurPrecon.__init__(self,
                             L,
                             prefix,
                             solver_info=solver_info)
        self.operator_constructor = SchurOperatorConstructor(self,
                                                             pde_type='navier_stokes')
        self.velocity_block_preconditioner = velocity_block_preconditioner
        if self.velocity_block_preconditioner:
            self._initialize_velocity_idx()

    def _initialize_velocity_idx(self):
        """
        This function creates index sets so that a block
        preconditioner ca be used for the velocity solve. One index
        set (e.g. is_vel_*) describes the global dof associated
        with the * component of the velocity.  The second
        is (e.g. is*_local) describes the local dof
        indexes relative to the velocity block.
        """
        L_range = self.L.getOwnershipRange()
        neqns = self.L.getSizes()[0][0]

        vel_is_func = self.model_info.dof_order_class.create_vel_DOF_IS

        velocity_DOF_full, velocity_DOF_local = vel_is_func(L_range,
                                                            neqns,
                                                            self.model_info.nc)

        for i, var in enumerate(self.get_velocity_var_names()):
            name_1 = "is_vel_" + var
            name_2 = "is"+var+"_local"
            setattr(self,name_1, velocity_DOF_full[i])
            setattr(self,name_2, velocity_DOF_local[i])

    def _initialize_velocity_block_preconditioner(self,global_ksp):
        r""" Initialize the velocity block preconditioner.

        """
        global_ksp.pc.getFieldSplitSubKSP()[0].pc.setType('fieldsplit')
        is_lst = []
        for var in self.get_velocity_var_names():
            is_local_name = "is" + var + "_local"
            is_local = getattr(self,is_local_name)
            is_lst.append((var,is_local))
        global_ksp.pc.getFieldSplitSubKSP()[0].pc.setFieldSplitIS(*is_lst)
        # ARB - need to run some tests to see what the best option is here
#        global_ksp.pc.getFieldSplitSubKSP()[0].pc.setFieldSplitType(1)  # This is for additive (e.g. Jacobi)

    def _setup_velocity_block_preconditioner(self,global_ksp):
        r"""To improve the effiency of the velocity-block solve in the
            Schur complement preconditioner, we can apply a block
            preconditioner.  This function builds an index set to
            support this.

        Parameters
        ----------
        global_ksp : xxx
           xxx

        Notes
        -----
        This is currently only set up for interlaced DOF ordering.
        """
        self.velocity_sub_matrix = global_ksp.getOperators()[0].createSubMatrix(self.isv,self.isv)

        for i, var in enumerate(self.get_velocity_var_names()):
            name_str = "is_vel_" + var
            name_str_mat = "velocity_" + var + "_sub_matrix"
            is_set = getattr(self, name_str)
            setattr(self,name_str_mat, global_ksp.getOperators()[0].createSubMatrix(is_set,
                                                                                 is_set))
            global_ksp.pc.getFieldSplitSubKSP()[0].pc.getFieldSplitSubKSP()[i].setOperators(getattr(self,name_str_mat),
                                                                                            getattr(self,name_str_mat))
            global_ksp.pc.getFieldSplitSubKSP()[0].pc.getFieldSplitSubKSP()[i].setFromOptions()
            global_ksp.pc.getFieldSplitSubKSP()[0].pc.getFieldSplitSubKSP()[i].setUp()


        global_ksp.pc.getFieldSplitSubKSP()[0].setUp()
        global_ksp.pc.setUp()

class NavierStokes_TwoPhasePCD(NavierStokesSchur):
    r""" Two-phase PCD Schur complement approximation class.
         Details of this operator are in the forthcoming paper
         'Preconditioners for Two-Phase Incompressible Navier-Stokes
         Flow', Bootland et. al. 2017.

         Since the two-phase Navier-Stokes problem used in the MPRANS
         module of Proteus
         has some additional features not include in the above paper,
         a few additional flags and options are avaliable.

         * density scaling - This flag allows the user to specify
           whether the advection and mass terms in the second term
           of the PCD operator should use the actual density or the
           scale with the number one.

         * numerical viscosity - This flag specifies whether the
           additional viscosity introduced from shock capturing
           stabilization should be included as part of the viscosity
           coefficient.

         * mass form - This flag allows the user to specify what form
           the mass matrix takes, lumped (True) or full (False).

         * number chebyshev its - This integer allows the user to
           specify how many Chebyshev its to use if a full mass matrix
           is used and a direct solver is not applied.
    """
    def __init__(self,
                 L,
                 prefix = None,
                 density_scaling = True,
                 numerical_viscosity = True,
                 lumped = True,
                 num_chebyshev_its = 0,
                 laplace_null_space = True,
                 velocity_block_preconditioner=False):
        """
        Initialize the two-phase PCD preconditioning class.

        Parameters
        ----------
        L : petsc4py Matrix
            Defines the problem's operator.
        prefix : str
            String allowing PETSc4py options.
        density_scaling : bool
            Indicates whether mass and advection terms should be
            scaled with the density (True) or 1 (False).
        numerical_viscosity : bool
            Indicates whether the viscosity used to calculate
            the inverse scaled mass matrix should include numerical
            viscosity (True) or not (False).
        lumped : bool
            Indicates whether the viscosity and density mass matrices
            should be lumped (True) or full (False).
        num_chebyshev_its : int
            This flag allows the user to apply the mass matrices with
            a chebyshev semi-iteration.  0  indicates the semi-
            iteration should not be used, where as a number 1,2,...
            indicates the number of iterations the method should take.
        laplace_null_space : bool
            Indicates whether the laplace operator inside the
            two-phase PCD operator has a constant null space.
        velocity_block_preconditioner : bool
            Indicates whether to use a block preconditioner for the
            velocity solve.
        """
        NavierStokesSchur.__init__(self,
                                   L,
                                   prefix,
                                   velocity_block_preconditioner)
        # Initialize the discrete operators
        self.N_rho = self.operator_constructor.initializeTwoPhaseCp_rho()
        self.A_invScaledRho = self.operator_constructor.initializeTwoPhaseInvScaledAp()
        self.Q_rho = self.operator_constructor.initializeTwoPhaseQp_rho()
        self.Q_invScaledVis = self.operator_constructor.initializeTwoPhaseInvScaledQp()
        # TP PCD scaling options
        self.density_scaling = density_scaling
        self.numerical_viscosity = numerical_viscosity
        self.lumped = lumped
        self.num_chebyshev_its = num_chebyshev_its
        self.laplace_null_space = laplace_null_space
        # Strong Dirichlet Pressure DOF
        try:
            self.strongPressureDOF = list(L.pde.dirichletConditionsForceDOF[0].DOFBoundaryPointDict.keys())
        except KeyError:
            self.strongPressureDOF = []
        if self.velocity_block_preconditioner:
            self.velocity_block_preconditioner_set = False

    def setUp(self,
              global_ksp,
              newton_its=None):
        from . import Comm
        comm = Comm.get()

        isp = self.operator_constructor.linear_smoother.isp
        isv = self.operator_constructor.linear_smoother.isv

        self.operator_constructor.updateNp_rho(density_scaling = self.density_scaling)
        self.Np_rho = self.N_rho.createSubMatrix(isp,
                                                 isp)

        if newton_its == 0:
            self.operator_constructor.updateInvScaledAp()
            self.operator_constructor.updateTwoPhaseQp_rho(density_scaling = self.density_scaling,
                                                           lumped = self.lumped)
            self.operator_constructor.updateTwoPhaseInvScaledQp_visc(numerical_viscosity = self.numerical_viscosity,
                                                                     lumped = self.lumped)
            self.Ap_invScaledRho = self.A_invScaledRho.createSubMatrix(isp,
                                                                       isp)
            self.Qp_rho = self.Q_rho.createSubMatrix(isp,
                                                     isp)
            self.Qp_invScaledVis = self.Q_invScaledVis.createSubMatrix(isp,
                                                                       isp)

        if newton_its == 0:
            self.operator_constructor.updateInvScaledAp()
            self.operator_constructor.updateTwoPhaseQp_rho(density_scaling = self.density_scaling,
                                                           lumped = self.lumped)
            self.operator_constructor.updateTwoPhaseInvScaledQp_visc(numerical_viscosity = self.numerical_viscosity,
                                                                     lumped = self.lumped)
            self.Ap_invScaledRho = self.A_invScaledRho.createSubMatrix(isp,
                                                                    isp)
            self.Qp_rho = self.Q_rho.createSubMatrix(isp,
                                                  isp)
            self.Qp_invScaledVis = self.Q_invScaledVis.createSubMatrix(isp,
                                                                    isp)

        # ****** Sp for Ap *******
        # TODO - This is included for a possible extension which exchanges Ap with Sp for short
        #        time steps.
        # A_mat = global_ksp.getOperators()[0]
        # self.A00 = A_mat.createSubMatrix(isv,
        #                               isv)
        # self.A01 = A_mat.createSubMatrix(isv,
        #                               isp)
        # self.A10 = A_mat.createSubMatrix(isp,
        #                               isv)
        # self.A11 = A_mat.createSubMatrix(isp,
        #                               isp)

        # dt = self.L.pde.timeIntegration.t - self.L.pde.timeIntegration.tLast
        # self.A00_inv = petsc_create_diagonal_inv_matrix(self.A00)
        # A00_invBt = self.A00_inv.matMult(self.A01)
        # self.Sp = self.A10.matMult(A00_invBt)
        # self.Sp.scale(- 1. )
        # self.Sp.axpy( 1. , self.A11)

        # End ******** Sp for Ap ***********

        try:
            if self.velocity_block_preconditioner_set is False:
                self._initialize_velocity_block_preconditioner(global_ksp)
                self.velocity_block_preconditioner_set = True
        except AttributeError:
            pass

        if self.velocity_block_preconditioner:
            self._setup_velocity_block_preconditioner(global_ksp)

        L_sizes = self.Qp_rho.size
        L_range = self.Qp_rho.owner_range

        self.TP_PCDInv_shell = p4pyPETSc.Mat().create()
        self.TP_PCDInv_shell.setSizes(L_sizes)
        self.TP_PCDInv_shell.setType('python')
        dt = self.L.pde.timeIntegration.t - self.L.pde.timeIntegration.tLast

        self.matcontext_inv = TwoPhase_PCDInv_shell(self.Qp_invScaledVis,
                                                    self.Qp_rho,
                                                    self.Ap_invScaledRho,
                                                    self.Np_rho,
                                                    True,
                                                    dt,
                                                    num_chebyshev_its = self.num_chebyshev_its,
                                                    strong_dirichlet_DOF = self.strongPressureDOF,
                                                    laplace_null_space = self.laplace_null_space,
                                                    par_info = self.L.pde.par_info)
        self.TP_PCDInv_shell.setPythonContext(self.matcontext_inv)
        self.TP_PCDInv_shell.setUp()
        global_ksp.pc.getFieldSplitSubKSP()[1].pc.setType('python')
        global_ksp.pc.getFieldSplitSubKSP()[1].pc.setPythonContext(self.matcontext_inv)
        global_ksp.pc.getFieldSplitSubKSP()[1].pc.setUp()

        if self.velocity_block_preconditioner:
            self._setup_velocity_block_preconditioner(global_ksp)

        global_ksp.pc.getFieldSplitSubKSP()[0].pc.setUp()

        self._setSchurlog(global_ksp)
        self._get_null_space_cls().apply_to_schur_block(global_ksp)
        self._setSchurlog(global_ksp)

class Schur_LSC(SchurPrecon):
    """
    The Least-Squares Communtator preconditioner for saddle
    point problems.
    """
    def __init__(self,
                 L,
                 prefix=None):
        """
        Initializes the pressure mass matrix class.

        Parameters
        ---------
        L : petsc4py matrix
            Defines the problem's operator.
        prefix : str
            Specifies PETSc preconditioner prefix for setting options
        """        
        SchurPrecon.__init__(self,
                             L,
                             prefix)
        self.operator_constructor = SchurOperatorConstructor(self)
        self.Q = self.operator_constructor.initializeQ()

    def setUp(self,
              global_ksp,
              newton_its=None):
        self.operator_constructor.updateQ()
        self.Qv = self.Q.createSubMatrix(self.operator_constructor.linear_smoother.isv,
                                      self.operator_constructor.linear_smoother.isv)
        self.Qv_hat = p4pyPETSc.Mat().create()
        self.Qv_hat.setSizes(self.Qv.getSizes())
        self.Qv_hat.setType('aij')
        self.Qv_hat.setUp()
        self.Qv_hat.setDiagonal(self.Qv.getDiagonal())

        self.B = global_ksp.getOperators()[0].createSubMatrix(self.isp,self.isv)
        self.F = global_ksp.getOperators()[0].createSubMatrix(self.isv,self.isv)
        self.Bt = global_ksp.getOperators()[0].createSubMatrix(self.isv,self.isp)

        self.matcontext_inv = LSCInv_shell(self.Qv_hat,self.B,self.Bt,self.F)

        self._setSchurApproximation(global_ksp)
        self._setSchurlog(global_ksp)

class NavierStokes3D(NavierStokesSchur):
    def __init__(self,
                 L,
                 prefix=None,
                 velocity_block_preconditioner=False):
        """
        Initializes a base class for Navier-Stokes Schur complement
        preconditioners.

        Parameters
        ---------
        L : petsc4py matrix
            Defines the problem's operator
        prefix : str
            Specifies PETSc preconditioner prefix for setting options
        velocity_block_preconditioner : Bool
            Indicates whether the velocity block should be solved as
            a block preconditioner
        """                
        NavierStokesSchur.__init__(self,
                                   L,
                                   prefix,
                                   velocity_block_preconditioner=velocity_block_preconditioner)
        if self.velocity_block_preconditioner:
            self.velocity_block_preconditioner_set = False

    def setUp(self,
              global_ksp=None,
              newton_its=None):
        try:
            if self.velocity_block_preconditioner_set is False:
                self._initialize_velocity_block_preconditioner(global_ksp)
                self.velocity_block_preconditioner_set = True
        except AttributeError:
            pass

        if self.velocity_block_preconditioner:
            self._setup_velocity_block_preconditioner(global_ksp)

        self._get_null_space_cls().apply_to_schur_block(global_ksp)


SimpleNavierStokes3D = NavierStokes3D

class SimpleDarcyFC(object):
    def __init__(self,L):
        L_sizes = L.getSizes()
        L_range = L.getOwnershipRange()
        print("L_sizes",L_sizes)
        neqns = L_sizes[0][0]
        print("neqns",neqns)
        self.saturationDOF = numpy.arange(L_range[0],L_range[0]+neqns/2,dtype="i")
        #print "saturation",self.saturationDOF
        self.pressureDOF = numpy.arange(L_range[0]+neqns/2,L_range[0]+neqns,dtype="i")
        #print "pressure",self.pressureDOF
        self.pc = p4pyPETSc.PC().create()
        self.pc.setType('fieldsplit')
        self.isp = p4pyPETSc.IS()
        self.isp.createGeneral(self.saturationDOF,comm=p4pyPETSc.COMM_WORLD)
        self.isv = p4pyPETSc.IS()
        self.isv.createGeneral(self.pressureDOF,comm=p4pyPETSc.COMM_WORLD)
        self.pc.setFieldSplitIS(self.isp)
        self.pc.setFieldSplitIS(self.isv)

    def setUp(self,
              global_ksp=None,
              newton_its=None):
        pass

class NavierStokes2D(NavierStokesSchur):
    def __init__(self,
                 L,
                 prefix=None,
                 velocity_block_preconditioner=False):
        """
        Initializes a base class for Navier-Stokes Schur complement
        preconditioners.

        Parameters
        ---------
        L : petsc4py matrix
            Defines the problem's operator
        prefix : str
            Specifies PETSc preconditioner prefix for setting options
        velocity_block_preconditioner : Bool
            Indicates whether the velocity block should be solved as
            a block preconditioner
        """        
        NavierStokesSchur.__init__(self,
                                   L,
                                   prefix,
                                   velocity_block_preconditioner=velocity_block_preconditioner)
        if self.velocity_block_preconditioner:
            self.velocity_block_preconditioner_set = False

    def setUp(self,
              global_ksp=None,
              newton_its=None):
        try:
            if self.velocity_block_preconditioner_set is False:
                self._initialize_velocity_block_preconditioner(global_ksp)
                self.velocity_block_preconditioner_set = True
        except AttributeError:
            pass

        if self.velocity_block_preconditioner:
            self._setup_velocity_block_preconditioner(global_ksp)

        self._get_null_space_cls().apply_to_schur_block(global_ksp)

SimpleNavierStokes2D = NavierStokes2D

class NavierStokesPressureCorrection(object):
    def __init__(self,L,prefix=None):
        self.L=L
        self.pc = p4pyPETSc.PC().create()
        if prefix:
            self.pc.setOptionsPrefix(prefix)
        self.pc.setFromOptions()
        self.hasNullSpace=True
        self.nsp = p4pyPETSc.NullSpace().create(constant=True,
                                                comm=p4pyPETSc.COMM_WORLD)
        self.L.setOption(p4pyPETSc.Mat.Option.SYMMETRIC, True)
        self.L.setNullSpace(self.nsp)
    def setUp(self,
              global_ksp=None,
              newton_its=None):
        pass

class SimpleDarcyFC(object):
    def __init__(self,L):
        L_sizes = L.getSizes()
        L_range = L.getOwnershipRange()
        neqns = L_sizes[0][0]
        self.saturationDOF = numpy.arange(L_range[0],L_range[0]+neqns/2,dtype="i")
        self.pressureDOF = numpy.arange(L_range[0]+neqns/2,L_range[0]+neqns,dtype="i")
        self.pc = p4pyPETSc.PC().create()
        self.pc.setType('fieldsplit')
        self.isp = p4pyPETSc.IS()
        self.isp.createGeneral(self.saturationDOF,comm=p4pyPETSc.COMM_WORLD)
        self.isv = p4pyPETSc.IS()
        self.isv.createGeneral(self.pressureDOF,comm=p4pyPETSc.COMM_WORLD)
        self.pc.setFieldSplitIS(self.isp)
        self.pc.setFieldSplitIS(self.isv)
    def setUp(self,
              global_ksp=None,
              newton_its=None):
        pass

class Jacobi(LinearSolver):
    """
    Damped Jacobi iteration.
    """
    from . import csmoothers
    def __init__(self,
                 L,
                 weight=1.0,
                 rtol_r  = 1.0e-4,
                 atol_r  = 1.0e-16,
                 rtol_du = 1.0e-4,
                 atol_du = 1.0e-16,
                 maxIts  = 100,
                 norm = l2Norm,
                 convergenceTest = 'r',
                 computeRates = True,
                 printInfo = True):
        LinearSolver.__init__(self,L,
                              rtol_r,
                              atol_r,
                              rtol_du,
                              atol_du,
                              maxIts,
                              norm,
                              convergenceTest,
                              computeRates,
                              printInfo)
        self.solverName = "Jacobi"
        self.M=Vec(self.n)
        self.w=weight
        self.node_order=numpy.arange(self.n,dtype="i")
    def prepare(self,b=None,newton_its=None):
        if type(self.L).__name__ == 'ndarray':
            self.M = self.w/numpy.diagonal(self.L)
        elif type(self.L).__name__ == 'SparseMatrix':
            self.csmoothers.jacobi_NR_prepare(self.L,self.w,1.0e-16,self.M)
    def solve(self,u,r=None,b=None,par_u=None,par_b=None,initialGuessIsZero=False):
        (r,b) = self.solveInitialize(u,r,b,initialGuessIsZero,par_u=par_u)
        while (not self.converged(r) and
               not self.failed()):
            if type(self.L).__name__ == 'ndarray':
                self.du[:]=r
                self.du*=self.M
            elif type(self.L).__name__ == "SparseMatrix":
                self.csmoothers.jacobi_NR_solve(self.L,self.M,r,self.node_order,self.du)
            u -= self.du
            self.refreshGhosts(par_u)
            self.computeResidual(u,r,b)

class GaussSeidel(LinearSolver):
    """
    Damped Gauss-Seidel.
    """
    from . import csmoothers
    def __init__(self,
                 connectionList,
                 L,
                 weight=0.33,
                 sym=False,
                 rtol_r  = 1.0e-4,
                 atol_r  = 1.0e-16,
                 rtol_du = 1.0e-4,
                 atol_du = 1.0e-16,
                 maxIts  = 100,
                 norm = l2Norm,
                 convergenceTest = 'r',
                 computeRates = True,
                 printInfo = True):
        LinearSolver.__init__(self,L,
                              rtol_r,
                              atol_r,
                              rtol_du,
                              atol_du,
                              maxIts,
                              norm,
                              convergenceTest,
                              computeRates,
                              printInfo)
        self.solverName = "Gauss-Seidel"
        self.connectionList=connectionList
        self.M=Vec(self.n)
        self.node_order=numpy.arange(self.n,dtype="i")
        self.w=weight
        self.sym=sym
    def prepare(self,b=None,newton_its=None):
        if type(self.L).__name__ == 'ndarray':
            self.M = self.w/numpy.diagonal(self.L)
        elif type(self.L).__name__ == 'SparseMatrix':
            self.csmoothers.gauss_seidel_NR_prepare(self.L,self.w,1.0e-16,self.M)
            #self.csmoothers.jacobi_NR_prepare(self.L,self.w,1.0e-16,self.M)
    def solve(self,u,r=None,b=None,par_u=None,par_b=None,initialGuessIsZero=False):
        (r,b) = self.solveInitialize(u,r,b,initialGuessIsZero,par_u=par_u)
        while (not self.converged(r) and
               not self.failed()):
            if type(self.L).__name__ == 'ndarray':
                self.du[:]=0.0
                for i in range(self.n):
                    rhat = r[i]
                    for j in self.connectionList[i]:
                        rhat -= self.L[j,i]*self.du[j]
                    self.du[i] = self.M[i]*rhat
                if self.sym == True:
                    u-= self.du
                    self.computeResidual(u,r,b)
                    self.du[:]=0.0
                    for i in range(self.n-1,-1,-1):
                        rhat = self.r[i]
                        for j in self.connectionList[i]:
                            rhat -= self.L[i,j]*self.du[j]
                    self.du[i] = self.M[i]*rhat
            elif type(self.L).__name__ == "SparseMatrix":
                self.csmoothers.gauss_seidel_NR_solve(self.L,self.M,r,self.node_order,self.du)
                #self.csmoothers.jacobi_NR_solve(self.L,self.M,r,self.node_order,self.du)
            u -= self.du
            self.refreshGhosts(par_u)
            self.computeResidual(u,r,b)

class StarILU(LinearSolver):
    """
    Alternating Schwarz Method on node stars.
    """
    from . import csmoothers
    def __init__(self,
                 connectionList,
                 L,
                 weight=1.0,
                 sym=False,
                 rtol_r  = 1.0e-4,
                 atol_r  = 1.0e-16,
                 rtol_du = 1.0e-4,
                 atol_du = 1.0e-16,
                 maxIts  = 100,
                 norm = l2Norm,
                 convergenceTest = 'r',
                 computeRates = True,
                 printInfo = True):
        LinearSolver.__init__(self,L,
                              rtol_r,
                              atol_r,
                              rtol_du,
                              atol_du,
                              maxIts,
                              norm,
                              convergenceTest,
                              computeRates,
                              printInfo)
        self.solverName = "StarILU"
        self.w=weight
        self.sym=sym
        if type(self.L).__name__ == 'ndarray':
            self.connectionList=connectionList
            self.subdomainIndecesList=[]
            self.subdomainSizeList=[]
            self.subdomainL=[]
            self.subdomainR=[]
            self.subdomainDU=[]
            self.subdomainSolvers=[]
            self.globalToSubdomain=[]
            for i in range(self.n):
                self.subdomainIndecesList.append([])
                connectionList[i].sort()
                self.globalToSubdomain.append(dict([(j,J+1) for J,j in
                                                    enumerate(connectionList[i])]))
                self.globalToSubdomain[i][i]=0
                nSubdomain = len(connectionList[i])+1
                self.subdomainR.append(Vec(nSubdomain))
                self.subdomainDU.append(Vec(nSubdomain))
                self.subdomainSizeList.append(len(connectionList[i]))
                self.subdomainL.append(Mat(nSubdomain,nSubdomain))
                for J,j in enumerate(connectionList[i]):
                    self.subdomainIndecesList[i].append(set(connectionList[i]) &
                                                        set(connectionList[j]))
                    self.subdomainIndecesList[i][J].update([i,j])
        elif type(L).__name__ == 'SparseMatrix':
            self.node_order=numpy.arange(self.n,dtype="i")
            self.asmFactorObject = self.csmoothers.ASMFactor(L)
    def prepare(self,b=None,newton_its=None):
        if type(self.L).__name__ == 'ndarray':
            self.subdomainSolvers=[]
            for i in range(self.n):
                self.subdomainL[i][0,0] = self.L[i,i]
                for J,j in enumerate(self.connectionList[i]):
                    #first do row 0 (star center)
                    self.subdomainL[i][J+1,0] = self.L[j,i]
                    #now do boundary rows
                    for k in self.subdomainIndecesList[i][J]:
                        K = self.globalToSubdomain[i][k]
                        self.subdomainL[i][K,J+1]=self.L[k,j]
                self.subdomainSolvers.append(LU(self.subdomainL[i]))
                self.subdomainSolvers[i].prepare()
        elif type(self.L).__name__ == 'SparseMatrix':
            self.csmoothers.asm_NR_prepare(self.L,self.asmFactorObject)
    def solve(self,u,r=None,b=None,par_u=None,par_b=None,initialGuessIsZero=False):
        (r,b) = self.solveInitialize(u,r,b,initialGuessIsZero,par_u=par_u)
        while (not self.converged(r) and
               not self.failed()):
            self.du[:]=0.0
            if type(self.L).__name__ == 'ndarray':
                for i in range(self.n):
                    #load subdomain residual
                    self.subdomainR[i][0] = r[i] - self.L[i,i]*self.du[i]
                    for j in self.connectionList[i]:
                        self.subdomainR[i][0] -= self.L[j,i]*self.du[j]
                    for J,j in enumerate(self.connectionList[i]):
                        self.subdomainR[i][J+1]=r[j] - self.L[j,j]*self.du[j]
                        for k in self.connectionList[j]:
                            self.subdomainR[i][J+1] -= self.L[k,j]*self.du[k]
                    #solve
                    self.subdomainSolvers[i].solve(u=self.subdomainDU[i],
                                                   b=self.subdomainR[i])
                    #update du
                    self.subdomainDU[i]*=self.w
                    self.du[i]+=self.subdomainDU[i][0]
                    for J,j in enumerate(self.connectionList[i]):
                        self.du[j] += self.subdomainDU[i][J+1]
            elif type(self.L).__name__ == 'SparseMatrix':
                #In parallel sweep only the stars centred on nodes this rank
                #owns. A star centred on an owned node is a correct square
                #system -- the owned row carries all of its columns -- while
                #one centred on a ghost node is assembled from a row the
                #subdomain holds only partially, and sweeping those stalls the
                #iteration rather than converging it. The subdomain lists its
                #owned nodes first, so node_order's leading entries are them.
                ownedNodes = par_u.dim_proc if par_u is not None else -1
                self.refreshResidualGhosts(par_u,r)
                self.csmoothers.asm_NR_solve(self.L,self.w,self.asmFactorObject,
                                             self.node_order,r,self.du,ownedNodes)
            u -= self.du
            self.refreshGhosts(par_u)
            self.computeResidual(u,r,b)

class StarBILU(LinearSolver):
    """
    Alternating Schwarz Method on 'blocks' consisting of consectutive rows in system for things like dg ...
    """
    from . import csmoothers
    def __init__(self,
                 connectionList,
                 L,
                 bs=1,
                 weight=1.0,
                 sym=False,
                 rtol_r  = 1.0e-4,
                 atol_r  = 1.0e-16,
                 rtol_du = 1.0e-4,
                 atol_du = 1.0e-16,
                 maxIts  = 100,
                 norm = l2Norm,
                 convergenceTest = 'r',
                 computeRates = True,
                 printInfo = True):
        LinearSolver.__init__(self,L,
                              rtol_r,
                              atol_r,
                              rtol_du,
                              atol_du,
                              maxIts,
                              norm,
                              convergenceTest,
                              computeRates,
                              printInfo)
        self.solverName = "StarBILU"
        self.w=weight
        self.sym=sym
        self.bs = bs
        if type(self.L).__name__ == 'ndarray':
            raise NotImplementedError
        elif type(L).__name__ == 'SparseMatrix':
            self.node_order=numpy.arange(self.n,dtype="i")
            self.basmFactorObject = self.csmoothers.BASMFactor(L,bs)
    def prepare(self,b=None,newton_its=None):
        if type(self.L).__name__ == 'ndarray':
            raise NotImplementedError
        elif type(self.L).__name__ == 'SparseMatrix':
            self.csmoothers.basm_NR_prepare(self.L,self.basmFactorObject)
    def solve(self,u,r=None,b=None,par_u=None,par_b=None,initialGuessIsZero=False):
        (r,b) = self.solveInitialize(u,r,b,initialGuessIsZero,par_u=par_u)
        while (not self.converged(r) and
               not self.failed()):
            #mwf debug
            logEvent("StarBILU norm_r= %s norm_du= %s " % (self.norm_r,self.norm_du))
            self.du[:]=0.0
            if type(self.L).__name__ == 'ndarray':
                raise NotImplementedError
            elif type(self.L).__name__ == 'SparseMatrix':
                self.csmoothers.basm_NR_solve(self.L,self.w,self.basmFactorObject,self.node_order,r,self.du)
            u -= self.du
            self.refreshGhosts(par_u)
            self.computeResidual(u,r,b)
class TwoLevel(LinearSolver):
    """
    A generic two-level multiplicative Schwarz solver.
    """
    def __init__(self,
                 prolong,
                 restrict,
                 coarseL,
                 preSmoother,
                 postSmoother,
                 coarseSolver,
                 L,
                 prepareCoarse=False,
                 rtol_r  = 1.0e-4,
                 atol_r  = 1.0e-16,
                 rtol_du = 1.0e-4,
                 atol_du = 1.0e-16,
                 maxIts  = 100,
                 norm = l2Norm,
                 convergenceTest = 'r',
                 computeRates = True,
                 printInfo = True,
                 par_u_coarse=None):
        LinearSolver.__init__(self,L,
                              rtol_r,
                              atol_r,
                              rtol_du,
                              atol_du,
                              maxIts,
                              norm,
                              convergenceTest,
                              computeRates,
                              printInfo)
        self.solverName = "TwoLevel"
        self.prolong = prolong
        self.restrict = restrict
        self.cL = coarseL
        self.preSmoother = preSmoother
        self.postSmoother = postSmoother
        self.coarseSolver = coarseSolver
        self.cb = Vec(prolong.shape[1])
        self.cr = Vec(prolong.shape[1])
        self.cdu = Vec(prolong.shape[1])
        #The coarse level's work vectors belong to this solver, so nothing else
        #holds a parallel counterpart for them. Without one the coarse solve
        #runs on stale ghost values and the smoothers down there have nothing
        #to refresh, which is what kept a parallel cycle from working.
        if par_u_coarse is not None:
            self.par_cb = par_u_coarse.duplicate_with_array(self.cb)
            self.par_cr = par_u_coarse.duplicate_with_array(self.cr)
            self.par_cdu = par_u_coarse.duplicate_with_array(self.cdu)
        else:
            self.par_cb = self.par_cr = self.par_cdu = None
        self.prepareCoarse=prepareCoarse
    def prepare(self,b=None,newton_its=None):
        self.preSmoother.prepare()
        self.postSmoother.prepare()
        if self.prepareCoarse is True:
            self.coarseSolver.prepare()
    def failed(self):
        """A fixed cycle count is this solver's convergence criterion.

        MGM builds its levels with convergenceTest='its' and maxIts set to the
        requested number of cycles, so ``its == maxIts`` is the *expected* exit,
        not a failure. The inherited failed() reports any such exit as a
        failure, so Newton saw linearSolverFailed on the very first step and
        gave up -- MGM could never drive a nonlinear solve. Linear problems hid
        this because they set maxNonlinearIts=1 and so only ever take the one
        Newton step.

        The iteration counter is still advanced here: solve() uses failed() as
        the loop counter as well as the loop guard.
        """
        if self.convergenceTest == 'its':
            self.its += 1
            return False
        return LinearSolver.failed(self)
    def solve(self,u,r=None,b=None,par_u=None,par_b=None,initialGuessIsZero=False):
        (r,b) = self.solveInitialize(u,r,b,initialGuessIsZero,par_u=par_u)
        while (not self.converged(r) and
               not self.failed()):
            #LinearSolver.solve is (u,r,b,par_u,par_b,initialGuessIsZero):
            #passing initialGuessIsZero positionally put it in the par_u slot,
            #which now matters -- the smoothers refresh their iterate's ghost
            #entries through par_u, and a bool quietly disables that.
            self.preSmoother.solve(u,r,b,par_u=par_u,par_b=par_b,
                                   initialGuessIsZero=initialGuessIsZero)
            initialGuessIsZero=False
            self.restrict.matvec(r,self.cb)
            self.cdu[:]=0.0
            #the coarse level's vectors are TwoLevel's own and have no parallel
            #counterpart yet, so the coarse solve is still serial-only
            self.coarseSolver.solve(u=self.cdu,r=self.cr,b=self.cb,
                                    par_u=self.par_cdu,par_b=self.par_cb,
                                    initialGuessIsZero=True)
            self.prolong.matvec(self.cdu,self.du)
            u-=self.du
            self.refreshGhosts(par_u)
            self.computeResidual(u,r,b)
            self.postSmoother.solve(u,r,b,par_u=par_u,par_b=par_b,
                                    initialGuessIsZero=False)

class MultilevelLinearSolver(object):
    """
    A generic multilevel solver.
    """
    def __init__(self,levelLinearSolverList,computeRates=False,printInfo=False):
        self.printInfo=printInfo
        self.solverList=levelLinearSolverList
        self.nLevels = len(self.solverList)
        self.computeEigenvalues = False
        for l in range(self.nLevels):
            levelLinearSolverList[l].computeRates=computeRates
            levelLinearSolverList[l].printInfo=self.printInfo
    def info(self):
        self.infoString="********************Start Multilevel Linear Solver Info*********************\n"
        for l in range(self.nLevels):
            self.infoString += "**************Start Level %i Info********************\n" % l
            self.infoString += self.solverList[l].info()
            self.infoString += "**************End Level %i Info********************\n" % l
        self.infoString+="********************End Multilevel Linear Solver Info*********************\n"
        return self.infoString
    def failed(self):
        """Newton calls failed() on whatever linear solver it was handed.

        MultilevelLinearSolver does not derive from LinearSolver, so it has to
        answer for itself; the finest level is the one that produced the solve.
        """
        return self.solverList[-1].failed()

class MGM(MultilevelLinearSolver):
    """
    A generic multigrid W cycle.
    """
    def __init__(self,
                 prolongList,
                 restrictList,
                 LList,
                 preSmootherList,
                 postSmootherList,
                 coarseSolver,
                 mgItsList=[],
                 printInfo=False,
                 computeRates=False,
                 par_duList=None):
        self.printInfo=printInfo
        self.nLevels = len(LList)
        self.solverList=[coarseSolver]
        for i in range(1,len(LList)):
            if mgItsList ==[]:
                mgItsList.append(1)
            par_u_coarse = par_duList[i-1] if par_duList is not None else None
            self.solverList.append(TwoLevel(prolong = prolongList[i],
                                            restrict = restrictList[i],
                                            coarseL = LList[i-1],
                                            preSmoother = preSmootherList[i],
                                            postSmoother = postSmootherList[i],
                                            coarseSolver = self.solverList[i-1],
                                            L = LList[i],
                                            maxIts = mgItsList[i],
                                            convergenceTest = 'its',
                                            computeRates=computeRates,
                                            printInfo=False,
                                            par_u_coarse=par_u_coarse))
        self.mgmSolver = self.solverList[self.nLevels-1]
        self.solverName = "TwoLevel"
    def prepare(self,b=None,newton_its=None):
        for s in self.solverList:
            s.prepare()

    def solve(self,u,r=None,b=None,par_u=None,par_b=None,initialGuessIsZero=False):
        #This passed initialGuessIsZero positionally into TwoLevel.solve's
        #par_u slot -- the same slip NI and TwoLevel had. With the smoothers
        #now reading par_u it is not cosmetic: without it they fall back to a
        #norm over the local vector, so the ranks disagree about when to stop,
        #take different numbers of iterations, and deadlock on the first
        #collective after they diverge.
        self.mgmSolver.solve(u=u,r=r,b=b,par_u=par_u,par_b=par_b,
                             initialGuessIsZero=initialGuessIsZero)

class NI(MultilevelLinearSolver):
    """
    A generic nested iteration solver.
    """
    def __init__(self,
                 solverList,
                 prolongList,
                 restrictList,
                 maxIts=None,
                 tolList=None,
                 atol=None,
                 computeRates=True,
                 printInfo=False,
                 par_duList=None):
        self.levelSolverList=solverList
        self.solverList = [self for n in range(len(solverList))]
        MultilevelLinearSolver.__init__(self,self.solverList,computeRates=computeRates,printInfo=printInfo)
        self.prolongList = prolongList
        self.restrictList = restrictList
        self.fineMesh = self.nLevels - 1
        self.uList=[]
        self.bList=[]
        self.levelDict={}
        for l in range(self.fineMesh+1):
            n = solverList[l].n
            self.levelDict[n] = l
            self.uList.append(Vec(n))
            self.bList.append(Vec(n))
        self.levelDict[solverList[-1].n]=self.fineMesh
        self.uList.append([])
        self.bList.append([])
        # Parallel counterparts of the per-level work vectors above. A level
        # solver that talks to PETSc (KSP_petsc4py) needs a ParVec for every
        # solve, not just the one on the finest level, and NI's coarse-level
        # vectors are its own -- they are not the model's. Each one therefore
        # gets a ParVec of its own, over NI's array but with that level's
        # parallel layout, taken from the model's du vector for the level.
        self.par_uList=[]
        self.par_bList=[]
        for l in range(self.fineMesh+1):
            par_du = par_duList[l] if par_duList is not None else None
            if par_du is None:
                self.par_uList.append(None)
                self.par_bList.append(None)
            else:
                self.par_uList.append(par_du.duplicate_with_array(self.uList[l]))
                self.par_bList.append(par_du.duplicate_with_array(self.bList[l]))
        self.maxIts = maxIts
        self.tolList = tolList
        self.atol_r=atol
        self.printInfo=printInfo
        self.infoString=''
    def setResTol(self,rtol,atol):
        if self.tolList is not None:
            for l in range(self.nLevels):
                self.tolList[l] = rtol
            self.atol_r = atol
    def prepare(self,b=None,newton_its=None):
        if b is not None:
            currentMesh = self.levelDict[b.shape[0]]
        else:
            currentMesh = self.fineMesh
        for s in self.levelSolverList[:currentMesh+1]:
            s.prepare()
    def solve(self,u,r=None,b=None,par_u=None,par_b=None,initialGuessIsZero=False):
        currentMesh = self.levelDict[b.shape[0]]
        if currentMesh > 0:
            self.uList[currentMesh][:] = u
            self.bList[currentMesh][:] = b
            for l in range(currentMesh,1,-1):
                if not initialGuessIsZero:
                    self.restrictList[l].matvec(self.uList[l],self.uList[l-1])
                self.restrictList[l].matvec(self.bList[l],self.bList[l-1])
            if initialGuessIsZero:
                self.uList[0][:]=0.0
        for l in range(currentMesh):
            if self.tolList is not None:
                self.switchToResidualConvergence(self.levelSolverList[l],
                                                 self.tolList[l])
            self.levelSolverList[l].solve(u=self.uList[l],b=self.bList[l],
                                          par_u=self.par_uList[l],
                                          par_b=self.par_bList[l],
                                          initialGuessIsZero=initialGuessIsZero)
            initialGuessIsZero=False
            if self.tolList is not None:
                self.revertToFixedIteration(self.levelSolverList[l])
            if l < currentMesh -1:
                self.prolongList[l+1].matvec(self.uList[l],self.uList[l+1])
            else:
                self.prolongList[l+1].matvec(self.uList[l],u)
        if self.tolList is not None:
            self.switchToResidualConvergence(self.levelSolverList[currentMesh],
                                             self.tolList[currentMesh])
        # LinearSolver.solve is (u,r,b,par_u,par_b,initialGuessIsZero): passing
        # initialGuessIsZero positionally put it in the par_u slot. The coarse
        # level solves above still carry no parallel vectors -- NI needs a
        # per-level ParVec for that, which is not built yet.
        self.levelSolverList[currentMesh].solve(u=u,r=r,b=b,par_u=par_u,par_b=par_b,
                                                initialGuessIsZero=initialGuessIsZero)
        self.infoString += "**************Start Level %i Info********************\n" % currentMesh
        self.infoString+=self.levelSolverList[currentMesh].info()
        self.infoString += "**************End Level %i Info********************\n" % currentMesh
        if self.tolList is not None:
            self.revertToFixedIteration(self.levelSolverList[currentMesh])
    def solveMultilevel(self,bList,uList,par_bList=None,par_uList=None,initialGuessIsZero=False):
        self.infoString="*************Start Multilevel Linear Solver Info*******************\n"
        for l in range(self.fineMesh):
            if self.tolList is not None:
                self.switchToResidualConvergence(self.levelSolverList[l],self.tolList[l])
            self.levelSolverList[l].solve(u=uList[l],b=bList[l],
                                          par_u=None if par_uList is None else par_uList[l],
                                          par_b=None if par_bList is None else par_bList[l],
                                          initialGuessIsZero=initialGuessIsZero)
            initialGuessIsZero=False
            if self.tolList is not None:
                self.revertToFixedIteration(self.levelSolverList[l])
            self.prolongList[l+1].matvec(uList[l],uList[l+1])
            self.infoString += "**************Start Level %i Info********************\n" % l
            self.infoString+=self.levelSolverList[l].info()
            self.infoString += "**************End Level %i Info********************\n" % l
        if self.tolList is not None:
            self.switchToResidualConvergence(self.levelSolverList[self.fineMesh],self.tolList[self.fineMesh])
        self.levelSolverList[self.fineMesh].solve(
            u=uList[self.fineMesh],b=bList[self.fineMesh],
            par_u=None if par_uList is None else par_uList[self.fineMesh],
            par_b=None if par_bList is None else par_bList[self.fineMesh],
            initialGuessIsZero=initialGuessIsZero)
        self.infoString += "**************Start Level %i Info********************\n" % l
        self.infoString+=self.levelSolverList[self.fineMesh].info()
        self.infoString += "**************End Level %i Info********************\n" % l
        if self.tolList is not None:
            self.revertToFixedIteration(self.levelSolverList[self.fineMesh])
        self.infoString+="********************End Multilevel Linear Solver Info*********************\n"
    def info(self):
        return self.infoString
    def switchToResidualConvergence(self,solver,rtol):
        self.saved_ctest = solver.convergenceTest
        self.saved_rtol_r = solver.rtol_r
        self.saved_atol_r = solver.atol_r
        self.saved_maxIts = solver.maxIts
        self.saved_printInfo = solver.printInfo
        solver.convergenceTest = 'r'
        solver.rtol_r = rtol
        solver.atol_r = self.atol_r
        solver.maxIts = self.maxIts
        solver.printInfo = self.printInfo
    def revertToFixedIteration(self,solver):
        solver.convergenceTest = self.saved_ctest
        solver.rtol_r = self.saved_rtol_r
        solver.atol_r = self.saved_atol_r
        solver.maxIts = self.saved_maxIts
        solver.printInfo = self.saved_printInfo

    def info(self):
        return self.infoString
    def failed(self):
        """NI sets solverList to [self]*nLevels, so the inherited failed()
        would recurse; ask the finest level solver instead."""
        return self.levelSolverList[-1].failed()
def petscMGHierarchy(par_linearOperatorList,prolongList,level,coarseSolver='direct'):
    """The MGHierarchy for the KSP on ``level``: levels 0..level-1 and the
    transfers between 0..level.

    The transfers have to be PETSc Mats, which is what the distributed nodal
    operators carry. A global proteus SparseMatrix is the wrong shape for a
    rank's vectors, so a hierarchy built over one would be wrong without
    complaint; refuse it here instead.
    """
    interpolations = [None]
    for k in range(1,level+1):
        mat = getattr(prolongList[k],'mat',None)
        if mat is None:
            raise ValueError(
                "petsc_MG needs the transfer operators as PETSc matrices, but "
                "level %i's is a %s. Only the distributed nodal operators "
                "qualify (see Transport.useParallelTransfers)."
                % (k,type(prolongList[k]).__name__))
        interpolations.append(mat)
    return MGHierarchy(list(par_linearOperatorList[:level]),interpolations,coarseSolver)


def starILURelaxation():
    """Default relaxation for StarILU: undamped in serial, damped in parallel.

    In serial the sweep over stars is multiplicative -- each star sees the
    updates of the ones before it -- and converges undamped. In parallel it is
    multiplicative only within a rank; between ranks every owned node is
    updated from the pre-sweep values on the other side, which makes it
    additive there. Undamped, that additive coupling is not convergent for
    every partition: on the 81-node Poisson mesh StarILU as a plain linear
    solver diverges on 3 ranks (checksum 2e15) while converging on 1, 2 and 4,
    and weights of 0.7 and below converge on all of them. 0.5 leaves margin,
    since which partitions fail is not something to tune to.

    Overridable through the relaxationFactor argument of
    multilevelLinearSolverChooser.
    """
    from . import Comm
    return 1.0 if Comm.get().size() == 1 else 0.5


def mgmCoarseSolver(L,par_L,prefix='mgm_coarse_',kind='direct'):
    """The solver for the coarsest level of a multigrid cycle.

    In serial that is a sparse LU factorization of the level's matrix. In
    parallel it must be an exact solve of the *global* coarse system, and a
    rank's subdomain matrix is not that: factoring it (what LU does) solves
    each rank's block on its own and drops the coupling between them. The cycle
    then still converges, but to a rank-count-dependent iterate -- a
    block-Jacobi coarse correction in place of the exact one -- and loses the
    property that makes multigrid's rate independent of the mesh.

    The coarse level is small by construction, so the default parallel solve is
    PETSc's redundant preconditioner: gather the coarse matrix, factor it in
    full, and apply it with no iteration (KSP preonly). Every rank holds a copy,
    which is cheap exactly because the level is coarse, and needs no external
    direct-solver package. It is replicated, though, and stops being cheap when the
    coarse level grows with the rank count; ``kind`` ('redundant', 'direct', 'amg';
    see coarseSolverOptions) chooses a distributed alternative.

    The solver reads PETSc options under ``prefix``, so a run can replace any of it --
    ``-mgm_coarse_pc_type gamg``, say. The defaults only fill options the user has
    not set.
    """
    from . import Comm
    if par_L is None or Comm.get().size() == 1:
        return LU(L=L)
    if not HAVE_PETSC4PY:
        return RedundantLU(L,par_L)
    applyCoarseSolverDefaults(p4pyPETSc.Options(prefix),'',kind)
    #maxIts is left generous on purpose: preonly reports one iteration, and a
    #solver whose iteration count equals maxIts is reported as having failed,
    #which Newton acts on when this solver is the level-0 solver of a nested
    #iteration.
    return KSP_petsc4py(L,par_L,
                        maxIts=100,
                        convergenceTest='r',
                        prefix=prefix)


"""
A function for setting up a multilevel linear solver.
"""
def multilevelLinearSolverChooser(linearOperatorList,
                                  par_linearOperatorList,
                                  multilevelLinearSolverType=NI,
                                  relativeToleranceList=None,
                                  absoluteTolerance=1.0e-8,
                                  solverConvergenceTest='r',
                                  solverMaxIts=500,
                                  printSolverInfo=False,
                                  computeSolverRates=False,
                                  levelLinearSolverType=MGM,
                                  printLevelSolverInfo=False,
                                  computeLevelSolverRates=False,
                                  smootherType=Jacobi,
                                  prolongList=None,
                                  restrictList=None,
                                  connectivityListList=None,
                                  cycles=3,
                                  preSmooths=3,
                                  postSmooths=3,
                                  printSmootherInfo=False,
                                  computeSmootherRates=False,
                                  smootherConvergenceTest='its',
                                  relaxationFactor=None,
                                  computeEigenvalues=False,
                                  parallelUsesFullOverlap = True,
                                  par_duList=None,
                                  solver_options_prefix=None,
                                  linearSolverLocalBlockSize=1,
                                  linearSmootherOptions=(),
                                  coarseSolver='direct'):
    logEvent("multilevelLinearSolverChooser type= %s" % multilevelLinearSolverType)
    if coarseSolver not in COARSE_SOLVERS:
        raise ValueError("coarseSolver must be one of %s, not %r" % (COARSE_SOLVERS,coarseSolver))
    if (multilevelLinearSolverType == KSP_petsc4py or
        multilevelLinearSolverType == LU or
        multilevelLinearSolverType == Jacobi or
        multilevelLinearSolverType == GaussSeidel or
        multilevelLinearSolverType == StarILU or
        multilevelLinearSolverType == StarBILU or
        multilevelLinearSolverType == MGM):
        levelLinearSolverType = multilevelLinearSolverType
        printLevelSolverInfo = printSolverInfo
        computeLevelSolverRates = computeSolverRates
    nLevels = len(linearOperatorList)
    multilevelLinearSolver = None
    levelLinearSolverList = []
    levelLinearSolver = None
    if levelLinearSolverType == MGM:
        preSmootherList=[]
        postSmootherList=[]
        mgItsList=[]
        for l in range(nLevels):
            mgItsList.append(cycles)
            if l > 0:
                if smootherType == Jacobi:
                    if relaxationFactor is None:
                        relaxationFactor = 4.0/5.0
                    preSmootherList.append(Jacobi(L=linearOperatorList[l],
                                                  weight=relaxationFactor,
                                                  maxIts=preSmooths,
                                                  convergenceTest = smootherConvergenceTest,
                                                  computeRates = computeSmootherRates,
                                                  printInfo = printSmootherInfo))
                    postSmootherList.append(Jacobi(L=linearOperatorList[l],
                                                   weight=relaxationFactor,
                                                   maxIts=postSmooths,
                                                   convergenceTest = smootherConvergenceTest,
                                                   computeRates = computeSmootherRates,
                                                   printInfo = printSmootherInfo))
                elif smootherType == GaussSeidel:
                    if relaxationFactor is None:
                        relaxationFactor = 0.33
                    preSmootherList.append(GaussSeidel(connectionList = connectivityListList[l],
                                                       L=linearOperatorList[l],
                                                       weight=relaxationFactor,
                                                       maxIts =  preSmooths,
                                                       convergenceTest = smootherConvergenceTest,
                                                       computeRates = computeSmootherRates,
                                                       printInfo = printSmootherInfo))
                    postSmootherList.append(GaussSeidel(connectionList = connectivityListList[l],
                                                        L=linearOperatorList[l],
                                                        weight=relaxationFactor,
                                                        maxIts =  postSmooths,
                                                        convergenceTest = smootherConvergenceTest,
                                                        computeRates = computeSmootherRates,
                                                        printInfo = printSmootherInfo))
                elif smootherType == StarILU:
                    if relaxationFactor is None:
                        relaxationFactor = starILURelaxation()
                    preSmootherList.append(StarILU(connectionList = connectivityListList[l],
                                                   L=linearOperatorList[l],
                                                   weight=relaxationFactor,
                                                   maxIts =  preSmooths,
                                                   convergenceTest = smootherConvergenceTest,
                                                   computeRates = computeSmootherRates,
                                                   printInfo = printSmootherInfo))
                    postSmootherList.append(StarILU(connectionList = connectivityListList[l],
                                                    L=linearOperatorList[l],
                                                    weight=relaxationFactor,
                                                    maxIts =  postSmooths,
                                                    convergenceTest = smootherConvergenceTest,
                                                    computeRates = computeSmootherRates,
                                                    printInfo = printSmootherInfo))
                elif smootherType == StarBILU:
                    if relaxationFactor is None:
                        relaxationFactor = 1.0
                    preSmootherList.append(StarBILU(connectionList = connectivityListList[l],
                                                    L=linearOperatorList[l],
                                                    bs = linearSolverLocalBlockSize,
                                                    weight=relaxationFactor,
                                                    maxIts =  preSmooths,
                                                    convergenceTest = smootherConvergenceTest,
                                                    computeRates = computeSmootherRates,
                                                    printInfo = printSmootherInfo))
                    postSmootherList.append(StarBILU(connectionList = connectivityListList[l],
                                                     L=linearOperatorList[l],
                                                     bs = linearSolverLocalBlockSize,
                                                     weight=relaxationFactor,
                                                     maxIts =  postSmooths,
                                                     convergenceTest = smootherConvergenceTest,
                                                     computeRates = computeSmootherRates,
                                                     printInfo = printSmootherInfo))
                else:
                    # Leaving this as a log message left preSmootherList short,
                    # so MGM.__init__ failed later with a bare IndexError that
                    # named neither the option nor the cause.
                    raise ValueError(
                        "MGM requires a smoother, but linearSmoother=%r is not one "
                        "of Jacobi, GaussSeidel, StarILU, StarBILU."
                        % (smootherType,))
            else:
                preSmootherList.append([])
                postSmootherList.append([])
                coarseSolver_ = mgmCoarseSolver(linearOperatorList[l],
                                                par_linearOperatorList[l]
                                                if par_linearOperatorList is not None
                                                else None,
                                                kind=coarseSolver)
        levelLinearSolver = MGM(prolongList = prolongList,
                                restrictList = restrictList,
                                LList = linearOperatorList,
                                preSmootherList = preSmootherList,
                                postSmootherList = postSmootherList,
                                coarseSolver = coarseSolver_,
                                mgItsList = mgItsList,
                                printInfo = printLevelSolverInfo,
                                computeRates = computeLevelSolverRates,
                                par_duList = par_duList)
        levelLinearSolverList = levelLinearSolver.solverList
    elif levelLinearSolverType == LU:
        for l in range(nLevels):
            levelLinearSolverList.append(LU(linearOperatorList[l],computeEigenvalues))
        levelLinearSolver = levelLinearSolverList
    elif levelLinearSolverType == KSP_petsc4py:
        for l in range(nLevels):
            preconditionerOptions = linearSmootherOptions
            if smootherType == petsc_MG:
                preconditionerOptions = petscMGHierarchy(par_linearOperatorList,prolongList,l,coarseSolver)
            levelLinearSolverList.append(KSP_petsc4py(linearOperatorList[l],par_linearOperatorList[l],
                                                      maxIts = solverMaxIts,
                                                      convergenceTest = solverConvergenceTest,
                                                      rtol_r = relativeToleranceList[l],
                                                      atol_r = absoluteTolerance,
                                                      computeRates = computeLevelSolverRates,
                                                      printInfo = printLevelSolverInfo,
                                                      prefix=solver_options_prefix,
                                                      Preconditioner=smootherType,
                                                      connectionList = connectivityListList[l],
                                                      linearSolverLocalBlockSize = linearSolverLocalBlockSize,
                                                      preconditionerOptions = preconditionerOptions))
        levelLinearSolver = levelLinearSolverList
    elif levelLinearSolverType == Jacobi:
        if relaxationFactor is None:
            relaxationFactor = 4.0/5.0
        for l in range(nLevels):
            levelLinearSolverList.append(Jacobi(L=linearOperatorList[l],
                                                weight=relaxationFactor,
                                                maxIts = solverMaxIts,
                                                convergenceTest = solverConvergenceTest,
                                                rtol_r = relativeToleranceList[l],
                                                atol_r = absoluteTolerance,
                                                computeRates = computeLevelSolverRates,
                                                printInfo = printLevelSolverInfo))
        levelLinearSolver = levelLinearSolverList
    elif levelLinearSolverType == GaussSeidel:
        if relaxationFactor is None:
            relaxationFactor=0.33
        for l in range(nLevels):
            levelLinearSolverList.append(GaussSeidel(connectionList = connectivityListList[l],
                                                     L=linearOperatorList[l],
                                                     weight = relaxationFactor,
                                                     maxIts = solverMaxIts,
                                                     convergenceTest = solverConvergenceTest,
                                                     rtol_r = relativeToleranceList[l],
                                                     atol_r = absoluteTolerance,
                                                     computeRates = computeLevelSolverRates,
                                                     printInfo = printLevelSolverInfo))
        levelLinearSolver = levelLinearSolverList
    elif levelLinearSolverType == StarILU:
        if relaxationFactor is None:
            relaxationFactor=starILURelaxation()
        for l in range(nLevels):
            levelLinearSolverList.append(StarILU(connectionList = connectivityListList[l],
                                                 L=linearOperatorList[l],
                                                 weight=relaxationFactor,
                                                 maxIts = solverMaxIts,
                                                 convergenceTest = solverConvergenceTest,
                                                 rtol_r = relativeToleranceList[l],
                                                 atol_r = absoluteTolerance,
                                                 computeRates = computeLevelSolverRates,
                                                 printInfo = printLevelSolverInfo))
        levelLinearSolver = levelLinearSolverList
    elif levelLinearSolverType == StarBILU:
        if relaxationFactor is None:
            relaxationFactor=1.0
        for l in range(nLevels):
            levelLinearSolverList.append(StarBILU(connectionList = connectivityListList[l],
                                                  L=linearOperatorList[l],
                                                  bs= linearSolverLocalBlockSize,
                                                  weight=relaxationFactor,
                                                  maxIts = solverMaxIts,
                                                  convergenceTest = solverConvergenceTest,
                                                  rtol_r = relativeToleranceList[l],
                                                  atol_r = absoluteTolerance,
                                                  computeRates = computeLevelSolverRates,
                                                  printInfo = printLevelSolverInfo))
        levelLinearSolver = levelLinearSolverList
    else:
        raise RuntimeError("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Unknown level linear solver %r" % (levelLinearSolverType,))
    if multilevelLinearSolverType == NI:
        multilevelLinearSolver = NI(solverList = levelLinearSolverList,
                                    prolongList = prolongList,
                                    restrictList = restrictList,
                                    maxIts  = solverMaxIts,
                                    tolList = relativeToleranceList,
                                    atol    = absoluteTolerance,
                                    printInfo= printSolverInfo,
                                    computeRates = computeSolverRates,
                                    par_duList = par_duList)
    elif (multilevelLinearSolverType == KSP_petsc4py or
          multilevelLinearSolverType == LU or
          multilevelLinearSolverType == Jacobi or
          multilevelLinearSolverType == GaussSeidel or
          multilevelLinearSolverType == StarILU or
          multilevelLinearSolverType == StarBILU or
          multilevelLinearSolverType == MGM):
        multilevelLinearSolver = MultilevelLinearSolver(levelLinearSolverList,
                                                        computeRates = computeSolverRates,
                                                        printInfo=printSolverInfo)
    else:
        raise RuntimeError("Unknown linear solver %s" % multilevelLinearSolverType)
    if (levelLinearSolverType == LU):
        directSolverFlag=True
    else:
        directSolverFlag=False
    for levelSolver in multilevelLinearSolver.solverList:
        levelSolver.par_fullOverlap = parallelUsesFullOverlap
    return (multilevelLinearSolver,directSolverFlag)

## @}

if __name__ == '__main__':
    from LinearAlgebra import *
    from . import LinearSolvers
    from .LinearSolvers import *
    import Gnuplot
    from Gnuplot import *
    from math import *
    from RandomArray import *
    gf = Gnuplot.Gnuplot()
    gf("set terminal x11")
    ginit = Gnuplot.Gnuplot()
    ginit("set terminal x11")
    gsol = Gnuplot.Gnuplot()
    gsol("set terminal x11")
    gsolNI = Gnuplot.Gnuplot()
    gsolNI("set terminal x11")
    gres = Gnuplot.Gnuplot()
    gres("set terminal x11")
    levels = 7
    n=2**levels + 1
    h = 1.0/(n-1.0)
    freq=10
    uFine = uniform(0,1,(n))
    uFine[0]=0.0
    uFine[n-1]=0.0
    xFine = numpy.arange(0,1.0+h,h,dtype='d')
    bFine = (freq*2*pi)**2*numpy.sin(freq*2*pi*xFine)
    gf.plot(Gnuplot.Data(xFine,bFine))
    ginit.plot(Gnuplot.Data(xFine,uFine))
    uList=[]
    bList=[]
    prolongList=[]
    restrictList=[]
    LList=[]
    LDList=[]
    hList=[]
    meshList=[]
    preSmootherList=[]
    postSmootherList=[]
    mgItsList=[]
    for l in range(levels):
        N = 2**(l+1) + 1
        L = SparseMat_old(N-2,N-2,3*(N-2),sym=True)
        LD = Mat(N-2,N-2)
        H = 1.0/(N-1.0)
        hList.append(H)
        mgItsList.append(6)
        meshList.append(numpy.arange(0,1.0+H,H,dtype='d')[1:N-1])
        u = uniform(0,1,(N))
        u[0]  = 0.0
        u[N-1] = 0.0
        b = (freq*2*pi)**2*numpy.sin(freq*2*pi*meshList[l])
        uList.append(u[1:N-1])
        bList.append(b)
        beginAssembly(L)
        for i in range(N-2):
            L[i,i] = 2.0/H**2
            LD[i,i] = 2.0/H**2
            if i > 0:
                L[i,i-1] = -1.0/H**2
                LD[i,i-1] = -1.0/H**2
            if i < N-3:
                L[i,i+1] = -1.0/H**2
                LD[i,i+1] = -1.0/H**2
            endAssembly(L)
        LList.append(L)
        LDList.append(LD)
        if l > 0:
            cN = (N - 1)/2 + 1
            restrict = SparseMat_old(cN-2,N-2,3*(N-2))
            prolong = SparseMat_old(N-2,cN-2,3*(N-2))
            for i in range(cN-2):
                restrict[i,2*i]   = 1.0/4.0
                restrict[i,2*i+1] = 2.0/4.0
                restrict[i,2*i+2] = 1.0/4.0
                prolong[2*i,i] = 1.0/2.0
                prolong[2*i+1,i]= 2.0/2.0
                prolong[2*i+2,i]= 1.0/2.0
            restrict.to_csr()
            restrictList.append(restrict)
            prolong.to_csr()
            prolongList.append(prolong)
            N = cN
            preSmootherList.append(Jacobi(L, 2.0/3.0,3))
            postSmootherList.append(Jacobi(L, 2.0/3.0,3))
        else:
            restrictList.append([])
            prolongList.append([])
            preSmootherList.append([])
            postSmootherList.append([])
            coarseSolver = Jacobi(L,1.0,1)
    mgm = MGM(prolongList,restrictList,LList,preSmootherList,postSmootherList,coarseSolver,mgItsList)
    mgm.prepare()
    rnorm=1.0
    mgits = 0
    while rnorm > 1.0e-8 and mgits < 20:
        mgits +=1
        mgm.solve(u=uFine[1:n-1],b=bFine[1:n-1])
        rnorm = wl2Norm(mgm.residual(),h)
    gsol.plot(Gnuplot.Data(xFine,uFine,title='numerical solution-MGM'),
              Gnuplot.Data(xFine,numpy.sin(freq*2*pi*xFine),title='exact solution'))
    #gres.plot(Gnuplot.Data(x[1:n-1],mgm.smootherList[0].res,title='final residual'))
    ni = NI(mgm.solverList,prolongList,restrictList)
    ni.prepare()
    ni.solveMultilevel(bList,uList)
    rnorm = wl2Norm(ni.residual(),h)
    gsolNI.plot(Gnuplot.Data(meshList[-1],uList[-1],
                             title='numerical solution-NI'),
                Gnuplot.Data(meshList[-1],numpy.sin(freq*2*pi*meshList[-1]),
                             title='exact solution'))
    evals=[]
    for a,b,u,h in zip(LDList,bList,uList,hList):
        lu = LU(a,computeRes=True)
        lu.prepare(b)
        lu.solve(u,b)
        dev = DenseEigenvalues(a)
        dev.computeEigenvalues()
        evals.append(dev.eigenvalues)
        ratio = (max(abs(dev.eigenvalues))/min(abs(dev.eigenvalues)))*(h**2)
        print("k*h**2 %12.5E" % ratio)
    gevals = Gnuplot.Gnuplot()
    gevals("set terminal x11")
    gevals.plot(Gnuplot.Data(evals[0],title='eigenvalues'))
    for ev in evals[1:]:
        gevals.replot(Gnuplot.Data(ev,title='eigenvalues'))
    input('Please press return to continue... \n')

class StorageSet(set):
    def __init__(self,initializer=[],shape=(0,),storageType='d'):
        set.__init__(self,initializer)
        self.shape = shape
        self.storageType = storageType
    def allocate(self,storageDict):
        for k in self:
            storageDict[k] = numpy.zeros(self.shape,self.storageType)

class OperatorConstructor(object):
    """ Base class for operator constructors. """
    def __init__(self,model):
        self.model = model

    def attachMassOperator(self):
        """Create the discrete Mass operator. """
        self._mass_val = self.model.nzval.copy()
        self._mass_val.fill(0.)
        self.MassOperator = SparseMat(self.model.nFreeVDOF_global,
                                      self.model.nFreeVDOF_global,
                                      self.model.nnz,
                                      self._mass_val,
                                      self.model.colind,
                                      self.model.rowptr)

    def attachInvScaledMassOperator(self):
        """ Create discrete InvScaled Mass operator. """
        self._inv_scaled_mass_val = self.model.nzval.copy()
        self._inv_scaled_mass_val.fill(0.)
        self.TPInvScaledMassOperator = SparseMat(self.model.nFreeVDOF_global,
                                                 self.model.nFreeVDOF_global,
                                                 self.model.nnz,
                                                 self._inv_scaled_mass_val,
                                                 self.model.colind,
                                                 self.model.rowptr)

    def attachScaledMassOperator(self):
        """ Create discrete Scaled Mass operator. """
        self._scaled_mass_val = self.model.nzval.copy()
        self._scaled_mass_val.fill(0.)
        self.TPScaledMassOperator = SparseMat(self.model.nFreeVDOF_global,
                                              self.model.nFreeVDOF_global,
                                              self.model.nnz,
                                              self._scaled_mass_val,
                                              self.model.colind,
                                              self.model.rowptr)

    def attachLaplaceOperator(self):
        """ Create discrete Laplace matrix operator. """
        self._laplace_val = self.model.nzval.copy()
        self._laplace_val.fill(0.)
        self.TPInvScaledLaplaceOperator = SparseMat(self.model.nFreeVDOF_global,
                                                    self.model.nFreeVDOF_global,
                                                    self.model.nnz,
                                                    self._laplace_val,
                                                    self.model.colind,
                                                    self.model.rowptr)

    def attachTPAdvectionOperator(self):
        """ Create discrete Advection matrix operator. """
        self._advection_val = self.model.nzval.copy()
        self._advection_val.fill(0.)
        self.TPScaledAdvectionOperator = SparseMat(self.model.nFreeVDOF_global,
                                                   self.model.nFreeVDOF_global,
                                                   self.model.nnz,
                                                   self._advection_val,
                                                   self.model.colind,
                                                   self.model.rowptr)

class OperatorConstructor_rans2p(OperatorConstructor):
    """ A class for building common discrete rans2p operators.

    Arguments:
    ----------
    LevelModel : :class:`proteus.mprans.RANS2P.LevelModel`
        Level transport model derived from the rans2p class.
    """
    def __init__(self,levelModel):
        OperatorConstructor.__init__(self,levelModel)

    def updateTPAdvectionOperator(self,
                                  density_scaling):
        """
        Update the discrete two-phase advection operator matrix.

        Parameters
        ----------
        density_scaling : bool
            Indicates whether advection terms should be scaled with
            the density (True) or 1 (False)
        """
        rho_0 = density_scaling*self.model.coefficients.rho_0 + (1-density_scaling)*1
        nu_0 = density_scaling*self.model.coefficients.nu_0 + (1-density_scaling)*1
        rho_1 = density_scaling*self.model.coefficients.rho_1 + (1-density_scaling)*1
        nu_1 = density_scaling*self.model.coefficients.nu_1 + (1-density_scaling)*1        

        self.TPScaledAdvectionOperator.getCSRrepresentation()[2].fill(0.)

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.model.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.model.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.model.mesh.nodeArray
        argsDict["mesh_l2g"] = self.model.mesh.elementNodesArray
        argsDict["dV_ref"] = self.model.elementQuadratureWeights[('u',0)]
        argsDict["p_trial_ref"] = self.model.u[0].femSpace.psi
        argsDict["p_grad_trial_ref"] = self.model.u[0].femSpace.grad_psi
        argsDict["vel_trial_ref"] = self.model.u[1].femSpace.psi
        argsDict["vel_grad_trial_ref"] = self.model.u[1].femSpace.grad_psi
        argsDict["elementDiameter"] = self.model.elementDiameter
        argsDict["nodeDiametersArray"] = self.model.mesh.nodeDiametersArray
        argsDict["nElements_global"] = self.model.mesh.nElements_global
        argsDict["useMetrics"] = self.model.coefficients.useMetrics
        argsDict["epsFact_rho"] = self.model.coefficients.epsFact_density
        argsDict["epsFact_mu"] = self.model.coefficients.epsFact
        argsDict["rho_0"] = rho_0
        argsDict["nu_0"] = nu_0
        argsDict["rho_1"] = rho_1
        argsDict["nu_1"] = nu_1
        argsDict["vel_l2g"] = self.model.u[1].femSpace.dofMap.l2g
        argsDict["u_dof"] = self.model.u[1].dof
        argsDict["v_dof"] = self.model.u[2].dof
        argsDict["w_dof"] = self.model.u[3].dof
        argsDict["useVF"] = self.model.coefficients.useVF
        argsDict["&vf"] = self.model.coefficients.q_vf
        argsDict["&phi"] = self.model.coefficients.q_phi
        argsDict["csrRowIndeces_p_p"] = self.model.csrRowIndeces[(0,0)]
        argsDict["csrColumnOffsets_p_p"] = self.model.csrColumnOffsets[(0,0)]
        argsDict["csrRowIndeces_u_u"] = self.model.csrRowIndeces[(1,1)]
        argsDict["csrColumnOffsets_u_u"] = self.model.csrColumnOffsets[(1,1)]
        argsDict["csrRowIndeces_v_v"] = self.model.csrRowIndeces[(2,2)]
        argsDict["csrColumnOffsets_v_v"] = self.model.csrColumnOffsets[(2,2)]
        argsDict["csrRowIndeces_w_w"] = self.model.csrRowIndeces[(3,3)]
        argsDict["csrColumnOffsets_w_w"] = self.model.csrColumnOffsets[(3,3)]
        argsDict["advection_matrix"] = self.TPScaledAdvectionOperator.getCSRrepresentation()[2]
        self.model.rans2p.getTwoPhaseAdvectionOperator(argsDict)

    def updateTPInvScaledLaplaceOperator(self):
        """ Create a discrete two phase laplace operator matrix. """
        self.TPInvScaledLaplaceOperator.getCSRrepresentation()[2].fill(0.)

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["mesh_trial_ref"] = self.model.u[0].femSpace.elementMaps.psi
        argsDict["mesh_grad_trial_ref"] = self.model.u[0].femSpace.elementMaps.grad_psi
        argsDict["mesh_dof"] = self.model.mesh.nodeArray
        argsDict["mesh_l2g"] = self.model.mesh.elementNodesArray
        argsDict["dV_ref"] = self.model.elementQuadratureWeights[('u',0)]
        argsDict["p_grad_trial_ref"] = self.model.u[0].femSpace.grad_psi
        argsDict["vel_grad_trial_ref"] = self.model.u[1].femSpace.grad_psi
        argsDict["elementDiameter"] = self.model.elementDiameter
        argsDict["nodeDiametersArray"] = self.model.mesh.nodeDiametersArray
        argsDict["nElements_global"] = self.model.mesh.nElements_global
        argsDict["useMetrics"] = self.model.coefficients.useMetrics
        argsDict["epsFact_rho"] = self.model.coefficients.epsFact_density
        argsDict["epsFact_mu"] = self.model.coefficients.epsFact
        argsDict["rho_0"] = self.model.coefficients.rho_0
        argsDict["nu_0"] = self.model.coefficients.nu_0
        argsDict["rho_1"] = self.model.coefficients.rho_1
        argsDict["nu_1"] = self.model.coefficients.nu_1
        argsDict["p_l2g"] = self.model.u[0].femSpace.dofMap.l2g
        argsDict["vel_l2g"] = self.model.u[1].femSpace.dofMap.l2g
        argsDict["p_dof"] = self.model.u[0].dof
        argsDict["u_dof"] = self.model.u[1].dof
        argsDict["v_dof"] = self.model.u[2].dof
        argsDict["w_dof"] = self.model.u[3].dof
        argsDict["useVF"] = self.model.coefficients.useVF
        argsDict["vf"] = self.model.coefficients.q_vf
        argsDict["phi"] = self.model.coefficients.q_phi
        argsDict["sdInfo_p_p_rowptr"] = self.model.coefficients.sdInfo[(1,1)][0]
        argsDict["sdInfo_p_p_colind"] = self.model.coefficients.sdInfo[(1,1)][1]
        argsDict["sdInfo_u_u_rowptr"] = self.model.coefficients.sdInfo[(1,1)][0]
        argsDict["sdInfo_u_u_colind"] = self.model.coefficients.sdInfo[(1,1)][1]
        argsDict["sdInfo_v_v_rowptr"] = self.model.coefficients.sdInfo[(2,2)][0]
        argsDict["sdInfo_v_v_colind"] = self.model.coefficients.sdInfo[(2,2)][1]
        argsDict["sdInfo_w_w_rowptr"] = self.model.coefficients.sdInfo[(3,3)][0]
        argsDict["sdInfo_w_w_colind"] = self.model.coefficients.sdInfo[(3,3)][1]
        argsDict["csrRowIndeces_p_p"] = self.model.csrRowIndeces[(0,0)]
        argsDict["csrColumnOffsets_p_p"] = self.model.csrColumnOffsets[(0,0)]
        argsDict["csrRowIndeces_u_u"] = self.model.csrRowIndeces[(1,1)]
        argsDict["csrColumnOffsets_u_u"] = self.model.csrColumnOffsets[(1,1)]
        argsDict["csrRowIndeces_v_v"] = self.model.csrRowIndeces[(2,2)]
        argsDict["csrColumnOffsets_v_v"] = self.model.csrColumnOffsets[(2,2)]
        argsDict["csrRowIndeces_w_w"] = self.model.csrRowIndeces[(3,3)]
        argsDict["csrColumnOffsets_w_w"] = self.model.csrColumnOffsets[(3,3)]
        argsDict["laplace_matrix"] = self.TPInvScaledLaplaceOperator.getCSRrepresentation()[2]
        self.model.rans2p.getTwoPhaseInvScaledLaplaceOperator(argsDict)

    def updateTwoPhaseMassOperator_rho(self,
                                       density_scaling = True,
                                       lumped = True):
        """
        Create a discrete TwoPhase Mass operator matrix.

        Parameters
        ----------
        density_scaling : bool
            Indicates whether advection terms should be scaled with
            the density (True) or 1 (False)
        lumped : bool
            Indicates whether the mass matrices should be lumped or
            full.

        """
        rho_0 = density_scaling*self.model.coefficients.rho_0 + (1-density_scaling)*1
        nu_0 = density_scaling*self.model.coefficients.nu_0 + (1-density_scaling)*1
        rho_1 = density_scaling*self.model.coefficients.rho_1 + (1-density_scaling)*1
        nu_1 = density_scaling*self.model.coefficients.nu_1 + (1-density_scaling)*1

        self.TPScaledMassOperator.getCSRrepresentation()[2].fill(0.)

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["scale_type"] = 1
        argsDict["use_numerical_viscosity"] = 0
        argsDict["lumped"] = lumped
        argsDict["&mesh_trial_ref"] = self.model.u[0].femSpace.elementMaps.psi
        argsDict["&mesh_grad_trial_ref"] = self.model.u[0].femSpace.elementMaps.grad_psi
        argsDict["&mesh_dof"] = self.model.mesh.nodeArray
        argsDict["mesh_l2g"] = self.model.mesh.elementNodesArray
        argsDict["dV_ref"] = self.model.elementQuadratureWeights[('u',0)]
        argsDict["p_trial_ref"] = self.model.u[0].femSpace.psi
        argsDict["p_test_ref"] = self.model.u[0].femSpace.psi
        argsDict["vel_trial_ref"] = self.model.u[1].femSpace.psi
        argsDict["vel_test_ref"] = self.model.u[1].femSpace.psi
        argsDict["elementDiameter"] = self.model.elementDiameter
        argsDict["nodeDiametersArray"] = self.model.mesh.nodeDiametersArray
        argsDict["numerical_viscosity"] = self.model.coefficients.numerical_viscosity
        argsDict["nElements_global"] = self.model.mesh.nElements_global
        argsDict["useMetrics"] = self.model.coefficients.useMetrics
        argsDict["epsFact_rho"] = self.model.coefficients.epsFact_density
        argsDict["epsFact_mu"] = self.model.coefficients.epsFact
        argsDict["rho_0"] = rho_0
        argsDict["nu_0"] = nu_0
        argsDict["rho_1"] = rho_1
        argsDict["nu_1"] = nu_1
        argsDict["p_l2g"] = self.model.u[0].femSpace.dofMap.l2g
        argsDict["vel_l2g"] = self.model.u[1].femSpace.dofMap.l2g
        argsDict["p_dof"] = self.model.u[0].dof
        argsDict["u_dof"] = self.model.u[1].dof
        argsDict["v_dof"] = self.model.u[2].dof
        argsDict["w_dof"] = self.model.u[3].dof
        argsDict["useVF"] = self.model.coefficients.useVF
        argsDict["vf"] = self.model.coefficients.q_vf
        argsDict["phi"] = self.model.coefficients.q_phi
        argsDict["csrRowIndeces_p_p"] = self.model.csrRowIndeces[(0,0)]
        argsDict["csrColumnOffsets_p_p"] = self.model.csrColumnOffsets[(0,0)]
        argsDict["csrRowIndeces_u_u"] = self.model.csrRowIndeces[(1,1)]
        argsDict["csrColumnOffsets_u_u"] = self.model.csrColumnOffsets[(1,1)]
        argsDict["csrRowIndeces_v_v"] = self.model.csrRowIndeces[(2,2)]
        argsDict["csrColumnOffsets_v_v"] = self.model.csrColumnOffsets[(2,2)]
        argsDict["csrRowIndeces_w_w"] = self.model.csrRowIndeces[(3,3)]
        argsDict["csrColumnOffsets_w_w"] = self.model.csrColumnOffsets[(3,3)]
        argsDict["mass_matrix"] = self.TPScaledMassOperator.getCSRrepresentation()[2]
        self.model.rans2p.getTwoPhaseScaledMassOperator(argsDict)

    def updateTwoPhaseInvScaledMassOperator(self,
                                            numerical_viscosity = True,
                                            lumped = True):
        """Create a discrete TwoPhase Mass operator matrix.

        Parameters
        ----------
        numerical_viscosity : bool
            Indicates whether the numerical viscosity should be
            included with the Laplace operator.Yes (True) or
            No (False)
        """
        self.TPInvScaledMassOperator.getCSRrepresentation()[2].fill(0.)

        argsDict = cArgumentsDict.ArgumentsDict()
        argsDict["scale_type"] = 0
        argsDict["use_numerical_viscosity"] = numerical_viscosity
        argsDict["lumped"] = lumped
        argsDict["&mesh_trial_ref"] = self.model.u[0].femSpace.elementMaps.psi
        argsDict["&mesh_grad_trial_ref"] = self.model.u[0].femSpace.elementMaps.grad_psi
        argsDict["&mesh_dof"] = self.model.mesh.nodeArray
        argsDict["mesh_l2g"] = self.model.mesh.elementNodesArray
        argsDict["dV_ref"] = self.model.elementQuadratureWeights[('u',0)]
        argsDict["p_trial_ref"] = self.model.u[0].femSpace.psi
        argsDict["p_test_ref"] = self.model.u[0].femSpace.psi
        argsDict["vel_trial_ref"] = self.model.u[1].femSpace.psi
        argsDict["vel_test_ref"] = self.model.u[1].femSpace.psi
        argsDict["elementDiameter"] = self.model.elementDiameter
        argsDict["nodeDiametersArray"] = self.model.mesh.nodeDiametersArray
        argsDict["numerical_viscosity"] = self.model.coefficients.numerical_viscosity
        argsDict["nElements_global"] = self.model.mesh.nElements_global
        argsDict["useMetrics"] = self.model.coefficients.useMetrics
        argsDict["epsFact_rho"] = self.model.coefficients.epsFact_density
        argsDict["epsFact_mu"] = self.model.coefficients.epsFact
        argsDict["rho_0"] = self.model.coefficients.rho_0
        argsDict["nu_0"] = self.model.coefficients.nu_0
        argsDict["rho_1"] = self.model.coefficients.rho_1
        argsDict["nu_1"] = self.model.coefficients.nu_1
        argsDict["p_l2g"] = self.model.u[0].femSpace.dofMap.l2g
        argsDict["vel_l2g"] = self.model.u[1].femSpace.dofMap.l2g
        argsDict["p_dof"] = self.model.u[0].dof
        argsDict["u_dof"] = self.model.u[1].dof
        argsDict["v_dof"] = self.model.u[2].dof
        argsDict["w_dof"] = self.model.u[3].dof
        argsDict["useVF"] = self.model.coefficients.useVF
        argsDict["vf"] = self.model.coefficients.q_vf
        argsDict["phi"] = self.model.coefficients.q_phi
        argsDict["csrRowIndeces_p_p"] = self.model.csrRowIndeces[(0,0)]
        argsDict["csrColumnOffsets_p_p"] = self.model.csrColumnOffsets[(0,0)]
        argsDict["csrRowIndeces_u_u"] = self.model.csrRowIndeces[(1,1)]
        argsDict["csrColumnOffsets_u_u"] = self.model.csrColumnOffsets[(1,1)]
        argsDict["csrRowIndeces_v_v"] = self.model.csrRowIndeces[(2,2)]
        argsDict["csrColumnOffsets_v_v"] = self.model.csrColumnOffsets[(2,2)]
        argsDict["csrRowIndeces_w_w"] = self.model.csrRowIndeces[(3,3)]
        argsDict["csrColumnOffsets_w_w"] = self.model.csrColumnOffsets[(3,3)]
        argsDict["mass_matrix"] = self.TPInvScaledMassOperator.getCSRrepresentation()[2]
        self.model.rans2p.getTwoPhaseScaledMassOperator(argsDict)

class OperatorConstructor_oneLevel(OperatorConstructor):
    """
    A class for building common discrete operators from one-level
    transport models.

    Arguments
    ---------
    OLT : :class:`proteus.Transport.OneLevelTransport`
        One level transport class from which operator construction
        will be based.
    """
    def __init__(self,OLT):
        OperatorConstructor.__init__(self,OLT)
        self._initializeOperatorConstruction()

    def updateMassOperator(self, rho=1.0):
        self._mass_val.fill(0.)
        try:
            rho = self.model.coefficients.rho
        except:
            pass
        _nd = self.model.coefficients.nd
        self.MassOperatorCoeff = TransportCoefficients.DiscreteMassMatrix(rho=rho, nd=_nd)
        _t = 1.0

        Mass_q = {}
        self._allocateMassOperatorQStorageSpace(Mass_q)
        if _nd == 2:
            self.MassOperatorCoeff.evaluate(_t,Mass_q)
        self._calculateMassOperatorQ(Mass_q)

        Mass_Jacobian = {}
        self._allocateMatrixSpace(self.MassOperatorCoeff,
                                  Mass_Jacobian)

        for ci,cjDict in self.MassOperatorCoeff.mass.items():
            for cj in cjDict:
                cfemIntegrals.updateMassJacobian_weak(Mass_q[('dm',ci,cj)],
                                                      Mass_q[('vXw*dV_m',cj,ci)],
                                                      Mass_Jacobian[ci][cj])
        self._createOperator(self.MassOperatorCoeff,Mass_Jacobian,self.MassOperator)

    def updateTPAdvectionOperator(self,phase_function=None):
        self._u = numpy.copy(self.model.q[('u',1)])
        self._v = numpy.copy(self.model.q[('u',2)])
        self._advective_field = [self._u, self._v]
        self._advection_val = self.model.nzval.copy()
        self._advection_val.fill(0.)
        _nd = self.model.coefficients.nd

        _rho_0 = self.model.coefficients.rho_0
        _rho_1 = self.model.coefficients.rho_1
        _nu_0 = self.model.coefficients.nu_0
        _nu_1 = self.model.coefficients.nu_1

        if phase_function == None:
            self.AdvectionOperatorCoeff = TransportCoefficients.DiscreteTwoPhaseAdvectionOperator(u = self._advective_field,
                                                                                                  nd = _nd,
                                                                                                  rho_0 = _rho_0,
                                                                                                  nu_0 = _nu_0,
                                                                                                  rho_1 = _rho_1,
                                                                                                  nu_1 = _nu_1,
                                                                                                  LS_model = _phase_func)
        else:
            self.AdvectionOperatorCoeff = TransportCoefficients.DiscreteTwoPhaseAdvectionOperator(u = self._advective_field,
                                                                                                  nd = _nd,
                                                                                                  rho_0 = _rho_0,
                                                                                                  nu_0 = _nu_0,
                                                                                                  rho_1 = _rho_1,
                                                                                                  nu_1 = _nu_1,
                                                                                                  phase_function = phase_function)

        _t = 1.0

        Advection_q = {}
        self._allocateAdvectionOperatorQStorageSpace(Advection_q)
        self._calculateQuadratureValues(Advection_q)
        if _nd==2:
            self.AdvectionOperatorCoeff.evaluate(_t,Advection_q)
        self._calculateAdvectionOperatorQ(Advection_q)

        Advection_Jacobian = {}
        self._allocateMatrixSpace(self.AdvectionOperatorCoeff,
                                  Advection_Jacobian)

        for ci,ckDict in self.AdvectionOperatorCoeff.advection.items():
            for ck in ckDict:
                cfemIntegrals.updateAdvectionJacobian_weak_lowmem(Advection_q[('df',ci,ck)],
                                                                  self._operatorQ[('v',ck)],
                                                                  Advection_q[('grad(w)*dV_f',ci)],
                                                                  Advection_Jacobian[ci][ck])
        self._createOperator(self.AdvectionOperatorCoeff, Advection_Jacobian, self.TPAdvectionOperator)

    def updateTPInvScaleLaplaceOperator(self):
        self._laplace_val = self.OLT.nzval.copy()
        self._laplace_val.fill(0.)

        _rho_0 = self.OLT.coefficients.rho_0
        _rho_1 = self.OLT.coefficients.rho_1
        _nu_0 = self.OLT.coefficients.nu_0
        _nu_1 = self.OLT.coefficients.nu_1

        _nd = self.OLT.coefficients.nd
        if self.OLT.coefficients.nu != None:
            _nu = self.OLT.coefficients.nu

        if phase_function == None:
            self.LaplaceOperatorCoeff = TransportCoefficients.DiscreteTwoPhaseInvScaledLaplaceOperator(nd=_nd,
                                                                                                       rho_0 = _rho_0,
                                                                                                       nu_0 = _nu_0,
                                                                                                       rho_1 = _rho_1,
                                                                                                       nu_1 = _nu_1,
                                                                                                       LS_model = _phase_func)
        else:
            self.LaplaceOperatorCoeff = TransportCoefficients.DiscreteTwoPhaseInvScaledLaplaceOperator(nd=_nd,
                                                                                                       rho_0 = _rho_0,
                                                                                                       nu_0 = _nu_0,
                                                                                                       rho_1 = _rho_1,
                                                                                                       nu_1 = _nu_1,
                                                                                                       phase_function = phase_function)

        _t = 1.0

        Laplace_phi = {}
        Laplace_dphi = {}
        self._initializeLaplacePhiFunctions(Laplace_phi,Laplace_dphi)
        self._initializeSparseDiffusionTensor(self.LaplaceOperatorCoeff)

        Laplace_q = {}
        self._allocateLaplaceOperatorQStorageSpace(Laplace_q)
        self._calculateQuadratureValues(Laplace_q)
        if _nd==2:
            self.LaplaceOperatorCoeff.evaluate(_t,Laplace_q)
        self._calculateLaplaceOperatorQ(Laplace_q)

        Laplace_Jacobian = {}
        self._allocateMatrixSpace(self.LaplaceOperatorCoeff,
                                  Laplace_Jacobian)

        for ci,ckDict in self.LaplaceOperatorCoeff.diffusion.items():
            for ck,cjDict in ckDict.items():
                for cj in set(list(cjDict.keys())+list(self.LaplaceOperatorCoeff.potential[ck].keys())):
                    cfemIntegrals.updateDiffusionJacobian_weak_sd(self.LaplaceOperatorCoeff.sdInfo[(ci,ck)][0],
                                                                  self.LaplaceOperatorCoeff.sdInfo[(ci,ck)][1],
                                                                  self.OLT.phi[ck].femSpace.dofMap.l2g, #??!!??
                                                                  Laplace_q[('a',ci,ck)],
                                                                  Laplace_q[('da',ci,ck,cj)],
                                                                  Laplace_q[('grad(phi)',ck)],
                                                                  Laplace_q[('grad(w)*dV_a',ck,ci)],
                                                                  Laplace_dphi[(ck,cj)].dof,
                                                                  self._operatorQ[('v',cj)],
                                                                  self._operatorQ[('grad(v)',cj)],
                                                                  Laplace_Jacobian[ci][cj])
        self._createOperator(self.LaplaceOperatorCoeff,
                             Laplace_Jacobian,
                             self.TPInvScaledLaplaceOperator)

    def updateTwoPhaseMassOperator_rho(self):
        pass

    def updateTwoPhaseInvScaledMassOperator(self):
        _rho_0 = self.OLT.coefficients.rho_0
        _rho_1 = self.OLT.coefficients.rho_1
        _nu_0 = self.OLT.coefficients.nu_0
        _nu_1 = self.OLT.coefficients.nu_1

        self._mass_val.fill(0.)

        _nd = self.OLT.coefficients.nd
        if self.OLT.coefficients.rho != None:
            _rho = self.OLT.coefficients.rho

        if phase_function == None:
            self.MassOperatorCoeff = TransportCoefficients.DiscreteTwoPhaseMassMatrix(nd = _nd,
                                                                                      rho_0 = _rho_0,
                                                                                      nu_0 = _nu_0,
                                                                                      rho_1 = _rho_1,
                                                                                      nu_1 = _nu_1,
                                                                                      LS_model = _phase_func)
        else:
            self.MassOperatorCoeff = TransportCoefficients.DiscreteTwoPhaseMassMatrix(nd = _nd,
                                                                                      rho_0 = _rho_0,
                                                                                      nu_0 = _nu_0,
                                                                                      rho_1 = _rho_1,
                                                                                      nu_1 = _nu_1,
                                                                                      phase_function = phase_function)

        _t = 1.0

        Mass_q = {}
        self._allocateTwoPhaseMassOperatorQStorageSpace(Mass_q)
        self._calculateQuadratureValues(Mass_q)
        if _nd == 2:
            self.MassOperatorCoeff.evaluate(_t,Mass_q)
        self._calculateTwoPhaseMassOperatorQ(Mass_q)

        Mass_Jacobian = {}
        self._allocateMatrixSpace(self.MassOperatorCoeff,
                                  Mass_Jacobian)

        for ci,cjDict in self.MassOperatorCoeff.mass.items():
            for cj in cjDict:
                cfemIntegrals.updateMassJacobian_weak(Mass_q[('dm',ci,cj)],
                                                      Mass_q[('vXw*dV_m',cj,ci)],
                                                      Mass_Jacobian[ci][cj])

        self._createOperator(self.MassOperatorCoeff,Mass_Jacobian,self.TPMassOperator)

    def _initializeOperatorConstruction(self):
        """ Collect basic values used by all attach operators functions. """
        self._operatorQ = {}
        self._attachJacobianInfo(self._operatorQ)
        self._attachQuadratureInfo()
        self._attachTestInfo(self._operatorQ)
        self._attachTrialInfo(self._operatorQ)

    def _attachQuadratureInfo(self):
        """Define the quadrature type used to build operators.

        """
        self._elementQuadrature = self.model._elementQuadrature
        self._elementBoundaryQuadrature = self.model._elementBoundaryQuadrature

    def _attachJacobianInfo(self,Q):
        """ This helper function attaches quadrature data related to 'J'

        Arguments
        ---------
        Q : dict
            A dictionary to store values at quadrature points.
        """
        scalar_quad = StorageSet(shape=(self.model.mesh.nElements_global,
                                        self.model.nQuadraturePoints_element) )
        tensor_quad = StorageSet(shape={})

        tensor_quad |= set(['J',
                            'inverse(J)'])
        scalar_quad |= set(['det(J)',
                            'abs(det(J))'])

        for k in tensor_quad:
            Q[k] = numpy.zeros((self.model.mesh.nElements_global,
                                self.model.nQuadraturePoints_element,
                                self.model.nSpace_global,
                                self.model.nSpace_global),
                                'd')

        scalar_quad.allocate(Q)

        self.model.u[0].femSpace.elementMaps.getJacobianValues(self.model.elementQuadraturePoints,
                                                             Q['J'],
                                                             Q['inverse(J)'],
                                                             Q['det(J)'])
        Q['abs(det(J))'] = numpy.absolute(Q['det(J)'])

    def _attachTestInfo(self,Q):
        """ Attach quadrature data for test functions.

        Arguments
        ---------
        Q : dict
            A dictionary to store values at quadrature points.

        Notes
        -----
        TODO - This function really doesn't need to compute the whole kitchen sink.
        Find a more efficient way to handle this.
        """
        test_shape_quad = StorageSet(shape={})
        test_shapeGradient_quad = StorageSet(shape={})

        test_shape_quad |= set([('w',ci) for ci in range(self.model.nc)])
        test_shapeGradient_quad |= set([('grad(w)',ci) for ci in range(self.model.nc)])

        for k in test_shape_quad:
            Q[k] = numpy.zeros(
                (self.model.mesh.nElements_global,
                 self.model.nQuadraturePoints_element,
                 self.model.nDOF_test_element[k[-1]]),
                'd')

        for k in test_shapeGradient_quad:
            Q[k] = numpy.zeros(
                (self.model.mesh.nElements_global,
                 self.model.nQuadraturePoints_element,
                 self.model.nDOF_test_element[k[-1]],
                 self.model.nSpace_global),
                'd')

        for ci in range(self.model.nc):
            if ('w',ci) in Q:
                self.model.testSpace[ci].getBasisValues(self.model.elementQuadraturePoints,
                                                      Q[('w',ci)])
            if ('grad(w)',ci) in Q:
                self.model.testSpace[ci].getBasisGradientValues(self.model.elementQuadraturePoints,
                                                              Q[('inverse(J)')],
                                                              Q[('grad(w)',ci)])

    def _attachTrialInfo(self,Q):
        """ Attach quadrature data for trial functions.

        Arguments
        ---------
        Q : dict
            A dictionary to store values at quadrature points.
        """
        trial_shape_quad = StorageSet(shape={})
        trial_shapeGrad_quad = StorageSet(shape={})

        trial_shape_quad |= set([('v',ci) for ci in range(self.model.nc)])
        trial_shapeGrad_quad |= set([('grad(v)',ci) for ci in range(self.model.nc)])

        for k in trial_shape_quad:
            Q[k] = numpy.zeros(
                (self.model.mesh.nElements_global,
                 self.model.nQuadraturePoints_element,
                 self.model.nDOF_test_element[k[-1]]),
                'd')

        for k in trial_shapeGrad_quad:
            Q[k] = numpy.zeros(
                (self.model.mesh.nElements_global,
                 self.model.nQuadraturePoints_element,
                 self.model.nDOF_test_element[k[-1]],
                 self.model.nSpace_global),
                'd')

        for ci in range(self.model.nc):
            if ('v',ci) in Q:
                self.model.testSpace[ci].getBasisValues(self.model.elementQuadraturePoints,
                                                      Q[('v',ci)])
            if ('grad(v)',ci) in Q:
                self.model.testSpace[ci].getBasisGradientValues(self.model.elementQuadraturePoints,
                                                              Q[('inverse(J)')],
                                                              Q[('grad(v)',ci)])

    def _allocateMassOperatorQStorageSpace(self,Q):
        """ Allocate space for mass operator values. """
        test_shape_quad = StorageSet(shape={})
        trial_shape_quad = StorageSet(shape={})
        trial_shape_X_test_shape_quad = StorageSet(shape={})
        tensor_quad = StorageSet(shape={})
        # TODO - ARB : I don't think the 3 is necessary here...It created a
        # confusing bug in the 2-phase problem...Need to investigate.
        scalar_quad = StorageSet(shape=(self.model.mesh.nElements_global,
                                        self.model.nQuadraturePoints_element,
                                        3))

        scalar_quad |= set([('u',ci) for ci in range(self.model.nc)])
        scalar_quad |= set([('m',ci) for ci in list(self.MassOperatorCoeff.mass.keys())])

        test_shape_quad |= set([('w*dV_m',ci) for ci in list(self.MassOperatorCoeff.mass.keys())])

        for ci,cjDict in self.MassOperatorCoeff.mass.items():
            trial_shape_X_test_shape_quad |= set([('vXw*dV_m',cj,ci) for cj in list(cjDict.keys())])

        for ci,cjDict in self.MassOperatorCoeff.mass.items():
            scalar_quad |= set([('dm',ci,cj) for cj in list(cjDict.keys())])

        for k in tensor_quad:
            Q[k] = numpy.zeros(
                (self.model.mesh.nElements_global,
                 self.model.nQuadraturePoints_element,
                 self.model.nSpace_global,
                 self.model.nSpace_global),
                'd')

        for k in test_shape_quad:
            Q[k] = numpy.zeros(
                (self.model.mesh.nElements_global,
                 self.model.nQuadraturePoints_element,
                 self.model.nDOF_test_element[k[-1]]),
                'd')

        for k in trial_shape_X_test_shape_quad:
            Q[k] = numpy.zeros((self.model.mesh.nElements_global,
                                self.model.nQuadraturePoints_element,
                                self.model.nDOF_trial_element[k[1]],
                                self.model.nDOF_test_element[k[2]]),'d')

        scalar_quad.allocate(Q)

    def _calculateMassOperatorQ(self,Q):
        """ Calculate values for mass operator. """
        elementQuadratureDict = {}

        for ci in list(self.MassOperatorCoeff.mass.keys()):
            elementQuadratureDict[('m',ci)] = self._elementQuadrature

        (elementQuadraturePoints,elementQuadratureWeights,
         elementQuadratureRuleIndeces) = Quadrature.buildUnion(elementQuadratureDict)
        for ci in range(self.model.nc):
            if ('w*dV_m',ci) in Q:
                cfemIntegrals.calculateWeightedShape(elementQuadratureWeights[('m',ci)],
                                                     self._operatorQ['abs(det(J))'],
                                                     self._operatorQ[('w',ci)],
                                                     Q[('w*dV_m',ci)])

        for ci in zip(list(range(self.model.nc)),list(range(self.model.nc))):
                cfemIntegrals.calculateShape_X_weightedShape(self._operatorQ[('v',ci[1])],
                                                             Q[('w*dV_m',ci[0])],
                                                             Q[('vXw*dV_m',ci[1],ci[0])])

    def _allocateMatrixSpace(self,coeff,matrixDict):
        """ Allocate space for Operator Matrix """
        for ci in range(self.model.nc):
            matrixDict[ci] = {}
            for cj in range(self.model.nc):
                if cj in coeff.stencil[ci]:
                    matrixDict[ci][cj] = numpy.zeros(
                        (self.model.mesh.nElements_global,
                         self.model.nDOF_test_element[ci],
                         self.model.nDOF_trial_element[cj]),
                        'd')

    def _createOperator(self,coeff,matrixDict,A):
        """ Takes the matrix dictionary and creates a CSR matrix """
        for ci in range(self.model.nc):
            for cj in coeff.stencil[ci]:
                cfemIntegrals.updateGlobalJacobianFromElementJacobian_CSR(self.model.l2g[ci]['nFreeDOF'],
                                                                          self.model.l2g[ci]['freeLocal'],
                                                                          self.model.l2g[cj]['nFreeDOF'],
                                                                          self.model.l2g[cj]['freeLocal'],
                                                                          self.model.csrRowIndeces[(ci,cj)],
                                                                          self.model.csrColumnOffsets[(ci,cj)],
                                                                          matrixDict[ci][cj],
                                                                          A)

    def _calculateQuadratureValues(self,Q):
        elementQuadratureDict = {}
        elementQuadratureDict[('m',1)] = self._elementQuadrature
        (elementQuadraturePoints,elementQuadratureWeights,
         elementQuadratureRuleIndeces) = Quadrature.buildUnion(elementQuadratureDict)
        self.model.u[0].femSpace.elementMaps.getValues(elementQuadraturePoints, Q['x'])

    def _allocateAdvectionOperatorQStorageSpace(self,Q):
           """Allocate storage space for the Advection operator values. """
           scalar_quad = StorageSet(shape=(self.model.mesh.nElements_global,
                                           self.model.nQuadraturePoints_element))
           points_quadrature = StorageSet(shape=(self.model.mesh.nElements_global,
                                                 self.model.nQuadraturePoints_element,
                                                 3))
           vector_quad = StorageSet(shape=(self.model.mesh.nElements_global,
                                           self.model.nQuadraturePoints_element,
                                           self.model.nSpace_global))
           tensor_quad = StorageSet(shape=(self.model.mesh.nElements_global,
                                           self.model.nQuadraturePoints_element,
                                           self.model.nSpace_global))
           gradients = StorageSet(shape={})

           points_quadrature |= set(['x'])
           scalar_quad |= set([('u',0)])
           vector_quad |= set([('f',ci) for ci in range(self.model.nc)])
           tensor_quad |= set([('df',0,0)])

           for i in range(self.model.nc):
               for j in range(1,self.model.nc):
                   tensor_quad |= set([('df',i,j)])

           gradients |= set([('grad(w)*dV_f',ci) for ci in list(self.AdvectionOperatorCoeff.advection.keys())])

           scalar_quad.allocate(Q)
           vector_quad.allocate(Q)

           for k in tensor_quad:
               Q[k] = numpy.zeros(
                   (self.model.mesh.nElements_global,
                    self.model.nQuadraturePoints_element,
                    self.model.nSpace_global),
                   'd')

           for k in gradients:
               Q[k] = numpy.zeros(
                   (self.model.mesh.nElements_global,
                    self.model.nQuadraturePoints_element,
                    self.model.nDOF_test_element[k[-1]],
                    self.model.nSpace_global),
                   'd')

           points_quadrature.allocate(Q)

class ChebyshevSemiIteration(LinearSolver):
    """ Class for implementing the ChebyshevSemiIteration.

    Notes
    -----
    The Chebyshev semi-iteration was developed in the 1960s
    by Golub and Varga.  It is an iterative technique for
    solving linear systems Ax = b with the property of preserving
    linearity with respect to the Krylov solves (see Wathen,
    Rees 2009 - Chebyshev semi-iteration in preconditioning
    for problems including the mass matrix).  This makes the method
    particularly well suited for solving sub-problems that arise in
    more complicated block preconditioners such as the Schur
    complement, provided one has an aprior bound on the systems
    eigenvalues (see Wathen 1987 - Realisitc eigenvalue bounds for
    Galerkin mass matrix).

    When implementing this method it is important you first
    have tight aprior bounds on the eigenvalues (denoted here as
    alpha and beta). This can be a challenge, but the references
    above do provide these results for many relevant mass matrices.

    Also, when implementing this method, the residual b - Ax0
    will be preconditioned with the inverse of diag(A).  Your eigenvalue
    bounds should reflect the spectrum of this preconditioned system.

    Arugments
    ---------
    A : :class: `p4pyPETSc.Mat`
        The linear system matrix

    alpha : float
        A's smallest eigenvalue

    beta : float
        A's largest eigenvalue

    save_iterations : bool
        A flag indicating whether to store each solution iteration

    Notes
    -----
    The Chebyshev semi-iteration is often used to solve subproblems of
    larger linear systems.  Thus, it is common that this class will
    use petsc4py matrices and vectors.  Currently this is the
    only case that is implemented, but an additional constructor
    has been included in the API for superlu matrices for future
    implementations.
    """

    def __init__(self,
                 A,
                 alpha,
                 beta,
                 save_iterations = False):
        self.A_petsc = A

        # Initialize Linear Solver with superlu matrix
        num_rows = A.getSizes()[1][0]
        num_cols = A.getSizes()[1][1]
        (self.rowptr, self.colind, self.nzval) = A.getValuesCSR()
        A_superlu = SparseMat(num_rows,
                              num_cols,
                              self.nzval.shape[0],
                              self.nzval,
                              self.colind,
                              self.rowptr)
        LinearSolver.__init__(self, A_superlu)

        self.r_petsc = p4pyPETSc.Vec().createWithArray(numpy.zeros(num_cols))
        self.r_petsc.setType('mpi')
        self.r_petsc_array = self.r_petsc.getArray()

        self.alpha = alpha
        self.beta = beta
        self.relax_parameter = (self.alpha + self.beta)/ 2.
        self.rho = (self.beta - self.alpha)/(self.alpha + self.beta)

        self.diag = self.A_petsc.getDiagonal().copy()
        self.diag.scale(self.relax_parameter)
        self.z = self.A_petsc.getDiagonal().copy()

        self.save_iterations = save_iterations
        if self.save_iterations:
            self.iteration_results = []

    @classmethod
    def chebyshev_superlu_constructor(cls):
        """
        Future home of a Chebyshev constructor for superlu matrices.
        """
        raise RuntimeError('This function is not implmented yet.')

    def apply(self, b, x, k=5):
        """ This function applies the Chebyshev-semi iteration.

        Parameters
        ----------
        b : :class: `p4pyPETSc.Vec`
            The righthand side vector

        x : :class: `p4pyPETSc.Vec`
            An initial guess for the solution.  Note that the
            solution will be returned in the vector too.

        k : int
            Number of iterations

        Returns
        -------
        x : :class:`p4pyPETSc.Vec`
            The result of the Chebyshev semi-iteration.
        """
        self.x_k = x
        self.x_k_array = self.x_k.getArray()
        self.x_km1 = x.copy()
        self.x_km1.zeroEntries()
        if self.save_iterations:
            self.iteration_results.append(self.x_k_array.copy())

        # Since b maybe locked in some instances, we create a copy
        b_copy = b.copy()

        for i in range(k):
            w = 1./(1-(self.rho**2)/4.)
            self.r_petsc_array.fill(0.)
            self.computeResidual(self.x_k_array,
                                 self.r_petsc_array,
                                 b_copy.getArray())
            # x_kp1 = w*(z + x_k - x_km1) + x_km1
            self.r_petsc.scale(-1)
            self.z.pointwiseDivide(self.r_petsc, self.diag)
            self.z.axpy(1., self.x_k)
            self.z.axpy(-1., self.x_km1)
            self.z.scale(w)
            self.z.axpy(1., self.x_km1)
            self.x_km1 = self.x_k.copy()
            self.x_k = self.z.copy()
            self.x_k_array = self.x_k.getArray()
            if self.save_iterations:
                self.iteration_results.append(self.x_k.getArray().copy())
        x.setArray(self.x_k_array)

# The implementation that is commented out here was adopted from
# the 1996 text Iterative Solution Methods by Owe Axelsson starting
# on page 179.  As currently written, this algorithm requires
# inputing the inverse diagonally preconditioned matrix A and b.  I'm
# not sure this is the best approach, but I'd like to leave this code
# in place for now in case it is useful in the future.

    # def _calc_theta_ell(self, ell):
    #     return ((2*ell + 1) / (2. * self.k) ) * math.pi

    # def _calc_tau(self, ell):
    #     theta = self._calc_theta_ell(ell)
    #     one_over_tau = ( (self.beta - self.alpha) / 2. * math.cos(theta) +
    #                      (self.beta + self.alpha) / 2.)
    #     return 1. / one_over_tau

    # def apply(self):
    #     for i in range(self.k):
    #         if i==0 and self.save_iterations:
    #             self.iteration_results.append(self.x_k.getArray().reshape(self.n,1).copy())
    #         elif i > 0:
    #             tau = self._calc_tau(i-1)
    #             resid = self._calc_residual(self.x_k)
    #             self.x_k.axpy(-tau, resid)
    #             if self.save_iterations:
    #                 self.iteration_results.append(self.x_k.getArray().reshape(self.n,1).copy())

class SolverNullSpace(object):

    def __init__(self,
                 proteus_ksp):
        self._set_global_ksp(proteus_ksp)

    @staticmethod
    def get_name():
        return 'no_null_space'

    @staticmethod
    def apply_to_schur_block(global_ksp):
        pass

    def _set_global_ksp(self,
                        proteus_ksp):
        self.proteus_ksp = proteus_ksp

    def get_global_ksp(self):
        return self.proteus_ksp

    def apply_ns(self,
                 par_b):
        pass

NoNullSpace = SolverNullSpace

class NavierStokesConstantPressure(SolverNullSpace):

    def __init__(self,
                 proteus_ksp):
        super(NavierStokesConstantPressure, self).__init__(proteus_ksp)

    @staticmethod
    def get_name():
        return 'constant_pressure'

    @staticmethod
    def apply_to_schur_block(global_ksp):
        nsp = p4pyPETSc.NullSpace().create(comm=p4pyPETSc.COMM_WORLD,
                                           vectors = (),
                                           constant = True)
        global_ksp.pc.getFieldSplitSubKSP()[1].getOperators()[0].setNullSpace(nsp)
        global_ksp.pc.getFieldSplitSubKSP()[1].getOperators()[1].setNullSpace(nsp)

    def apply_ns(self,
                 par_b):
        """
        Applies the global null space created from a pure Neumann boundary
        problem.

        Arguments
        ---------
        par_b : petsc4py vector
        """
        # Check whether a global null space vector for a constant
        # pressure has been created.  If not, create one.
        try:
            self.pressure_null_space
        except AttributeError:
            self._defineNullSpaceVec(par_b)
            self.pressure_null_space = p4pyPETSc.NullSpace().create(constant=False,
                                                                    vectors=self.global_null_space,
                                                                    comm=p4pyPETSc.COMM_WORLD)

        # Using the global constant pressure null space, assign it to
        # the global ksp object
        self.get_global_ksp().ksp.getOperators()[0].setNullSpace(self.pressure_null_space)

    def _defineNullSpaceVec(self,
                            par_b):
        """ Setup a global null space vector.

        TODO (ARB)
        ----------
        This needs to be tested more throughly for parallel
        implmentations.
        """
        from proteus import Comm
        comm = Comm.get()
        ksp = self.get_global_ksp()
        stabilized = False
        if ksp.par_L.pde.u[0].femSpace.dofMap.nDOF_all_processes==ksp.par_L.pde.u[1].femSpace.dofMap.nDOF_all_processes:
            stabilized = True

        rank = p4pyPETSc.COMM_WORLD.rank
        size = p4pyPETSc.COMM_WORLD.size
        null_space_vector = par_b.copy()
        null_space_vector.getArray().fill(0.)
        N_DOF_pressure = ksp.par_L.pde.u[0].femSpace.dofMap.nDOF_all_processes
        N_DOF_pressure_subdomain_owned = ksp.par_L.pde.u[0].femSpace.dofMap.dof_offsets_subdomain_owned[comm.rank()+1] -ksp.par_L.pde.u[0].femSpace.dofMap.dof_offsets_subdomain_owned[comm.rank()]
        if stabilized:
            tmp = null_space_vector.getArray()[::ksp.par_L.pde.nSpace_global+1]
            assert ksp.par_L.pde.isActiveDOF_p[:N_DOF_pressure_subdomain_owned].shape[0] == tmp.shape[0], str(tmp.shape) + " "+ str(ksp.par_L.pde.isActiveDOF_p[:N_DOF_pressure_subdomain_owned].shape)
            N_ACTIVE_DOF_pressure = comm.globalSum(ksp.par_L.pde.isActiveDOF_p[:N_DOF_pressure_subdomain_owned].sum())
            tmp[:] = np.where(ksp.par_L.pde.isActiveDOF_p[:N_DOF_pressure_subdomain_owned]==1.0, 1.0/sqrt(N_ACTIVE_DOF_pressure),0.0)
        else:
            n_DOF_pressure = ksp.par_L.pde.u[0].femSpace.dofMap.nDOF
            tmp = null_space_vector.getArray()[0:n_DOF_pressure]
            tmp[:] = 1.0/(sqrt(N_DOF_pressure))
        null_space_vector.assemblyBegin()
        null_space_vector.assemblyEnd()
        self.global_null_space = [null_space_vector]

class ConstantComponentsNullSpace(SolverNullSpace):
    """Constants in some components only: e.g. the pressure of an
    incompressible flow whose velocity is given on the whole boundary.

    The components are ``coefficients.nullSpaceComponents`` (indices).
    The vector is 1 on every free degree of freedom of those components,
    0 elsewhere, normalized; it is set as both the null space and the
    transpose null space of the operator, so KSP also removes it from the
    right-hand side. That is right when the constant is in the left null
    space too, as it is for a velocity-pressure system with Dirichlet
    velocity: the continuity rows tested with q = 1 sum to the boundary
    flux of interior basis functions, which is zero.

    In serial and in parallel alike: each rank sets the entries it owns.
    Which component an owned entry belongs to comes from the model's own
    layout -- ``owned_local`` maps a rank's owned PETSc entries to its
    local degrees of freedom, and ``offset``/``stride`` map those to
    components -- so it holds for interleaved (equal-order, block size nc)
    and component-blocked (mixed spaces) layouts. In serial ``owned_local``
    is the identity.
    """
    def __init__(self, proteus_ksp):
        super(ConstantComponentsNullSpace, self).__init__(proteus_ksp)

    @staticmethod
    def get_name():
        return 'constant_components'

    @staticmethod
    def components_of_owned(model):
        """The component of each degree of freedom this rank owns, in the
        order of its part of the PETSc vector."""
        local = sum(model.nFreeDOF_global)
        component = np.full(local, -1, dtype='i')
        for ci in range(model.coefficients.nc):
            component[model.offset[ci] +
                      model.stride[ci] * np.arange(model.nFreeDOF_global[ci])] = ci
        return component[np.asarray(model.owned_local)]

    def apply_ns(self, par_b):
        try:
            self.components_null_space
        except AttributeError:
            model = self.get_global_ksp().par_L.pde
            vector = par_b.copy()
            values = vector.getArray()
            owned = self.components_of_owned(model)
            if owned.shape[0] != values.shape[0]:
                raise RuntimeError("ConstantComponentsNullSpace: %d owned degrees of "
                                   "freedom, but the vector holds %d on this rank"
                                   % (owned.shape[0], values.shape[0]))
            mask = np.isin(owned, list(model.coefficients.nullSpaceComponents))
            n = p4pyPETSc.COMM_WORLD.tompi4py().allreduce(int(mask.sum()))
            values.fill(0.0)
            values[mask] = 1.0
            values /= sqrt(n)
            vector.assemblyBegin()
            vector.assemblyEnd()
            self.components_null_space = p4pyPETSc.NullSpace().create(
                constant=False, vectors=[vector], comm=p4pyPETSc.COMM_WORLD)
        operator = self.get_global_ksp().ksp.getOperators()[0]
        operator.setNullSpace(self.components_null_space)
        operator.setTransposeNullSpace(self.components_null_space)


class ConstantNullSpace(SolverNullSpace):
    def __init__(self,
                 proteus_ksp):
        super(ConstantNullSpace, self).__init__(proteus_ksp)

    @staticmethod
    def get_name():
        return 'constant'

    def apply_ns(self,
                 par_b):
        """
        Applies the global null space created from a pure Neumann boundary
        problem.

        Arguments
        ---------
        par_b : petsc4py vector
        """
        # Check whether a global null space vector for a constant
        # has been created.  If not, create one.
        try:
            self.constant_null_space
        except AttributeError:
            self.constant_null_space = p4pyPETSc.NullSpace().create(constant=True,
                                                                    comm=p4pyPETSc.COMM_WORLD)

        # Using the global constant pressure null space, assign it to
        # the global ksp object.
        self.get_global_ksp().ksp.getOperators()[0].setNullSpace(self.constant_null_space)
