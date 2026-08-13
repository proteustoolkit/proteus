import numpy as np
from pytest import approx
from proteus import equivalent_polynomials as eqp

def test_1D():
    from proteus.Quadrature import GaussEdge
    polyOrders = [1,2,3]
    quadOrderMax = 9
    elements = [[0.,1.],[-2.,2.],[9.,10.]]
    phiList  = [[-1.,-1.],[1.,1.],[-1.,1.],[0.,1.],[1.,0.]]
    for nP in polyOrders:
        for qO in range(nP,quadOrderMax):
            quad = GaussEdge(order=qO)
            gf = eqp.Simplex(nSpace=1, nP=nP, nQ=len(quad.points))
            for phi in phiList:
                for e in elements:
                    dV = e[1]-e[0]
                    assert(dV > 0)
                    if phi[0]*phi[1] < 0.0:
                        theta = -phi[0]/(phi[1] - phi[0])
                        x_0 = (1-theta)*e[0] + theta*e[1]
                        int_H_exact = abs(e[1] - x_0)
                        int_ImH_exact = abs(x_0 - e[0])
                        int_D_exact = 1.0
                        if phi[0] > 0.0:
                            tmp=int_H_exact
                            int_H_exact = int_ImH_exact
                            int_ImH_exact = tmp
                    elif phi[0] < 0.0:
                        int_H_exact = 0.0
                        int_ImH_exact = dV
                        int_D_exact = 0.0
                    elif phi[0] > 0.0:
                        int_H_exact = dV
                        int_ImH_exact = 0.0
                        int_D_exact = 0.0
                    elif phi[0] == 0.0:
                        if phi[1] > 0.0:
                            int_H_exact = dV
                            int_ImH_exact = 0.0
                            int_D_exact = 0.0
                        if phi[1] < 0.0:
                            int_H_exact = 0.0
                            int_ImH_exact = dV
                            int_D_exact = 1.0
                    elif phi[1] == 0.0:
                        if phi[0] > 0.0:
                            int_H_exact = dV
                            int_ImH_exact = 0.0
                            int_D_exact = 0.0
                        if phi[0] < 0.0:
                            int_H_exact = 0.0
                            int_ImH_exact = dV
                            int_D_exact = 1.0
                            
                    gf.calculate(np.array(phi),
                                 np.array([[e[0],0.,0.],
                                           [e[1],0.,0.]]),
                                 np.array(quad.points))
                    int_H=0.0
                    int_ImH=0.0
                    int_D=0.0
                    for k in range(len(quad.points)):
                        gf.set_quad(k)
                        int_H += quad.weights[k]*gf.H*dV
                        int_ImH += quad.weights[k]*gf.ImH*dV
                        int_D += quad.weights[k]*gf.D*dV
                    assert(int_H == approx(int_H_exact,1e-15))
                    assert(int_ImH == approx(int_ImH_exact,1e-15))
                    assert(int_D == approx(int_D_exact,1e-15))

def test_2D():
    from proteus.Quadrature import GaussTriangle
    polyOrders = [1,2,3]
    quadOrderMax = 6
    elements = [
        np.array([[0.,0.],[0.,1.],[1.,0.]])
    ]
    phiList  = [[-1.,-1.,-1.],
                [1.,1.,1.],
                [-1.,1.,1.],
                [0.,1.,1.],
                [1.,0.,0.]]
    for nP in polyOrders:
        for qO in range(nP,quadOrderMax):
            quad = GaussTriangle(order=qO)
            gf = eqp.Simplex(nSpace=2, nP=nP, nQ=len(quad.points))
            for phi in phiList:
                for e in elements:
                    b_0 = e[2,:] - e[0,:]
                    b_1 = e[1,:] - e[0,:]
                    Jac = np.array([b_0,b_1]).transpose()
                    dV = abs(np.linalg.det(Jac))
                    area = dV/2.0
                    if phi[0]*phi[1] < 0.0 and phi[0]*phi[2] < 0.0:
                        theta0 = -phi[0]/(phi[1] - phi[0])
                        theta1 = -phi[0]/(phi[2] - phi[0])
                        x_0 = (1-theta0)*e[0,:] + theta0*e[1,:]
                        x_1 = (1-theta1)*e[0,:] + theta1*e[2,:]
                        b_0 = x_1 - e[0,:]
                        b_1 = x_0 - e[0,:]
                        Jac_0 = np.array([b_0,b_1]).transpose()
                        int_ImH_exact = np.linalg.det(Jac_0)/2.0
                        int_H_exact = area - int_ImH_exact
                        int_D_exact = np.linalg.norm(x_1 - x_0)
                        if phi[0] > 0.0:
                            tmp=int_H_exact
                            int_H_exact = int_ImH_exact
                            int_ImH_exact = tmp
                    elif phi[0] > 0.0 and phi[1] == 0.0 and phi[2] == 0.0:
                        int_H_exact = area
                        int_ImH_exact = 0.0
                        int_D_exact = np.linalg.norm(e[1,:] - e[2,:])
                    elif phi[0] < 0.0:
                        int_H_exact = 0.0
                        int_ImH_exact = area
                        int_D_exact = 0.0
                    elif phi[0] > 0.0:
                        int_H_exact = area
                        int_ImH_exact = 0.0
                        int_D_exact = 0.0
                    elif phi[0] == 0.0:
                        if phi[1] > 0.0:
                            int_H_exact = area
                            int_ImH_exact = 0.0
                            int_D_exact = 0.0
                        if phi[1] < 0.0:
                            int_H_exact = 0.0
                            int_ImH_exact = area
                            int_D_exact = 1.0
                    elif phi[1] == 0.0:
                        if phi[0] > 0.0:
                            int_H_exact = area
                            int_ImH_exact = 0.0
                            int_D_exact = 0.0
                        if phi[0] < 0.0:
                            int_H_exact = 0.0
                            int_ImH_exact = area
                            int_D_exact = 1.0
                            
                    gf.calculate(np.array(phi),
                                 np.array([[e[0,0],e[0,1],0.],
                                           [e[1,0],e[1,1],0.],
                                           [e[2,0],e[2,1],0.]]),
                                 np.array(quad.points))
                    int_H=0.0
                    int_ImH=0.0
                    int_D=0.0
                    for k in range(len(quad.points)):
                        gf.set_quad(k)
                        int_H += quad.weights[k]*gf.H*dV
                        int_ImH += quad.weights[k]*gf.ImH*dV
                        int_D += quad.weights[k]*gf.D*dV
                    assert(int_H == approx(int_H_exact,1e-15))
                    assert(int_ImH == approx(int_ImH_exact,1e-15))
                    assert(int_D == approx(int_D_exact,1e-15))

def test_3D():
    from proteus.Quadrature import GaussTetrahedron
    polyOrders = [1,2,3]
    quadOrderMax = 8
    elements = [
        np.array([[0.,0.,0.],
                  [1.,0.,0.],
                  [0.,1.,0.],
                  [0.,0.,1.]]),
       np.array([[659.82702468, 583.95327591, 217.72111178],
                 [312.90104741, 109.12205319,  82.38957441],
                 [371.40050898, 837.09283397, 117.87194133],
                 [666.41881358, 107.49166421, 419.82032267]]),
       np.array([[1.,1.,0.5],
                 [0.,0.,1.25],
                 [0.,1.,0.5],
                 [0.,0.,0.5,]])
    ]
    phiList=  [
        [-1.,-1.,-1.,-1.],
        [1.,1.,1.,1.],
        [-1.,1.,1.,1.],
        [1.,-1.,-1.,-1.],
        [-1.,1.,-1.,-1.],
        [1.,-1.,1.,1.],
        [-1.,1.,-1.,-1.],
        [1.,1.,-1.,1.],
        [-1.,-1.,1.,-1.],
        [1.,1.,1.,-1.],
        [-1.,-1.,-1.,1.],
        [-1.,1.,1.,-1.],
        [1.,-1.,-1.,1.],
        [-1.,1.,1.,-1.],
        [-2.,3.,4.,5.],
        [3.,-2.,-4.,-5.],
        [-2.,3.,-4.,-5.],
        [3.,-2.,4.,5.],
        [-2.,3.,-4.,-5.],
        [3.,4.,-2.,5.],
        [-2.,-3.,4.,-5.],
        [3.,4.,5.,-2.],
        [-2.,-3.,-4.,5.],
        [0.,1.,1.,1.],
        [9.52267983, -348.3143481, -10.31124012, -2.73674249],
        [-9.52267983, 348.3143481, -10.31124012, -2.73674249],
        [-9.52267983, -348.3143481, 10.31124012, -2.73674249],
        [-9.52267983, -348.3143481, -10.31124012, 2.73674249],
        [0.5, -0.25, 0.5, 0.5]]
    for nP in polyOrders:
        for qO in range(nP,quadOrderMax):
            quad = GaussTetrahedron(order=qO)
            print("Simplex in ",3,nP,len(quad.points))
            gf = eqp.Simplex(nSpace=3, nP=nP, nQ=len(quad.points))
            for phi in phiList:
                for e in elements:
                    b_0 = e[3,:] - e[0,:]
                    b_1 = e[2,:] - e[0,:]
                    b_2 = e[1,:] - e[0,:]
                    Jac = np.array([b_0, b_1, b_2]).transpose()
                    dV = abs(np.linalg.det(Jac))
                    volume = dV/6.0
                    if phi[0] < 0.0 and phi[1] > 0.0 and phi[2] > 0.0 and phi[3] < 0.0:
                        print("====quad cut===== ",phi)
                        print("qO ",qO)
                        print("nP ",nP)
                        theta01 = 0.5 - 0.5*(phi[1]+phi[0])/(phi[1] - phi[0])
                        theta02 = 0.5 - 0.5*(phi[2]+phi[0])/(phi[2] - phi[0])
                        theta31 = 0.5 - 0.5*(phi[1]+phi[3])/(phi[1] - phi[3])
                        theta32 = 0.5 - 0.5*(phi[2]+phi[3])/(phi[2] - phi[3])
                        x_01 = (1-theta01)*e[0,:] + theta01*e[1,:]
                        x_02 = (1-theta02)*e[0,:] + theta02*e[2,:]
                        x_31 = (1-theta31)*e[3,:] + theta31*e[1,:]
                        x_32 = (1-theta32)*e[3,:] + theta32*e[2,:]
                        b_0 = x_01 - e[1,:]
                        b_1 = x_02 - e[1,:]
                        b_2 = x_31 - e[1,:]
                        Jac_0 = np.array([b_0,b_1,b_2]).transpose()
                        int_ImH_exact = volume
                        int_H_exact = 0.0
                        int_D_exact = 0.0
                        int_H_exact += abs(np.linalg.det(Jac_0))/6.0
                        int_ImH_exact -= abs(np.linalg.det(Jac_0))/6.0
                        int_D_exact += 0.5*np.linalg.norm(np.cross(x_02 - x_01,
                                                                   x_32 - x_01))
                        b_0 = x_02 - e[2,:]
                        b_1 = x_32 - e[2,:]
                        b_2 = x_31 - e[2,:]
                        Jac_0 = np.array([b_0,b_1,b_2]).transpose()
                        int_H_exact += abs(np.linalg.det(Jac_0))/6.0
                        int_ImH_exact -= abs(np.linalg.det(Jac_0))/6.0
                        int_D_exact += 0.5*np.linalg.norm(np.cross(x_32 - x_01,
                                                                   x_31 - x_01))
                        b_0 = e[1,:] - x_02
                        b_1 = e[2,:] - x_02
                        b_2 = x_31   - x_02
                        Jac_0 = np.array([b_0,b_1,b_2]).transpose()
                        int_H_exact += abs(np.linalg.det(Jac_0))/6.0
                        int_ImH_exact -= abs(np.linalg.det(Jac_0))/6.0
                    elif phi[0] > 0.0 and phi[1] < 0.0 and phi[2] < 0.0 and phi[3] > 0.0:
                        print("====quad cut=====  inside out",phi)
                        print("qO ",qO)
                        print("nP ",nP)
                        theta01 = 0.5 - 0.5*(phi[1]+phi[0])/(phi[1] - phi[0])
                        theta02 = 0.5 - 0.5*(phi[2]+phi[0])/(phi[2] - phi[0])
                        theta31 = 0.5 - 0.5*(phi[1]+phi[3])/(phi[1] - phi[3])
                        theta32 = 0.5 - 0.5*(phi[2]+phi[3])/(phi[2] - phi[3])
                        x_01 = (1-theta01)*e[0,:] + theta01*e[1,:]
                        x_02 = (1-theta02)*e[0,:] + theta02*e[2,:]
                        x_31 = (1-theta31)*e[3,:] + theta31*e[1,:]
                        x_32 = (1-theta32)*e[3,:] + theta32*e[2,:]
                        b_0 = x_01 - e[1,:]
                        b_1 = x_02 - e[1,:]
                        b_2 = x_31 - e[1,:]
                        Jac_0 = np.array([b_0,b_1,b_2]).transpose()
                        int_ImH_exact = 0.0
                        int_H_exact = volume
                        int_D_exact = 0.0
                        int_H_exact -= abs(np.linalg.det(Jac_0))/6.0
                        int_ImH_exact += abs(np.linalg.det(Jac_0))/6.0
                        int_D_exact += 0.5*np.linalg.norm(np.cross(x_02 - x_01,
                                                                   x_32 - x_01))
                        b_0 = x_02 - e[2,:]
                        b_1 = x_32 - e[2,:]
                        b_2 = x_31 - e[2,:]
                        Jac_0 = np.array([b_0,b_1,b_2]).transpose()
                        int_H_exact -= abs(np.linalg.det(Jac_0))/6.0
                        int_ImH_exact += abs(np.linalg.det(Jac_0))/6.0
                        int_D_exact += 0.5*np.linalg.norm(np.cross(x_32 - x_01,
                                                                   x_31 - x_01))
                        b_0 = e[1,:] - x_02
                        b_1 = e[2,:] - x_02
                        b_2 = x_31   - x_02
                        Jac_0 = np.array([b_0,b_1,b_2]).transpose()
                        int_H_exact -= abs(np.linalg.det(Jac_0))/6.0
                        int_ImH_exact += abs(np.linalg.det(Jac_0))/6.0
                    elif phi[0]*phi[1] < 0.0 and phi[0]*phi[2] < 0.0 and phi[0]*phi[3] < 0.0:
                        theta0 = -phi[0]/(phi[1] - phi[0])
                        theta1 = -phi[0]/(phi[2] - phi[0])
                        theta2 = -phi[0]/(phi[3] - phi[0])
                        x_0 = (1-theta0)*e[0,:] + theta0*e[1,:]
                        x_1 = (1-theta1)*e[0,:] + theta1*e[2,:]
                        x_2 = (1-theta2)*e[0,:] + theta2*e[3,:]
                        b_0 = x_2 - e[0,:]
                        b_1 = x_1 - e[0,:]
                        b_2 = x_0 - e[0,:]
                        Jac_0 = np.array([b_0,b_1,b_2]).transpose()
                        int_ImH_exact = abs(np.linalg.det(Jac_0))/6.0
                        int_H_exact = volume - int_ImH_exact
                        int_D_exact = 0.5*np.linalg.norm(np.cross(x_2 - x_0, x_1 - x_0))
                        if phi[0] > 0.0:
                            tmp=int_H_exact
                            int_H_exact = int_ImH_exact
                            int_ImH_exact = tmp
                    elif phi[0]*phi[1] < 0.0 and phi[1]*phi[2] < 0.0 and phi[1]*phi[3] < 0.0:
                        theta0 = -phi[1]/(phi[0] - phi[1])
                        theta1 = -phi[1]/(phi[2] - phi[1])
                        theta2 = -phi[1]/(phi[3] - phi[1])
                        x_0 = (1-theta0)*e[1,:] + theta0*e[0,:]
                        x_1 = (1-theta1)*e[1,:] + theta1*e[2,:]
                        x_2 = (1-theta2)*e[1,:] + theta2*e[3,:]
                        b_0 = x_2 - e[1,:]
                        b_1 = x_1 - e[1,:]
                        b_2 = x_0 - e[1,:]
                        Jac_0 = np.array([b_0,b_1,b_2]).transpose()
                        int_ImH_exact = abs(np.linalg.det(Jac_0))/6.0
                        int_H_exact = volume - int_ImH_exact
                        int_D_exact = 0.5*np.linalg.norm(np.cross(x_2 - x_0, x_1 - x_0))
                        if phi[1] > 0.0:
                            tmp=int_H_exact
                            int_H_exact = int_ImH_exact
                            int_ImH_exact = tmp
                    elif phi[0]*phi[2] < 0.0 and phi[1]*phi[2] < 0.0 and phi[3]*phi[2] < 0.0:
                        theta0 = -phi[2]/(phi[0] - phi[2])
                        theta1 = -phi[2]/(phi[1] - phi[2])
                        theta2 = -phi[2]/(phi[3] - phi[2])
                        x_0 = (1-theta0)*e[2,:] + theta0*e[0,:]
                        x_1 = (1-theta1)*e[2,:] + theta1*e[1,:]
                        x_2 = (1-theta2)*e[2,:] + theta2*e[3,:]
                        b_0 = x_2 - e[2,:]
                        b_1 = x_1 - e[2,:]
                        b_2 = x_0 - e[2,:]
                        Jac_0 = np.array([b_0,b_1,b_2]).transpose()
                        int_ImH_exact = abs(np.linalg.det(Jac_0))/6.0
                        int_H_exact = volume - int_ImH_exact
                        int_D_exact = 0.5*np.linalg.norm(np.cross(x_2 - x_0, x_1 - x_0))
                        if phi[2] > 0.0:
                            tmp=int_H_exact
                            int_H_exact = int_ImH_exact
                            int_ImH_exact = tmp
                    elif phi[0]*phi[3] < 0.0 and phi[1]*phi[3] < 0.0 and phi[2]*phi[3] < 0.0:
                        theta0 = -phi[3]/(phi[0] - phi[3])
                        theta1 = -phi[3]/(phi[1] - phi[3])
                        theta2 = -phi[3]/(phi[2] - phi[3])
                        x_0 = (1-theta0)*e[3,:] + theta0*e[0,:]
                        x_1 = (1-theta1)*e[3,:] + theta1*e[1,:]
                        x_2 = (1-theta2)*e[3,:] + theta2*e[2,:]
                        b_0 = x_2 - e[3,:]
                        b_1 = x_1 - e[3,:]
                        b_2 = x_0 - e[3,:]
                        Jac_0 = np.array([b_0,b_1,b_2]).transpose()
                        int_ImH_exact = abs(np.linalg.det(Jac_0))/6.0
                        int_H_exact = volume - int_ImH_exact
                        int_D_exact = 0.5*np.linalg.norm(np.cross(x_2 - x_0, x_1 - x_0))
                        if phi[3] > 0.0:
                            tmp=int_H_exact
                            int_H_exact = int_ImH_exact
                            int_ImH_exact = tmp
                    elif phi[0] < 0.0:
                        int_H_exact = 0.0
                        int_ImH_exact = volume
                        int_D_exact = 0.0
                    elif phi[0] > 0.0:
                        int_H_exact = volume
                        int_ImH_exact = 0.0
                        int_D_exact = 0.0
                    elif phi[0] == 0.0:
                        if phi[1] > 0.0:
                            int_H_exact = volume
                            int_ImH_exact = 0.0
                            int_D_exact = 0.0
                        if phi[1] < 0.0:
                            int_H_exact = 0.0
                            int_ImH_exact = volume
                            int_D_exact = 1.0
                    elif phi[1] == 0.0:
                        if phi[0] > 0.0:
                            int_H_exact = volume
                            int_ImH_exact = 0.0
                            int_D_exact = 0.0
                        if phi[0] < 0.0:
                            int_H_exact = 0.0
                            int_ImH_exact = volume
                            int_D_exact = 1.0
                    phi_in = np.array(phi)
                    print("phi in ", phi_in) 
                    nodesIn = e
                    print("nodes in ",nodesIn)
                    print("quad in ",np.array(quad.points))
                    gf.calculate(phi_in,
                                 nodesIn,
                                 np.array(quad.points))
                    int_H=0.0
                    int_ImH=0.0
                    int_D=0.0
                    for k in range(len(quad.points)):
                        gf.set_quad(k)
                        print("k ", k, gf.H, gf.ImH, gf.D)
                        int_H += quad.weights[k]*gf.H*dV
                        int_ImH += quad.weights[k]*gf.ImH*dV
                        int_D += quad.weights[k]*gf.D*dV
                    print("dV ", dV)
                    print(int_H, int_H_exact)
                    print(int_ImH, int_ImH_exact)
                    print(int_D, int_D_exact)
                    assert(int_H == approx(int_H_exact,1e-12,1e-12))
                    assert(int_ImH == approx(int_ImH_exact,1e-12,1e-12))
                    assert(int_D == approx(int_D_exact,1e-11,1e-11))
def test_edge_H():
    """The edge-restricted moment-fit Heaviside must reproduce, exactly, the integral of any
    polynomial of degree <= nP over the cut sub-interval:
        int_0^1 H_hat(t) t**d dt == int_theta^1 t**d dt   for d = 0..nP.
    That identity is the whole reason it can integrate a discontinuous facet integrand with an
    ordinary whole-edge quadrature rule. Checked for both edge orientations."""
    import numpy as np
    gauss_t, gauss_w = np.polynomial.legendre.leggauss(12)
    gauss_t = 0.5 * (gauss_t + 1.0)
    gauss_w = 0.5 * gauss_w
    for nP in [1, 2, 3, 4]:
        for theta in [0.05, 0.25, 0.5, 0.6180339887, 0.9]:
            for increasing in [True, False]:
                # phi linear along the edge, vanishing at t = theta
                if increasing:
                    phi0, phi1 = -theta, 1.0 - theta
                else:
                    phi0, phi1 = theta, theta - 1.0
                C_H = eqp.calc_edge_H(phi0, phi1, nP)
                Hh = np.array([eqp.eval_edge_poly(C_H, t, nP) for t in gauss_t])
                for d in range(nP + 1):
                    got = float(np.sum(gauss_w * Hh * gauss_t**d))
                    # exact integral of t**d over the region where phi > 0
                    if increasing:   # phi>0 on (theta,1)
                        exact = (1.0 - theta**(d + 1)) / (d + 1)
                    else:            # phi>0 on (0,theta)
                        exact = theta**(d + 1) / (d + 1)
                    assert got == approx(exact, abs=1.0e-10), \
                        f"nP={nP} theta={theta} inc={increasing} d={d}: {got} vs {exact}"


# ---------------------------------------------------------------------------
# Triply-cut cells: is the product of two equivalent polynomials sound?
#
# In the 3-phase (cutFEM + IFEM) setting one element can be cut by two
# independent level sets at once -- a fluid/fluid interface (phi_f) and a
# fluid/solid boundary (phi_s) -- splitting it into three regions. ADR.h weights
# the bulk terms of such an element by the *product* of two separately
# moment-fitted polynomials, e.g. ImH_f * H_s for the mua-side fluid.
#
# Each factor is exact in the moment sense on its own:
#     int_e Hhat * p = int_region p     for every p in P^nP.
# These tests use the real eqp.Simplex machinery to ask whether that survives
# multiplication, and whether one "composite" polynomial fitted against the true
# three-region geometry restores it.
#
# Only H/ImH are used. eqp's D currently returns the cut measure with a flipped
# sign (see test_2D), which is a separate pre-existing issue.
# ---------------------------------------------------------------------------

def _tc_clip(poly, n, c, keep_negative):
    """Sutherland-Hodgman clip of a convex polygon by {x : n.x + c <= 0} (or >= 0)."""
    sval = lambda p: n[0]*p[0] + n[1]*p[1] + c
    inside = lambda p: (sval(p) <= 0.0) if keep_negative else (sval(p) >= 0.0)
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i+1) % len(poly)]
        ain, bin_ = inside(a), inside(b)
        if ain:
            out.append(a)
        if ain != bin_:
            sa, sb = sval(a), sval(b)
            out.append(a + (sa/(sa - sb))*(b - a))
    return np.array(out) if len(out) >= 3 else np.zeros((0, 2))


def _tc_monomials(x, y, nP):
    return np.array([x**(d-k) * y**k for d in range(nP+1) for k in range(d+1)])


def _tc_quad(e, order):
    """Physical quadrature points/weights on triangle e, plus the reference points."""
    from proteus.Quadrature import GaussTriangle
    q = GaussTriangle(order=order)
    J = np.array([e[1]-e[0], e[2]-e[0]]).transpose()
    dV = abs(np.linalg.det(J))
    pts = np.array([e[0] + J.dot(np.array([qp[0], qp[1]])) for qp in q.points])
    return np.array(q.points), pts, np.array(q.weights)*dV


def _tc_region_moments(poly, nP, order=10):
    """Exact integral of every monomial up to degree nP over a convex polygon."""
    acc = np.zeros((nP+1)*(nP+2)//2)
    for i in range(1, max(len(poly)-1, 0)):
        _, pts, w = _tc_quad(np.array([poly[0], poly[i], poly[i+1]]), order)
        for xy, wk in zip(pts, w):
            acc += wk*_tc_monomials(xy[0], xy[1], nP)
    return acc


def _tc_gf(nP, ref_pts, e, phi):
    """An eqp.Simplex fitted to level set phi on element e."""
    gf = eqp.Simplex(nSpace=2, nP=nP, nQ=len(ref_pts))
    gf.calculate(np.array(phi),
                 np.array([[e[0,0],e[0,1],0.],[e[1,0],e[1,1],0.],[e[2,0],e[2,1],0.]]),
                 ref_pts)
    return gf


def _tc_regions(e, ns, cs, nf, cf):
    """(fluid-minus, fluid-plus, solid); ADR.h convention: phi_s>0 fluid, phi_f<0 Omega^-."""
    tri = np.array([e[0], e[1], e[2]])
    fluid = _tc_clip(tri, ns, cs, keep_negative=False)
    solid = _tc_clip(tri, ns, cs, keep_negative=True)
    fminus = _tc_clip(fluid, nf, cf, True) if len(fluid) else fluid
    fplus = _tc_clip(fluid, nf, cf, False) if len(fluid) else fluid
    return fminus, fplus, solid


# a triangle genuinely cut by both level sets, so it carries all three regions
_TC_E = np.array([[0., 0.], [0., 1.], [1., 0.]])
_TC_NS, _TC_CS = np.array([-1., -1.]), 0.35   # phi_s = 0.35 - x - y ; solid where < 0
_TC_NF, _TC_CF = np.array([-1., 1.]), -0.05   # phi_f = y - x - 0.05 ; Omega^- where < 0
_TC_PHI_S = [_TC_NS.dot(_TC_E[i]) + _TC_CS for i in range(3)]
_TC_PHI_F = [_TC_NF.dot(_TC_E[i]) + _TC_CF for i in range(3)]


def test_triply_cut_single_eqp_is_moment_exact():
    """Baseline, using eqp.Simplex: one fit reproduces its own region's moments."""
    for nP in [1, 2]:
        ref, pts, w = _tc_quad(_TC_E, 3*nP+2)
        tri = np.array([_TC_E[0], _TC_E[1], _TC_E[2]])
        for phi, region, use_H in [
                (_TC_PHI_S, _tc_clip(tri, _TC_NS, _TC_CS, False), True),
                (_TC_PHI_F, _tc_clip(tri, _TC_NF, _TC_CF, True), False)]:
            gf = _tc_gf(nP, ref, _TC_E, phi)
            got = np.zeros((nP+1)*(nP+2)//2)
            for k, (xy, wk) in enumerate(zip(pts, w)):
                gf.set_quad(k)
                got += wk*(gf.H if use_H else gf.ImH)*_tc_monomials(xy[0], xy[1], nP)
            assert got == approx(_tc_region_moments(region, nP), abs=1e-12)


def test_triply_cut_product_of_eqp_is_not_moment_exact():
    """ImH_f * H_s does NOT inherit the exactness of its factors.

    This is the weight ADR.h applies to the mua-side bulk terms of a doubly-cut
    element. Both factors are exact alone (previous test) but their product is a
    degree-2*nP polynomial that nothing constrains. Quadrature is exact to degree
    3*nP, so any discrepancy is the mathematics, not integration error.
    """
    fminus, _, _ = _tc_regions(_TC_E, _TC_NS, _TC_CS, _TC_NF, _TC_CF)
    rel = {}
    for nP in [1, 2]:
        ref, pts, w = _tc_quad(_TC_E, 3*nP+2)
        gf_s = _tc_gf(nP, ref, _TC_E, _TC_PHI_S)
        gf_f = _tc_gf(nP, ref, _TC_E, _TC_PHI_F)
        product = np.zeros((nP+1)*(nP+2)//2)
        for k, (xy, wk) in enumerate(zip(pts, w)):
            gf_s.set_quad(k); gf_f.set_quad(k)
            product += wk*gf_f.ImH*gf_s.H*_tc_monomials(xy[0], xy[1], nP)
        exact = _tc_region_moments(fminus, nP)
        rel[nP] = np.abs(product - exact).max()/abs(exact[0])
    assert min(rel.values()) > 1e-3, "product form unexpectedly accurate: %s" % rel


def test_triply_cut_composite_moment_fit_is_exact():
    """The proposed fix: fit ONE polynomial per intersected region.

    Right-hand side built from the true three-region geometry; the Gram matrix is
    that of the whole element, exactly as in the single-level-set construction.
    Exactness against every p in P^nP then holds by construction. Also checks the
    three regions partition the element.
    """
    for nP in [1, 2]:
        _, pts, w = _tc_quad(_TC_E, 2*nP+2)
        nMon = (nP+1)*(nP+2)//2
        M = np.zeros((nMon, nMon))
        for xy, wk in zip(pts, w):
            m = _tc_monomials(xy[0], xy[1], nP)
            M += wk*np.outer(m, m)
        total = np.zeros(nMon)
        for region in _tc_regions(_TC_E, _TC_NS, _TC_CS, _TC_NF, _TC_CF):
            b = _tc_region_moments(region, nP)
            assert M.dot(np.linalg.solve(M, b)) == approx(b, abs=1e-12)
            total += b
        whole = _tc_region_moments(np.array([_TC_E[0], _TC_E[1], _TC_E[2]]), nP)
        assert total == approx(whole, abs=1e-12), "regions must partition the element"


def test_triply_cut_composite_fit_survives_slivers():
    """Slivers are the risk case; the composite fit stays exact as a region vanishes.

    The Gram matrix is that of the whole element and does not depend on the cut, so
    only the right-hand side shrinks -- which is what keeps this well posed however
    thin the region becomes.
    """
    nP = 2
    _, pts, w = _tc_quad(_TC_E, 2*nP+2)
    nMon = (nP+1)*(nP+2)//2
    M = np.zeros((nMon, nMon))
    for xy, wk in zip(pts, w):
        m = _tc_monomials(xy[0], xy[1], nP)
        M += wk*np.outer(m, m)
    assert np.linalg.cond(M) < 1e6
    for cs in [0.5, 0.2, 0.05, 1.0e-2, 1.0e-4, 1.0e-8]:
        fminus, _, _ = _tc_regions(_TC_E, np.array([-1., -1.]), cs, _TC_NF, _TC_CF)
        b = _tc_region_moments(fminus, nP)
        assert M.dot(np.linalg.solve(M, b)) == approx(b, abs=1e-13), "sliver cs=%g" % cs


def _tc_segment(e, n, c):
    """The segment where the line n.x + c = 0 crosses triangle e, as (P0, P1)."""
    pts = []
    for i in range(3):
        a, b = e[i], e[(i+1) % 3]
        sa, sb = n[0]*a[0]+n[1]*a[1]+c, n[0]*b[0]+n[1]*b[1]+c
        if sa*sb < 0:
            t = sa/(sa-sb)
            pts.append(a + t*(b-a))
    return pts


def _tc_segment_moments(P0, P1, nf, cf, keep_negative, nP):
    """Exact monomial moments (arc-length) over the part of segment P0->P1 on one side of phi_f."""
    s0 = nf[0]*P0[0]+nf[1]*P0[1]+cf
    s1 = nf[0]*P1[0]+nf[1]*P1[1]+cf
    lo, hi = 0.0, 1.0
    if s0*s1 < 0:                       # the fluid interface cuts this segment
        t = s0/(s0-s1)
        inside0 = (s0 <= 0) if keep_negative else (s0 >= 0)
        lo, hi = (0.0, t) if inside0 else (t, 1.0)
    else:
        inside = (s0 <= 0) if keep_negative else (s0 >= 0)
        if not inside:
            return np.zeros((nP+1)*(nP+2)//2)
    L = np.linalg.norm(P1-P0)
    from proteus.Quadrature import GaussEdge
    q = GaussEdge(order=8)
    acc = np.zeros((nP+1)*(nP+2)//2)
    for pt, w in zip(q.points, q.weights):
        t = lo + (hi-lo)*pt[0]
        xy = P0 + t*(P1-P0)
        acc += w*(hi-lo)*L*_tc_monomials(xy[0], xy[1], nP)
    return acc


def test_triply_cut_product_with_dirac_is_not_moment_exact():
    """ImH_f * D_s does NOT give the moments of the solid surface inside the water.

    The Nitsche condition on an embedded solid has to be split between the fluid regions when
    both level sets cut the element. Weighting the Dirac fit D_s by the region fit ImH_f is the
    same product-of-two-fits error as in the volume case (section 7.1 of the notes): D_s is exact
    for all of Gamma_s, but nothing constrains ImH_f * D_s on the part of it lying in the water.
    """
    ref, pts, w = _tc_quad(_TC_E, 3*2+2)
    P = _tc_segment(_TC_E, _TC_NS, _TC_CS)
    assert len(P) == 2, "phi_s must cut the element"
    rel = {}
    for nP in [1, 2]:
        gf_s = _tc_gf(nP, ref, _TC_E, _TC_PHI_S)
        gf_f = _tc_gf(nP, ref, _TC_E, _TC_PHI_F)
        prod = np.zeros((nP+1)*(nP+2)//2)
        for k, (xy, wk) in enumerate(zip(pts, w)):
            gf_s.set_quad(k); gf_f.set_quad(k)
            prod += wk*gf_f.ImH*abs(gf_s.D)*_tc_monomials(xy[0], xy[1], nP)
        exact = _tc_segment_moments(P[0], P[1], _TC_NF, _TC_CF, True, nP)
        rel[nP] = np.abs(prod-exact).max()/abs(exact[0])
    assert min(rel.values()) > 1e-3, "product with the Dirac was unexpectedly accurate: %s" % rel
