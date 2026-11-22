"""Kirsch solution: infinite plate with a circular hole of radius a under remote uniaxial stress S in x, plane strain."""
import numpy as np


def displacement(x, y, S, a, E, nu):
    G = E/(2.0*(1.0 + nu))
    kappa = 3.0 - 4.0*nu
    r = np.sqrt(x*x + y*y)
    th = np.arctan2(y, x)
    c2, s2 = np.cos(2*th), np.sin(2*th)
    f = S*a/(8.0*G)
    ur = f*((r/a)*((kappa - 1.0) + 2.0*c2) + (2.0*a/r)*(1.0 + (kappa + 1.0)*c2) - (2.0*a**3/r**3)*c2)
    ut = f*(-(2.0*r/a) - (2.0*a/r)*(kappa - 1.0) - (2.0*a**3/r**3))*s2
    c, s = np.cos(th), np.sin(th)
    return ur*c - ut*s, ur*s + ut*c


def stress(x, y, S, a, nu):
    """Returns sxx, syy, szz, sxy (plane strain)."""
    r = np.sqrt(x*x + y*y)
    th = np.arctan2(y, x)
    c2, s2 = np.cos(2*th), np.sin(2*th)
    q = (a/r)**2
    srr = 0.5*S*(1 - q) + 0.5*S*(1 - 4*q + 3*q*q)*c2
    stt = 0.5*S*(1 + q) - 0.5*S*(1 + 3*q*q)*c2
    srt = -0.5*S*(1 + 2*q - 3*q*q)*s2
    c, s = np.cos(th), np.sin(th)
    sxx = srr*c*c - 2*srt*s*c + stt*s*s
    syy = srr*s*s + 2*srt*s*c + stt*c*c
    sxy = (srr - stt)*s*c + srt*(c*c - s*s)
    return sxx, syy, nu*(sxx + syy), sxy
