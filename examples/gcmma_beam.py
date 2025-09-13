# SPDX-License-Identifier: GPL-3.0-or-later

"""
GCMMA-MMA-Python

This file is part of GCMMA-MMA-Python. GCMMA-MMA-Python is licensed under the terms of GNU 
General Public License as published by the Free Software Foundation. For more information and 
the LICENSE file, see <https://github.com/arjendeetman/GCMMA-MMA-Python>. 

The orginal work is written by Krister Svanberg in MATLAB. This is the Python implementation 
of the code written by Arjen Deetman.

This script solves the "beam problem" from the MMA paper of Krister Svanberg. 

    Minimize:
        0.0624*(x(1) + x(2) + x(3) + x(4) + x(5))
    
    Subject to:
        61/(x(1)^3) + 37/(x(2)^3) + 19/(x(3)^3) +  7/(x(4)^3) +  1/(x(5)^3) <= 1,
        1 <= x(j) <= 10, for j=1,..,5.
"""

# Loading modules
from __future__ import division
from mmapy import gcmmasub, kktcheck, asymp, concheck, raaupdate
from util import setup_logger
from typing import Tuple
import numpy as np
import os


def main() -> None:
    # Logger
    path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(path, "gcmma_beam.log")
    logger = setup_logger(file)
    logger.info("Started\n")
    
    # Set numpy print options
    np.set_printoptions(precision=4, formatter={'float': '{: 0.4f}'.format})
    
    # Beam initial settings
    m = 1
    n = 5
    epsimin = 1e-7
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))
    zerom = np.zeros((m, 1))
    xval = 5 * eeen
    xold1 = xval.copy()
    xold2 = xval.copy()
    xmin = eeen.copy()
    xmax = 10 * eeen
    low = xmin.copy()
    upp = xmax.copy()
    c = 1000 * eeem
    d = eeem.copy()
    a0 = 1
    a = zerom.copy()
    raa0 = 0.01
    raa = 0.01 * eeem
    raa0eps = 1e-6
    raaeps = 1e-6 * eeem
    innerit = 0
    outeriter = 0
    maxoutit = 20
    kkttol = 0
    
    # Calculate function values and gradients of the objective and constraints functions
    if outeriter == 0:
        f0val, df0dx, fval, dfdx = beam2(xval)
        outvector1 = np.array([outeriter, innerit, f0val, fval])
        outvector2 = xval.flatten()
        
        # Log
        logger.info("outvector1 = {}".format(outvector1))
        logger.info("outvector2 = {}\n".format(outvector2))
    
    # The iterations start
    kktnorm = kkttol + 10
    outit = 0
    
    while kktnorm > kkttol and outit < maxoutit:
        outit += 1
        outeriter += 1
        
        # The parameters low, upp, raa0, and raa are calculated:
        low, upp, raa0, raa = asymp(
            outeriter, n, xval, xold1, xold2, xmin, xmax, low, upp, raa0, raa, raa0eps, raaeps, df0dx, dfdx)
        
        # The MMA subproblem is solved at the point xval:
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = gcmmasub(
            m, n, outeriter, epsimin, xval, xmin, xmax, low, upp, raa0, raa, f0val, df0dx, fval, dfdx, a0, a, c, d)
        
        # The user should now calculate function values (no gradients) of the objective and constraint
        # functions at the point xmma (the optimal solution of the subproblem).
        f0valnew, fvalnew = beam1(xmma)
        
        # It is checked if the approximations are conservative:
        conserv = concheck(m, epsimin, f0app, f0valnew, fapp, fvalnew)
        
        # While the approximations are non-conservative (conserv = 0), repeated inner iterations are made:
        innerit = 0
        if conserv == 0:
            while conserv == 0 and innerit <= 15:
                innerit += 1
                
                # New values on the parameters raa0 and raa are calculated:
                raa0, raa = raaupdate(
                    xmma, xval, xmin, xmax, low, upp, f0valnew, fvalnew, f0app, fapp, raa0, raa, raa0eps, raaeps, epsimin)
                
                # The GCMMA subproblem is solved with these new raa0 and raa:
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = gcmmasub(
                    m, n, outeriter, epsimin, xval, xmin, xmax, low, upp, raa0, raa, f0val, df0dx, fval, dfdx, a0, a, c, d)
                
                # The user should now calculate function values (no gradients) of the objective and 
                # constraint functions at the point xmma (the optimal solution of the subproblem).
                f0valnew, fvalnew = beam1(xmma)
                
                # It is checked if the approximations have become conservative:
                conserv = concheck(m, epsimin, f0app, f0valnew, fapp, fvalnew)
        
        # Some vectors are updated:
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        
        # Re-calculate function values and gradients of the objective and constraints functions
        f0val, df0dx, fval, dfdx = beam2(xval)
        
        # The residual vector of the KKT conditions is calculated
        residu, kktnorm, residumax = kktcheck(
            m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, fval, dfdx, a0, a, c, d)
        
        outvector1 = np.array([outeriter, innerit, f0val, fval])
        outvector2 = xval.flatten()
        
        # Log
        logger.info("outvector1 = {}".format(outvector1))
        logger.info("outvector2 = {}".format(outvector2))
        logger.info("kktnorm    = {}\n".format(kktnorm))
    
    # Final log
    logger.info("Finished")


def beam1(xval: np.ndarray) -> Tuple[float, float]:
    nx = 5
    eeen = np.ones((nx, 1))
    c1 = 0.0624
    c2 = 1
    aaa = np.array([[61.0, 37.0, 19.0, 7.0, 1.0]]).T
    xval2 = xval * xval
    xval3 = xval2 * xval
    xinv3 = eeen / xval3
    f0val = c1 * np.dot(eeen.T, xval)
    fval = np.dot(aaa.T, xinv3) - c2
    return f0val, fval


def beam2(xval: np.ndarray) -> Tuple[float, np.ndarray, float, np.ndarray]:
    nx = 5
    eeen = np.ones((nx, 1))
    c1 = 0.0624
    c2 = 1
    aaa = np.array([[61.0, 37.0, 19.0, 7.0, 1.0]]).T
    xval2 = xval * xval
    xval3 = xval2 * xval
    xval4 = xval2 * xval2
    xinv3 = eeen / xval3
    xinv4 = eeen / xval4
    f0val = c1 * np.dot(eeen.T, xval).item()
    df0dx = c1 * eeen
    fval = (np.dot(aaa.T, xinv3) - c2).item()
    dfdx = -3 * (aaa * xinv4).T
    return f0val, df0dx, fval, dfdx


if __name__ == "__main__":
    main()