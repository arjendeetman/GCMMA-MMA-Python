﻿# SPDX-License-Identifier: GPL-3.0-or-later

"""
GCMMA-MMA-Python

This file is part of GCMMA-MMA-Python. GCMMA-MMA-Python is licensed under the terms of GNU 
General Public License as published by the Free Software Foundation. For more information and 
the LICENSE file, see <https://github.com/arjendeetman/GCMMA-MMA-Python>. 

The orginal work is written by Krister Svanberg in MATLAB. This is the Python implementation 
of the code written by Arjen Deetman.

This script solves the "toy problem", formulated as follows:
    
    Minimize:
         x(1)^2 + x(2)^2 + x(3)^2
    
    Subject to: 
        (x(1)-5)^2 + (x(2)-2)^2 + (x(3)-1)^2 <= 9
        (x(1)-3)^2 + (x(2)-4)^2 + (x(3)-3)^2 <= 9
        0 <= x(j) <= 5, for j=1,2,3.
"""

# Loading modules
from __future__ import division
from mmapy import mmasub, kktcheck
from util import setup_logger
from typing import Tuple
import numpy as np
import os


def main() -> None:
    # Logger
    path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(path, "mma_toy2.log")
    logger = setup_logger(file)
    logger.info("Started\n")
    
    # Set numpy print options
    np.set_printoptions(precision=4, formatter={'float': '{: 0.4f}'.format})
    
    # Initial settings
    m, n = 2, 3
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))
    zerom = np.zeros((m, 1))
    xval = np.array([[4, 3, 2]]).T
    xold1 = xval.copy()
    xold2 = xval.copy()
    xmin = zeron.copy()
    xmax = 5 * eeen
    low = xmin.copy()
    upp = xmax.copy()
    move = 1.0
    c = 1000 * eeem
    d = eeem.copy()
    a0 = 1
    a = zerom.copy()
    innerit = 0
    outeriter = 0
    maxoutit = 11
    kkttol = 0
    
    # Calculate function values and gradients of the objective and constraints functions
    if outeriter == 0:
        f0val, df0dx, fval, dfdx = toy2(xval)
        outvector1 = np.concatenate((np.array([outeriter, innerit, f0val]), fval.flatten()))
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
        
        # The MMA subproblem is solved at the point xval:
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
            m, n, outeriter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d, move)
        
        # Some vectors are updated:
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        
        # Re-calculate function values and gradients of the objective and constraints functions
        f0val, df0dx, fval, dfdx = toy2(xval)
        
        # The residual vector of the KKT conditions is calculated
        residu, kktnorm, residumax = kktcheck(
            m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, fval, dfdx, a0, a, c, d)
        
        outvector1 = np.concatenate((np.array([outeriter, innerit, f0val]), fval.flatten()))
        outvector2 = xval.flatten()
        
        # Log
        logger.info("outvector1 = {}".format(outvector1))
        logger.info("outvector2 = {}".format(outvector2))
        logger.info("kktnorm    = {}\n".format(kktnorm))
    
    # Final log
    logger.info("Finished")


def toy2(xval: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    f0val = xval[0][0]**2 + xval[1][0]**2 + xval[2][0]**2
    df0dx = 2 * xval
    fval1 = ((xval.T - np.array([[5, 2, 1]]))**2).sum() - 9
    fval2 = ((xval.T - np.array([[3, 4, 3]]))**2).sum() - 9
    fval = np.array([[fval1, fval2]]).T
    dfdx1 = 2 * (xval.T - np.array([[5, 2, 1]]))
    dfdx2 = 2 * (xval.T - np.array([[3, 4, 3]]))
    dfdx = np.concatenate((dfdx1, dfdx2))
    return f0val, df0dx, fval, dfdx


if __name__ == "__main__":
    main()