"""
GCMMA-MMA-Python

This file is part of GCMMA-MMA-Python. GCMMA-MMA-Python is licensed under the terms of GNU 
General Public License as published by the Free Software Foundation. For more information and 
the LICENSE file, see <https://github.com/arjendeetman/GCMMA-MMA-Python>. 

The orginal work is written by Krister Svanberg in MATLAB. This is the Python implementation 
of the code written by Arjen Deetman.

Optimization of a simple function with one design variable and no constraint functions. 

    Minimize:
        (x - 50)^2 + 25
    
    Subject to: 
        1 <= x <= 100
"""

# Loading modules
from __future__ import division
from mma import mmasub, kktcheck
from util import setup_logger
from typing import Tuple
import numpy as np
import os


def main() -> None:
    # Logger
    path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(path, "mma_function.log")
    logger = setup_logger(file)
    logger.info("Started\n")
    
    # Set numpy print options
    np.set_printoptions(precision=4, formatter={'float': '{: 0.4f}'.format})
    
    # Initial settings
    m, n = 1, 1
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))
    zerom = np.zeros((m, 1))
    xval = 1 * eeen
    xold1 = xval.copy()
    xold2 = xval.copy()
    xmin = eeen.copy()
    xmax = 100 * eeen
    low = xmin.copy()
    upp = xmax.copy()
    move = 1.0
    c = 1000 * eeem
    d = eeem.copy()
    a0 = 1
    a = zerom.copy()
    innerit = 0
    outeriter = 0
    maxoutit = 20
    kkttol = 0
    
    # Calculate function values and gradients of the objective and constraints functions
    if outeriter == 0:
        f0val, df0dx, fval, dfdx = funct(xval, n, eeen, zeron)
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
        
        # The MMA subproblem is solved at the point xval:
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
            m, n, outeriter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d, move)
        
        # Some vectors are updated:
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        
        # Re-calculate function values and gradients of the objective and constraints functions
        f0val, df0dx, fval, dfdx = funct(xval, n, eeen, zeron)
        
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


def funct(xval: np.ndarray, n: int, eeen: np.ndarray, zeron: np.ndarray) -> Tuple[float, np.ndarray, float, np.ndarray]:
    f0val = (xval.item() - 50) ** 2 + 25
    df0dx = eeen * (2 * (xval.item() - 50))
    fval = 0.0
    dfdx = zeron
    return f0val, df0dx, fval, dfdx


if __name__ == "__main__":
    main()