"""
GCMMA-MMA-Python

This file is part of GCMMA-MMA-Python. GCMMA-MMA-Python is licensed under the terms of GNU 
General Public License as published by the Free Software Foundation. For more information and 
the LICENSE file, see <https://github.com/arjendeetman/GCMMA-MMA-Python>. 

The orginal work is written by Krister Svanberg in MATLAB. This is the Python implementation 
of the code written by Arjen Deetman.

This script solves the "three bar truss problem", formulated as follows:

    Minimize:
        max{f1(x), f2(x), f3(x)}

    Subject to:
        x1 + x2 + x3 <= V
        0.001 <= xj <= V, for j = 1, 2, 3

    Where:
        - xj: Volume of the j-th bar
        - fi(x): Compliance for the i-th load case, calculated as pi' * ui
        - V: Upper bound on the total volume (V = 3)

    Problem Description:
        - Bar 1 connects nodes 1 and 4.
        - Bar 2 connects nodes 2 and 4.
        - Bar 3 connects nodes 3 and 4.
        - Nodes coordinates: 
            Node 1: (-1, 0)
            Node 2: (-1/sqrt(2), -1/sqrt(2))
            Node 3: (0, -1)
            Node 4: (0, 0)
        - Nodes 1, 2, 3 are fixed.
        - Load vectors at node 4:
            Load vector p1 = (1, 0)'
            Load vector p2 = (1, 1)'
            Load vector p3 = (0, 1)'

    Displacement vector ui is obtained from the system K(x) * ui = pi, where K(x) is the stiffness matrix 
    and pi is the load vector. The stiffness matrix is given by K(x) = R * D(x) * R', where D(x) is a diagonal 
    matrix with diagonal elements x1, x2, x3. The derivatives of the functions fi(x) are given by:
    dfi/dxj = -(rj' * ui)^2, where rj is the j-th column of the matrix R.

MMA Formulation:

    Minimize:
        z + 1000 * (y1 + y2 + y3 + y4)

    Subject to:
        f1(x) - z - y1 <= 0
        f2(x) - z - y2 <= 0
        f3(x) - z - y3 <= 0
        x1 + x2 + x3 - 3 - y4 <= 0
        0 <= xj <= 3, for j = 1, 2, 3
        yi >= 0, for i = 1, 2, 3, 4
        z >= 0
"""

# Loading modules
from __future__ import division
from mma import mmasub, kktcheck
from scipy.linalg import solve # or use numpy: from numpy.linalg import solve
from util import setup_logger
from typing import Tuple
import numpy as np
import os


def main() -> None:
    # Logger setup
    path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(path, "mma_truss2.log")
    logger = setup_logger(file)
    logger.info("Started\n")
    
    # Set numpy print options
    np.set_printoptions(precision=4, formatter={'float': '{: 0.4f}'.format})
    
    # Problem dimensions and initial settings
    m, n = 4, 3
    epsimin = 1e-7
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))
    zerom = np.zeros((m, 1))
    xval = eeen.copy()
    xold1 = xval.copy()
    xold2 = xval.copy()
    xmin = 0.001 * eeen
    xmax = 3 * eeen
    low = xmin.copy()
    upp = xmax.copy()
    move = 1.0
    c = 1000 * eeem
    d = zerom.copy()
    a0 = 1
    a = np.array([[1, 1, 1, 0]]).T
    outeriter = 0
    maxoutit = 6
    kkttol = 0
    
    # Initial function evaluations
    if outeriter == 0:
        f0val, df0dx, fval, dfdx = truss2(xval)
        innerit = 0
        outvector1 = np.concatenate((np.array([outeriter]), xval.flatten()))
        outvector2 = fval.flatten()
        # Log initial values
        logger.info("outvector1 = {}".format(outvector1))
        logger.info("outvector2 = {}\n".format(outvector2))
    
    # Iterative optimization process
    kktnorm = kkttol + 10
    outit = 0
    
    while kktnorm > kkttol and outit < maxoutit:
        outit += 1
        outeriter += 1
        
        # Solve the MMA subproblem
        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = mmasub(
            m, n, outeriter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d, move)
        
        # Update vectors
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        
        # Recalculate function values and gradients
        f0val, df0dx, fval, dfdx = truss2(xval)
        
        # Calculate KKT residuals
        residu, kktnorm, residumax = kktcheck(
            m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, fval, dfdx, a0, a, c, d)
        
        outvector1 = np.concatenate((np.array([outeriter]), xval.flatten()))
        outvector2 = fval.flatten()
        
        # Log iteration results
        logger.info("outvector1 = {}".format(outvector1))
        logger.info("outvector2 = {}".format(outvector2))
        logger.info("kktnorm    = {}\n".format(kktnorm))
    
    # Final log
    logger.info("Finished")


def truss2(xval: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Calculate the objective function, constraints, and their gradients for the truss optimization problem.

    Args:
        xval (np.ndarray): Current design variables (volumes of the bars).

    Returns:
        Tuple[float, np.ndarray, np.ndarray, np.ndarray]: Function values, gradients, and other necessary values.
    """

    e = np.array([[1, 1, 1]]).T
    f0val = 0
    df0dx = 0 * e
    D = np.diag(xval.flatten())
    sq2 = 1.0 / np.sqrt(2.0)
    R = np.array([[1, sq2, 0], [0, sq2, 1]])
    p1 = np.array([[1, 0]]).T
    p2 = np.array([[1, 1]]).T
    p3 = np.array([[0, 1]]).T
    K = np.dot(R,D).dot(R.T)
    u1 = solve(K, p1)
    u2 = solve(K, p2)
    u3 = solve(K, p3)
    compl1 = np.dot(p1.T, u1)
    compl2 = np.dot(p2.T, u2)
    compl3 = np.dot(p3.T, u3)
    volume = np.dot(e.T, xval)
    V = 3.0
    vol1 = volume - V
    fval = np.concatenate((compl1, compl2, compl3, vol1))
    rtu1 = np.dot(R.T, u1)
    rtu2 = np.dot(R.T, u2)
    rtu3 = np.dot(R.T, u3)
    dcompl1 = -rtu1 * rtu1
    dcompl2 = -rtu2 * rtu2
    dcompl3 = -rtu3 * rtu3
    dfdx = np.concatenate((dcompl1.T, dcompl2.T, dcompl3.T, e.T))
    return f0val, df0dx, fval, dfdx

if __name__ == "__main__":
    main()