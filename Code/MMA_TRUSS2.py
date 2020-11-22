########################################################################################################
### GCMMA-MMA-Python         															             ### 
###                                                                                                  ###
### This file is part of GCMMA-MMA-Python. GCMMA-MMA-Python is licensed under the terms of GNU       ###
### General Public License as published by the Free Software Foundation. For more information and    ###
### the LICENSE file, see <https://github.com/arjendeetman/GCMMA-MMA-Python>.                        ###
###                                                                                                  ###
### The orginal work is written by Krister Svanberg in MATLAB.                                       ###
### This is the python version of the code written Arjen Deetman.                                    ###
### version 09-11-2019                                                                               ###
########################################################################################################


"""
Orginal work written by Krister Svanberg in Matlab. This is the python version of the code written
by Arjen Deetman. 

This script is the "three bar truss problem":

    mininimize max{f1(x),f2(x),f3(x)}
    subject to:
    x1 + x2 + x3 <= V,
    0.001 <= xj <= V, for j=1,2,3.

    where
    xj = volume of the j:th bar,
    fi(x) = pi'*ui = compliance for the i:th loadcase,
    V = 3 = upper bound on the total volume.

    Since the length of each bar is = 1, xj is also the cross section area of the j:the bar.
  
    Bar 1 connects the nodes 1 and 4.
    Bar 2 connects the nodes 2 and 4.
    Bar 3 connects the nodes 3 and 4.
    The coordinates of node 1 are (-1,0).
    The coordinates of node 2 are (-1/sqrt(2),-1/sqrt(2)).
    The coordinates of node 3 are (0,-1).
    The coordinates of node 4 are (0,0).
    The nodes 1,2,3 are fixed, while the load vectors for
    the different loadcases are applied at node 4.
    The load vector p1 = (1,0)' (loadcase 1).
    The load vector p2 = (1,1)' (loadcase 2).
    The load vector p3 = (0,1)' (loadcase 3).

    The displacement vector ui is obtained from the system K(x)*ui = pi, i=1,2,3, where K(x) is the 
    stiffness matrix and pi is the load vector.	The stiffness matrix is given by K(x) = R*D(x)*R',
    where D(x) is a diagonal matrix with diagonal elements x1,x2,x3. The constant matrix R is given 
    in the code below. The derivatives of the functions fi(x) are then given by
    dfi/dxj = -(rj'*ui)^2 ,	where rj is the j:th column of the matrix R.

    The problem is written on the following form required by MMA.

    minimize  z + 1000*(y1+y2+y3+y4)
    subject to the constraints:
               f1(x) - z - y1 <= 0
               f2(x) - z - y2 <= 0
               f3(x) - z - y3 <= 0
        x1 + x2 + x3 - 3 - y4 <= 0
                      0 <= xj <= 3, j=1,2,3
                           yi >= 0, i=1,2,3,4
                            z >= 0.
"""

########################################################################################################
### LOADING MODULES                                                                                  ###
########################################################################################################

# Loading modules
from __future__ import division
from scipy.linalg import solve # or use numpy: from numpy.linalg import solve
import numpy as np
import logging
import sys
import os

# Import MMA functions
from MMA import mmasub,subsolv,kktcheck


########################################################################################################
### MAIN FUNCTION                                                                                    ###
########################################################################################################

def main():
    # Logger
    path = os.path.dirname(os.path.realpath(__file__))
    file = os.path.join(path, "MMA_TRUSS2.log")
    logger = setup_logger(file)
    logger.info("Started\n")
    # Set numpy print options
    np.set_printoptions(precision=4, formatter={'float': '{: 0.4f}'.format})
    # Beam initial settings
    m = 4
    n = 3
    epsimin = 0.0000001
    eeen = np.ones((n,1))
    eeem = np.ones((m,1))
    zeron = np.zeros((n,1))
    zerom = np.zeros((m,1))
    xval = eeen.copy()
    xold1 = xval.copy()
    xold2 = xval.copy()
    xmin = 0.001*eeen
    xmax = 3*eeen
    low = xmin.copy()
    upp = xmax.copy()
    move = 1.0
    c = 1000*eeem
    d = zerom.copy()
    a0 = 1
    a = np.array([[1,1,1,0]]).T
    outeriter = 0
    maxoutit = 6
    kkttol = 0		
    # Calculate function values and gradients of the objective and constraints functions
    if outeriter == 0:
        f0val,df0dx,fval,dfdx = truss2(xval)
        innerit = 0
        outvector1 = np.concatenate((np.array([outeriter]), xval.flatten()))
        outvector2 = fval.flatten()
        # Log
        logger.info("outvector1 = {}".format(outvector1))
        logger.info("outvector2 = {}\n".format(outvector2))
    # The iterations starts
    kktnorm = kkttol+10
    outit = 0
    while (kktnorm > kkttol) and (outit < maxoutit):
        outit += 1
        outeriter += 1
        # The MMA subproblem is solved at the point xval:
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp =  \
            mmasub(m,n,outeriter,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,move)
        # Some vectors are updated:
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        # Re-calculate function values and gradients of the objective and constraints functions
        f0val,df0dx,fval,dfdx = truss2(xval)
        # The residual vector of the KKT conditions is calculated
        residu,kktnorm,residumax = \
            kktcheck(m,n,xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,xmin,xmax,df0dx,fval,dfdx,a0,a,c,d)
        outvector1 = np.concatenate((np.array([outeriter]), xval.flatten()))
        outvector2 = fval.flatten()
        # Log
        logger.info("outvector1 = {}".format(outvector1))
        logger.info("outvector2 = {}".format(outvector2))
        logger.info("kktnorm    = {}\n".format(kktnorm))
    # Final log
    logger.info("Finished")


########################################################################################################
### FUNCTIONS                                                                                        ###
########################################################################################################

# Setup logger
def setup_logger(logfile):
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # Create file handler and set level to debug
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    # Add formatter to ch and fh
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    # Add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)
    # Open logfile and reset
    with open(logfile, 'w'): pass
    # Return logger
    return logger

# Truss function
def truss2(xval):
    e = np.array([[1,1,1]]).T
    f0val = 0
    df0dx = 0*e
    D = np.diag(xval.flatten())
    sq2 = 1.0/np.sqrt(2.0)
    R = np.array([[1,sq2,0],[0,sq2,1]])
    p1 = np.array([[1,0]]).T
    p2 = np.array([[1,1]]).T
    p3 = np.array([[0,1]]).T
    K = np.dot(R,D).dot(R.T)
    u1 = solve(K,p1)
    u2 = solve(K,p2)
    u3 = solve(K,p3)
    compl1 = np.dot(p1.T,u1)
    compl2 = np.dot(p2.T,u2)
    compl3 = np.dot(p3.T,u3)
    volume = np.dot(e.T,xval)
    V = 3.0
    vol1 = volume-V
    fval = np.concatenate((compl1,compl2,compl3,vol1))
    rtu1 = np.dot(R.T,u1)
    rtu2 = np.dot(R.T,u2)
    rtu3 = np.dot(R.T,u3)
    dcompl1 = -rtu1*rtu1
    dcompl2 = -rtu2*rtu2
    dcompl3 = -rtu3*rtu3
    dfdx = np.concatenate((dcompl1.T,dcompl2.T,dcompl3.T,e.T))
    return f0val,df0dx,fval,dfdx


########################################################################################################
### RUN MAIN FUNCTION                                                                                ###
########################################################################################################

# Run main function / program
if __name__ == "__main__":
    main()