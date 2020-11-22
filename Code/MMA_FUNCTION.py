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
Optimization of a simple function with one design variable and no contraint functions. 

    minimize (x-50)^2+25
    subjected to 1 =< x(j) =< 100, for j=1 
"""

########################################################################################################
### LOADING MODULES                                                                                  ###
########################################################################################################

# Loading modules
from __future__ import division
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
    file = os.path.join(path, "MMA_FUNCTION.log")
    logger = setup_logger(file)
    logger.info("Started\n")
    # Set numpy print options
    np.set_printoptions(precision=4, formatter={'float': '{: 0.4f}'.format})
    # Initial settings
    m = 1
    n = 1
    epsimin = 0.0000001
    eeen = np.ones((n,1))
    eeem = np.ones((m,1))
    zeron = np.zeros((n,1))
    zerom = np.zeros((m,1))
    xval = 1*eeen
    xold1 = xval.copy()
    xold2 = xval.copy()
    xmin = eeen.copy()
    xmax = 100*eeen
    low = xmin.copy()
    upp = xmax.copy()
    move = 1.0
    c = 1000*eeem
    d = eeem.copy()
    a0 = 1
    a = zerom.copy()
    outeriter = 0
    maxoutit = 20
    kkttol = 0		
    # Calculate function values and gradients of the objective and constraints functions
    if outeriter == 0:
        f0val,df0dx,fval,dfdx = funct(xval,n,eeen,zeron)
        innerit = 0
        outvector1 = np.array([outeriter, innerit, f0val, fval])
        outvector2 = xval.flatten()
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
        xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,low,upp = \
            mmasub(m,n,outeriter,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d,move)
        # Some vectors are updated:
        xold2 = xold1.copy()
        xold1 = xval.copy()
        xval = xmma.copy()
        # Re-calculate function values and gradients of the objective and constraints functions
        f0val,df0dx,fval,dfdx = funct(xval,n,eeen,zeron)
        # The residual vector of the KKT conditions is calculated
        residu,kktnorm,residumax = \
            kktcheck(m,n,xmma,ymma,zmma,lam,xsi,eta,mu,zet,s,xmin,xmax,df0dx,fval,dfdx,a0,a,c,d)
        outvector1 = np.array([outeriter, innerit, f0val, fval])
        outvector2 = xval.flatten()
        # Log
        logger.info("outvector1 = {}".format(outvector1))
        logger.info("outvector2 = {}".format(outvector2))
        logger.info("kktnorm    = {}\n".format(kktnorm))
    # Final log
    logger.info(" Finished")


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

# Beam function
def funct(xval,n,eeen,zeron):
    f0val = (xval.item()-50)**2+25
    df0dx = eeen*(2*(xval.item()-50))
    fval = 0.0
    dfdx = zeron
    return f0val,df0dx,fval,dfdx


########################################################################################################
### RUN MAIN FUNCTION                                                                                ###
########################################################################################################

# Run main function / program
if __name__ == "__main__":
    main()