"""
GCMMA-MMA-Python

This file is part of GCMMA-MMA-Python. GCMMA-MMA-Python is licensed under the terms of GNU 
General Public License as published by the Free Software Foundation. For more information and 
the LICENSE file, see <https://github.com/arjendeetman/GCMMA-MMA-Python>. 

The orginal work is written by Krister Svanberg in MATLAB. This is the Python implementation 
of the code written by Arjen Deetman.

Functionality:
- `mmasub`: Solves the MMA subproblem.
- `gcmmasub`: Solves the GCMMA subproblem.
- `subsolv`: Performs a primal-dual Newton method to solve subproblems.
- `kktcheck`: Checks the Karush-Kuhn-Tucker (KKT) conditions for the solution.

Dependencies:
- numpy: Numerical operations and array handling.
- scipy: Sparse matrix operations and linear algebra.

To use this module, import the desired functions and provide the necessary arguments 
according to the specific problem being solved.
"""

# Loading modules
from __future__ import division
from scipy.sparse import diags # or use numpy: from numpy import diag as diags
from scipy.linalg import solve # or use numpy: from numpy.linalg import solve
from typing import Tuple
import numpy as np

def mmasub(m: int, n: int, iter: int, xval: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
           xold1: np.ndarray, xold2: np.ndarray, f0val: float,  df0dx: np.ndarray, fval: np.ndarray,
           dfdx: np.ndarray, low: np.ndarray, upp: np.ndarray, a0: float, a: np.ndarray, c: np.ndarray,
           d: np.ndarray, move: float = 0.5, asyinit: float = 0.5, asydecr: float = 0.7, asyincr: float = 1.2, 
           asymin: float = 0.01, asymax: float = 10, raa0: float = 0.00001, 
           albefa: float = 0.1) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, 
                                         np.ndarray, float, np.ndarray, np.ndarray]:

    """
    Solve the MMA (Method of Moving Asymptotes) subproblem for optimization.

    Minimize:
        f_0(x) + a_0 * z + sum(c_i * y_i + 0.5 * d_i * (y_i)^2)

    Subject to:
        f_i(x) - a_i * z - y_i <= 0,    i = 1,...,m
        xmin_j <= x_j <= xmax_j,        j = 1,...,n
        z >= 0, y_i >= 0,               i = 1,...,m

    Args:
        m (int): Number of constraints.
        n (int): Number of variables.
        iter (int): Current iteration number (1 for the first call to mmasub).
        xval (np.ndarray): Current values of the design variables.
        xmin (np.ndarray): Lower bounds for the design variables.
        xmax (np.ndarray): Upper bounds for the design variables.
        xold1 (np.ndarray): Design variables from one iteration ago (provided that iter > 1).
        xold2 (np.ndarray): Design variables from two iterations ago (provided that iter > 2).
        f0val (float): Objective function value at xval.
        df0dx (np.ndarray): Gradient of the objective function at xval.
        fval (np.ndarray): Constraint function values at xval.
        dfdx (np.ndarray): Gradient of the constraint functions at xval.
        low (np.ndarray): Lower bounds for the variables from the previous iteration (provided that iter > 1).
        upp (np.ndarray): Upper bounds for the variables from the previous iteration (provided that iter > 1).
        a0 (float): Constant in the term a_0 * z.
        a (np.ndarray): Coefficients for the term a_i * z.
        c (np.ndarray): Coefficients for the term c_i * y_i.
        d (np.ndarray): Coefficients for the term 0.5 * d_i * (y_i)^2.
        move (float): Move limit for the design variables. The default is 0.5.
        asyinit (float): Factor to calculate the initial distance of the asymptotes. The default value is 0.5.
        asydecr (float): Factor by which the asymptotes distance is decreased. The default value is 0.7.
        asyincr (float): Factor by which the asymptotes distance is increased. The default value is 1.2.
        asymin (float): Factor to calculate the minimum distance of the asymptotes. The default value is 0.01.
        asymax (float): Factor to calculate the maximum distance of the asymptotes. The default value is 10. 
        raa0 (float): Parameter representing the function approximation's accuracy. The default value is 0.00001.
        albefa (float): Factor to calculate the bounds alfa and beta. The default value is 0.1. 

    Returns:
        Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
            - xmma (np.ndarray): Optimal values of the design variables.
            - ymma (np.ndarray): Optimal values of the slack variables for constraints.
            - zmma (float): Optimal value of the regularization variable z.
            - lam (np.ndarray): Lagrange multipliers for the constraints.
            - xsi (np.ndarray): Lagrange multipliers for the lower bounds on design variables.
            - eta (np.ndarray): Lagrange multipliers for the upper bounds on design variables.
            - mu (np.ndarray): Lagrange multipliers for the slack variables of the constraints.
            - zet (float): Lagrange multiplier for the regularization term z.
            - s (np.ndarray): Slack variables for the general constraints.
            - low (np.ndarray): Updated lower bounds for the design variables.
            - upp (np.ndarray): Updated upper bounds for the design variables.
    """
    
    epsimin = 0.0000001
    eeen = np.ones((n, 1), dtype=float)
    eeem = np.ones((m, 1), dtype=float)
    zeron = np.zeros((n, 1), dtype=float)
    
    # Calculation of the asymptotes low and upp
    if iter <= 2:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - asymax * (xmax - xmin)
        lowmax = xval - asymin * (xmax - xmin)
        uppmin = xval + asymin * (xmax - xmin)
        uppmax = xval + asymax * (xmax - xmin)
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)

    # Calculation of the bounds alfa and beta
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = np.maximum(zzz1, zzz2)
    alfa = np.maximum(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = np.minimum(zzz1, zzz2)
    beta = np.minimum(zzz, xmax)

    # Calculations of p0, q0, P, Q and b
    xmami = xmax - xmin
    xmami_eps = 0.00001 * eeen
    xmami = np.maximum(xmami, xmami_eps)
    xmami_inv = eeen / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    ux_inv = eeen / ux1
    xl_inv = eeen / xl1
    p0 = zeron.copy()
    q0 = zeron.copy()
    p0 = np.maximum(df0dx, 0)
    q0 = np.maximum(-df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmami_inv
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0 * ux2
    q0 = q0 * xl2
    P = np.zeros((m, n), dtype=float)
    Q = np.zeros((m, n), dtype=float)
    P = np.maximum(dfdx, 0)
    Q = np.maximum(-dfdx, 0)
    PQ = 0.001 * (P + Q) + raa0 * np.dot(eeem, xmami_inv.T)
    P = P + PQ
    Q = Q + PQ
    P = (diags(ux2.flatten(), 0).dot(P.T)).T
    Q = (diags(xl2.flatten(), 0).dot(Q.T)).T
    b = np.dot(P, ux_inv) + np.dot(Q, xl_inv) - fval

    # Solving the subproblem using the primal-dual Newton method
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)
    
    # Return values
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp

def gcmmasub(m: int, n: int, iter: int, epsimin: float, xval: np.ndarray, xmin: np.ndarray, 
             xmax: np.ndarray, low: np.ndarray, upp: np.ndarray, raa0: float, raa: np.ndarray, 
             f0val: np.ndarray, df0dx: np.ndarray, fval: np.ndarray, dfdx: np.ndarray, a0: float, 
             a: np.ndarray, c: np.ndarray, d: np.ndarray, albefa: float = 0.1) -> Tuple[np.ndarray, np.ndarray, 
            float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    
    """
    Solve the GCMMA (Generalized Convex Method of Moving Asymptotes) subproblem for optimization.

    Minimize:
        r0 + SUM[p0_j / (upp_j - x_j) + q0_j / (x_j - low_j)]
    
    Subject to:
        r = fval - SUM[P_ij / (upp_j - x_j) + Q_ij / (x_j - low_j)]
        f0app = r0 + SUM[p0_j / (upp_j - x_j) + q0_j / (x_j - low_j)]
        where
        p0, q0, r0, P, Q, r, b are calculated based on provided inputs.

    Args:
        m (int): Number of constraints.
        n (int): Number of variables.
        iter (int): Iteration number (not used in the function).
        epsimin (float): A small positive number for numerical stability.
        xval (np.ndarray): Current values of the design variables.
        xmin (np.ndarray): Minimum bounds for the design variables.
        xmax (np.ndarray): Maximum bounds for the design variables.
        low (np.ndarray): Lower bounds for the design variables in the constraints.
        upp (np.ndarray): Upper bounds for the design variables in the constraints.
        raa0 (float): Coefficient for the regularization term.
        raa (np.ndarray): Coefficients for regularization in the constraints.
        f0val (np.ndarray): Value of the objective function at the current design.
        df0dx (np.ndarray): Gradient of the objective function at the current design.
        fval (np.ndarray): Value of the constraint functions at the current design.
        dfdx (np.ndarray): Gradient of the constraint functions at the current design.
        a0 (float): Coefficient in the objective function.
        a (np.ndarray): Coefficients for the constraints.
        c (np.ndarray): Coefficients for the linear terms in the objective function.
        d (np.ndarray): Coefficients for the quadratic terms in the objective function.
        albefa (float): Factor to calculate the bounds alfa and beta. The default value is 0.1. 

    Returns:
        Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
            - xmma (np.ndarray): Optimal values of the design variables.
            - ymma (np.ndarray): Optimal values of the slack variables for constraints.
            - zmma (float): Optimal value of the regularization variable.
            - lam (np.ndarray): Lagrange multipliers for the constraints.
            - xsi (np.ndarray): Multipliers for the lower bounds on design variables.
            - eta (np.ndarray): Multipliers for the upper bounds on design variables.
            - mu (np.ndarray): Slack variables for the linear constraints.
            - zet (np.ndarray): Slack variables for the regularization terms.
            - s (np.ndarray): Additional variables in the subproblem.
            - f0app (float): Approximation of the objective function value.
            - fapp (float): Approximation of the constraint function values.
    """
    
    eeen = np.ones((n, 1))
    zeron = np.zeros((n, 1))

    # Calculations of the bounds alfa and beta
    zzz = low + albefa * (xval - low)
    alfa = np.maximum(zzz, xmin)
    zzz = upp - albefa * (upp - xval)
    beta = np.minimum(zzz, xmax)

    # Calculations of p0, q0, r0, P, Q, r and b.
    xmami = xmax - xmin
    xmami_eps = 0.00001 * eeen
    xmami = np.maximum(xmami, xmami_eps)
    xmami_inv = eeen / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    ux_inv = eeen / ux1
    xl_inv = eeen / xl1

    # Initializations for p0, q0
    p0 = zeron.copy()
    q0 = zeron.copy()
    p0 = np.maximum(df0dx, 0)
    q0 = np.maximum(-df0dx, 0)
    pq0 = p0 + q0
    p0 += 0.001 * pq0
    q0 += 0.001 * pq0
    p0 += raa0 * xmami_inv
    q0 += raa0 * xmami_inv
    p0 *= ux2
    q0 *= xl2
    r0 = f0val - np.dot(p0.T, ux_inv) - np.dot(q0.T, xl_inv)
    
    P = np.zeros((m, n)) # To be made sparse
    Q = np.zeros((m, n)) # To be made sparse
    P = (diags(ux2.flatten(), 0).dot(P.T)).T
    Q = (diags(xl2.flatten(), 0).dot(Q.T)).T
    b = np.dot(P, ux_inv) + np.dot(Q, xl_inv) - fval
    P = np.maximum(dfdx, 0)
    Q = np.maximum(-dfdx, 0)
    PQ = P + Q
    P += 0.001 * PQ
    Q += 0.001 * PQ
    P += np.dot(raa, xmami_inv.T)
    Q += np.dot(raa, xmami_inv.T)
    P = (diags(ux2.flatten(), 0).dot(P.T)).T
    Q = (diags(xl2.flatten(), 0).dot(Q.T)).T
    r = fval - np.dot(P, ux_inv) - np.dot(Q, xl_inv)
    b = -r

    # Solving the subproblem using the primal-dual Newton method
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = subsolv(m, n, epsimin, low, upp, alfa, beta, p0, q0, P, Q, a0, a, b, c, d)

    # Calculations of f0app and fapp
    ux1 = upp - xmma
    xl1 = xmma - low
    ux_inv = eeen / ux1
    xl_inv = eeen / xl1
    f0app = r0 + np.dot(p0.T, ux_inv) + np.dot(q0.T, xl_inv)
    fapp = r + np.dot(P, ux_inv) + np.dot(Q, xl_inv)

    # Return values
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp

def subsolv(m: int, n: int, epsimin: float, low: np.ndarray, upp: np.ndarray, alfa: np.ndarray, 
            beta: np.ndarray, p0: np.ndarray, q0: np.ndarray, P: np.ndarray, Q: np.ndarray, 
            a0: float, a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, 
            np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    
    """
    Solve the MMA (Method of Moving Asymptotes) subproblem for optimization.

    Minimize:
        SUM[p0j/(uppj-xj) + q0j/(xj-lowj)] + a0*z + SUM[ci*yi + 0.5*di*(yi)^2]
    
    Subject to:
        SUM[pij/(uppj-xj) + qij/(xj-lowj)] - ai*z - yi <= bi,
        alfa_j <= xj <= beta_j, yi >= 0, z >= 0.

    Args:
        m (int): Number of constraints.
        n (int): Number of variables.
        epsimin (float): A small positive number to ensure numerical stability.
        low (np.ndarray): Lower bounds for the variables x_j.
        upp (np.ndarray): Upper bounds for the variables x_j.
        alfa (np.ndarray): Lower asymptotes for the variables.
        beta (np.ndarray): Upper asymptotes for the variables.
        p0 (np.ndarray): Coefficients for the lower bound terms.
        q0 (np.ndarray): Coefficients for the upper bound terms.
        P (np.ndarray): Matrix of coefficients for the lower bound terms in the constraints.
        Q (np.ndarray): Matrix of coefficients for the upper bound terms in the constraints.
        a0 (float): Constant term in the objective function.
        a (np.ndarray): Coefficients for the constraints involving z.
        b (np.ndarray): Right-hand side constants in the constraints.
        c (np.ndarray): Coefficients for the terms involving y in the constraints.
        d (np.ndarray): Coefficients for the quadratic terms involving y in the objective function.

    Returns:
        Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
            - xmma (np.ndarray): Optimal values of the variables x_j.
            - ymma (np.ndarray): Optimal values of the variables y_i.
            - zmma (float): Optimal value of the variable z.
            - slack (np.ndarray): Slack variables for the general MMA constraints.
            - lagrange (np.ndarray): Lagrange multipliers for the constraints.
    """

    een = np.ones((n, 1))
    eem = np.ones((m, 1))
    epsi = 1
    epsvecn = epsi * een
    epsvecm = epsi * eem
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = np.array([[1.0]])
    lam = eem.copy()
    xsi = een / (x - alfa)
    xsi = np.maximum(xsi, een)
    eta = een / (beta - x)
    eta = np.maximum(eta, een)
    mu = np.maximum(eem, 0.5 * c)
    zet = np.array([[1.0]])
    s = eem.copy()
    itera = 0

    # Start while loop for numerical stability
    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1
        plam = p0 + np.dot(P.T, lam)
        qlam = q0 + np.dot(Q.T, lam)
        gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)
        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - np.dot(a.T, lam)
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm
        residu1 = np.concatenate((rex, rey, rez), axis=0)
        residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
        residu = np.concatenate((residu1, residu2), axis=0)
        residunorm = np.sqrt(np.dot(residu.T, residu).item())
        residumax = np.max(np.abs(residu))
        ittt = 0

        # Start inner while loop for optimization
        while (residumax > 0.9 * epsi) and (ittt < 200):
            ittt += 1
            itera += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2
            plam = p0 + np.dot(P.T, lam)
            qlam = q0 + np.dot(Q.T, lam)
            gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)
            GG = (diags(uxinv2.flatten(), 0).dot(P.T)).T - (diags(xlinv2.flatten(), 0).dot(Q.T)).T
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - np.dot(a.T, lam) - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv

            # Solve system of equations
            if m < n:
                blam = dellam + dely / diagy - np.dot(GG, (delx / diagx))
                bb = np.concatenate((blam, delz), axis=0)
                Alam = np.asarray(diags(diaglamyi.flatten(), 0) +
                                  (diags(diagxinv.flatten(), 0).dot(GG.T).T).dot(GG.T))
                AAr1 = np.concatenate((Alam, a), axis=1)
                AAr2 = np.concatenate((a, -zet / z), axis=0).T
                AA = np.concatenate((AAr1, AAr2), axis=0)
                solut = solve(AA, bb)
                dlam = solut[0:m]
                dz = solut[m:m + 1]
                dx = -delx / diagx - np.dot(GG.T, dlam) / diagx
            else:
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy
                Axx = np.asarray(diags(diagx.flatten(), 0) +
                                 (diags(diaglamyiinv.flatten(), 0).dot(GG).T).dot(GG))
                azz = zet / z + np.dot(a.T, (a / diaglamyi))
                axz = np.dot(-GG.T, (a / diaglamyi))
                bx = delx + np.dot(GG.T, (dellamyi / diaglamyi))
                bz = delz - np.dot(a.T, (dellamyi / diaglamyi))
                AAr1 = np.concatenate((Axx, axz), axis=1)
                AAr2 = np.concatenate((axz.T, azz), axis=1)
                AA = np.concatenate((AAr1, AAr2), axis=0)
                bb = np.concatenate((-bx, -bz), axis=0)
                solut = solve(AA, bb)
                dx = solut[0:n]
                dz = solut[n:n + 1]
                dlam = np.dot(GG, dx) / diaglamyi - dz * (a / diaglamyi) + dellamyi / diaglamyi

            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam
            xx = np.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)
            dxx = np.concatenate((dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0)

            # Step length determination
            stepxx = -1.01 * dxx / xx
            stmxx = np.max(stepxx)
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = np.max(stepbeta)
            stmalbe = np.maximum(stmalfa, stmbeta)
            stmalbexx = np.maximum(stmalbe, stmxx)
            stminv = np.maximum(stmalbexx, 1.0)
            steg = 1.0 / stminv

            # Update variables
            xold = x.copy()
            yold = y.copy()
            zold = z.copy()
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet.copy()
            sold = s.copy()
            
            itto = 0
            resinew = 2 * residunorm

            while (resinew > residunorm) and (itto < 50):
                itto += 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = een / ux1
                xlinv1 = een / xl1
                plam = p0 + np.dot(P.T, lam)
                qlam = q0 + np.dot(Q.T, lam)
                gvec = np.dot(P, uxinv1) + np.dot(Q, xlinv1)
                dpsidx = plam / ux2 - qlam / xl2
                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - np.dot(a.T, lam)
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = zet * z - epsi
                res = lam * s - epsvecm
                residu1 = np.concatenate((rex, rey, rez), axis=0)
                residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
                residu = np.concatenate((residu1, residu2), axis=0)
                resinew = np.sqrt(np.dot(residu.T, residu))
                steg = steg / 2
            residunorm = resinew.copy()
            residumax = np.max(np.abs(residu))
            steg = 2 * steg

        epsi = 0.1 * epsi

    xmma = x.copy()
    ymma = y.copy()
    zmma = z.copy()
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s

    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma

def kktcheck(m: int, n: int, x: np.ndarray, y: np.ndarray, z: float, lam: np.ndarray, xsi: np.ndarray,
            eta: np.ndarray, mu: np.ndarray, zet: float, s: np.ndarray, xmin: np.ndarray, xmax: np.ndarray,
            df0dx: np.ndarray, fval: np.ndarray, dfdx: np.ndarray, a0: float, a: np.ndarray, c: np.ndarray,
            d: np.ndarray) -> Tuple[np.ndarray, float, float]:
    
    """
    Evaluate the residuals for the Karush-Kuhn-Tucker (KKT) conditions of a nonlinear programming problem.

    The KKT conditions are necessary for optimality in constrained optimization problems. This function computes 
    the residuals for these conditions based on the current values of the variables, constraints, and Lagrange multipliers.

    Args:
        m (int): Number of general constraints.
        n (int): Number of variables.
        x (np.ndarray): Current values of the variables.
        y (np.ndarray): Current values of the general constraints' slack variables.
        z (float): Current value of the single variable in the problem.
        lam (np.ndarray): Lagrange multipliers for the general constraints.
        xsi (np.ndarray): Lagrange multipliers for the lower bound constraints on variables.
        eta (np.ndarray): Lagrange multipliers for the upper bound constraints on variables.
        mu (np.ndarray): Lagrange multipliers for the non-negativity constraints on slack variables.
        zet (float): Lagrange multiplier for the non-negativity constraint on z.
        s (np.ndarray): Slack variables for the general constraints.
        xmin (np.ndarray): Lower bounds for the variables.
        xmax (np.ndarray): Upper bounds for the variables.
        df0dx (np.ndarray): Gradient of the objective function with respect to the variables.
        fval (np.ndarray): Values of the constraint functions.
        dfdx (np.ndarray): Jacobian matrix of the constraint functions.
        a0 (float): Coefficient for the term involving z in the objective function.
        a (np.ndarray): Coefficients for the terms involving z in the constraints.
        c (np.ndarray): Coefficients for the terms involving y in the constraints.
        d (np.ndarray): Coefficients for the quadratic terms involving y in the objective function.

    Returns:
        Tuple[np.ndarray, float, float]:
            - residu (np.ndarray): Residual vector for the KKT conditions.
            - residunorm (float): Norm of the residual vector.
            - residumax (float): Maximum absolute value among the residuals.
    """

    # Compute residuals for the KKT conditions
    rex = df0dx + np.dot(dfdx.T, lam) - xsi + eta
    rey = c + d * y - mu - lam
    rez = a0 - zet - np.dot(a.T, lam)
    relam = fval - a * z - y + s
    rexsi = xsi * (x - xmin)
    reeta = eta * (xmax - x)
    remu = mu * y
    rezet = zet * z
    res = lam * s
    
    # Concatenate residuals into a single vector
    residu1 = np.concatenate((rex, rey, rez), axis=0)
    residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
    residu = np.concatenate((residu1, residu2), axis=0)
    
    # Calculate norm and maximum value of the residual vector
    residunorm = np.sqrt(np.dot(residu.T, residu).item())
    residumax = np.max(np.abs(residu))
    
    return residu, residunorm, residumax

def raaupdate(xmma: np.ndarray, xval: np.ndarray, xmin: np.ndarray, xmax: np.ndarray, low: np.ndarray, upp: np.ndarray, 
              f0valnew: np.ndarray, fvalnew: np.ndarray, f0app: np.ndarray, fapp: np.ndarray, raa0: np.ndarray, 
              raa: np.ndarray, raa0eps: np.ndarray, raaeps: np.ndarray,  epsimin: float) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Update the parameters raa0 and raa during an inner iteration.

    This function adjusts the values of raa0 and raa based on the current function values and 
    approximations. The updated values help in improving the accuracy of the function approximations.

    Parameters:
        xmma (np.ndarray): Current values of the optimization variables.
        xval (np.ndarray): Current values of the variables for comparison.
        xmin (np.ndarray): Lower bounds of the optimization variables.
        xmax (np.ndarray): Upper bounds of the optimization variables.
        low (np.ndarray): Lower bounds for the interval of the variables.
        upp (np.ndarray): Upper bounds for the interval of the variables.
        f0valnew (np.ndarray): New function values at the initial point.
        fvalnew (np.ndarray): New function values at the subsequent points.
        f0app (np.ndarray): Current approximation of the function at the initial point.
        fapp (np.ndarray): Current approximation of the function at subsequent points.
        raa0 (np.ndarray): Parameter representing the initial function approximation's accuracy.
        raa (np.ndarray): Parameter representing the subsequent function approximation's accuracy.
        raa0eps (np.ndarray): Minimum value for raa0.
        raaeps (np.ndarray): Minimum value for raa.
        epsimin (float): Small perturbation added to the function approximations.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - raa0 (np.ndarray): Updated values of the raa0 variable.
            - raa (np.ndarray): Updated values of the raa variable.
    """
    
    raacofmin = 1e-12
    eeem = np.ones((raa.size, 1))
    eeen = np.ones((xmma.size, 1))
    xmami = xmax - xmin
    xmamieps = 0.00001 * eeen
    xmami = np.maximum(xmami, xmamieps)
    xxux = (xmma - xval) / (upp - xmma)
    xxxl = (xmma - xval) / (xmma - low)
    xxul = xxux * xxxl
    ulxx = (upp - low) / xmami
    raacof = np.dot(xxul.T, ulxx)
    raacof = np.maximum(raacof, raacofmin)
    f0appe = f0app + 0.5 * epsimin

    if np.all(f0valnew > f0appe):
        deltaraa0 = (1.0 / raacof) * (f0valnew - f0app)
        zz0 = 1.1 * (raa0 + deltaraa0)
        zz0 = np.minimum(zz0, 10 * raa0)
        raa0 = zz0
    
    fappe = fapp + 0.5 * epsimin * eeem
    fdelta = fvalnew - fappe
    deltaraa = (1 / raacof) * (fvalnew - fapp)
    zzz = 1.1 * (raa + deltaraa)
    zzz = np.minimum(zzz, 10 * raa)
    raa[np.where(fdelta > 0)] = zzz[np.where(fdelta > 0)]
    
    return raa0, raa


def concheck(m: int, epsimin: np.ndarray, f0app: np.ndarray, f0valnew: np.ndarray, fapp: np.ndarray, fvalnew: np.ndarray) -> int:
    
    """
    Check if the current approximations are conservative.

    This function evaluates if the current approximations meet the conservativeness criterion.
    It compares the updated approximations with the new values and sets the `conserv` parameter
    to 1 if the approximations are conservative and 0 otherwise.

    Parameters:
        m (int)  : The dimension of the approximation vectors.
        epsimin (np.ndarray): Small perturbation added to the approximations.
        f0app (np.ndarray): Current approximation of the function at the initial point.
        f0valnew (np.ndarray): New function values at the initial point.
        fapp (np.ndarray): Current approximation of the function at subsequent points.
        fvalnew (np.ndarray): New function values at subsequent points.

    Returns:
        int: A flag indicating if the approximations are conservative (1) or not (0).
    """
    
    eeem = np.ones((m, 1))
    f0appe = f0app + epsimin
    fappe = fapp + epsimin * eeem
    arr1 = np.concatenate((f0appe.flatten(), fappe.flatten()))
    arr2 = np.concatenate((f0valnew.flatten(), fvalnew.flatten()))
    
    if np.all(arr1 >= arr2):
        conserv = 1
    else:
        conserv = 0
    
    return conserv


def asymp(outeriter: int, n: int,xval: np.ndarray, xold1: np.ndarray, xold2: np.ndarray, xmin: np.ndarray,
    xmax: np.ndarray, low: np.ndarray, upp: np.ndarray, raa0: float, raa: np.ndarray, raa0eps: float,
    raaeps: float, df0dx: np.ndarray, dfdx: np.ndarray, asyinit: float = 0.5, asydecr: float = 0.7, 
    asyincr: float = 1.2, asymin: float = 0.01, asymax: float = 10) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    
    """
    Calculate the parameters raa0, raa, low, and upp at the beginning of each outer iteration.

    Parameters:
        outeriter (int): Current outer iteration number.
        n (int): Number of design variables.
        xval (np.ndarray): Current values of the design variables.
        xold1 (np.ndarray): Values of the design variables from the previous iteration.
        xold2 (np.ndarray): Values of the design variables from two iterations ago.
        xmin (np.ndarray): Lower bounds for the design variables.
        xmax (np.ndarray): Upper bounds for the design variables.
        low (np.ndarray): Lower asymptote bounds (input/output).
        upp (np.ndarray): Upper asymptote bounds (input/output).
        raa0 (float): Parameter for the objective function (output).
        raa (np.ndarray): Parameter for the constraints (output).
        raa0eps (float): Minimum value for raa0.
        raaeps (float): Minimum value for raa.
        df0dx (np.ndarray): Gradient of the objective function.
        dfdx (np.ndarray): Gradient of the constraints.
        asyinit (float): Factor to calculate the initial distance of the asymptotes. The default value is 0.5.
        asydecr (float): Factor by which the asymptotes distance is decreased. The default value is 0.7.
        asyincr (float): Factor by which the asymptotes distance is increased. The default value is 1.2.
        asymin (float): Factor to calculate the minimum asymptote distance. The default value is 0.01.
        asymax (float): Factor to calculate the maximum asymptote distance. The default value is 10. 

    Returns:
        Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
            - low (np.ndarray): Updated values for the lower asymptotes.
            - upp (np.ndarray): Updated values for the upper asymptotes.
            - raa0 (float): Updated value of the raa0 parameter.
            - raa (np.ndarray): Updated values of the raa parameter.
    """
    
    eeen = np.ones((n, 1))
    xmami = xmax - xmin
    xmamieps = 0.00001 * eeen
    xmami = np.maximum(xmami, xmamieps)
    raa0 = np.dot(np.abs(df0dx).T, xmami)
    raa0 = np.maximum(raa0eps, (0.1 / n) * raa0)
    raa = np.dot(np.abs(dfdx), xmami)
    raa = np.maximum(raaeps, (0.1 / n) * raa)
    
    if outeriter <= 2:
        low = xval - asyinit * xmami
        upp = xval + asyinit * xmami
    else:
        xxx = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[xxx > 0] = asyincr
        factor[xxx < 0] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - asymax * xmami
        lowmax = xval - asymin * xmami
        uppmin = xval + asymin * xmami
        uppmax = xval + asymax * xmami
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)
    
    return low, upp, raa0, raa