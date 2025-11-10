# %%
"""
Created on July 12 2021
Caplets and floorlets under the Hull-White Model

This code is purely educational and comes from "Financial Engineering" course by L.A. Grzelak
The course is based on the book “Mathematical Modeling and Computation
in Finance: With Exercises and Python and MATLAB Computer Codes”,
by C.W. Oosterlee and L.A. Grzelak, World Scientific Publishing Europe Ltd, 2019.
@author: Lech A. Grzelak
"""

import numpy as np
import enum
import scipy.stats as st
import scipy.integrate as integrate


# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0


def HW_theta(lambd, eta, P0T):
    """@author: Lech A. Grzelak"""
    dt = 0.0001
    f0T = lambda t: -(np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)
    theta = (
        lambda t: 1.0 / lambd * (f0T(t + dt) - f0T(t - dt)) / (2.0 * dt)
        + f0T(t)
        + eta * eta / (2.0 * lambd * lambd) * (1.0 - np.exp(-2.0 * lambd * t))
    )
    # print("CHANGED THETA")
    return theta  # lambda t: 0.1+t-t


def HW_A(lambd, eta, P0T, T1, T2):
    """@author: Lech A. Grzelak"""
    tau = T2 - T1
    zGrid = np.linspace(0.0, tau, 250)
    B_r = lambda tau: 1.0 / lambd * (np.exp(-lambd * tau) - 1.0)
    theta = HW_theta(lambd, eta, P0T)
    temp1 = lambd * integrate.trapezoid(theta(T2 - zGrid) * B_r(zGrid), zGrid)

    temp2 = eta * eta / (4.0 * np.power(lambd, 3.0)) * (
        np.exp(-2.0 * lambd * tau) * (4 * np.exp(lambd * tau) - 1.0) - 3.0
    ) + eta * eta * tau / (2.0 * lambd * lambd)

    return temp1 + temp2


def HW_B(lambd, eta, T1, T2):
    """@author: Lech A. Grzelak"""
    return 1.0 / lambd * (np.exp(-lambd * (T2 - T1)) - 1.0)


def HW_Mu_FrwdMeasure(P0T, lambd, eta, T):
    """@author: Lech A. Grzelak"""
    # time-step needed for differentiation
    dt = 0.0001
    f0T = lambda t: -(np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = HW_theta(lambd, eta, P0T)
    zGrid = np.linspace(0.0, T, 500)

    theta_hat = lambda t, T: theta(t) + eta * eta / lambd * 1.0 / lambd * (
        np.exp(-lambd * (T - t)) - 1.0
    )

    temp = lambda z: theta_hat(z, T) * np.exp(-lambd * (T - z))

    r_mean = r0 * np.exp(-lambd * T) + lambd * integrate.trapezoid(temp(zGrid), zGrid)

    return r_mean


def HWVar_r(lambd, eta, T):
    """@author: Lech A. Grzelak"""
    return eta * eta / (2.0 * lambd) * (1.0 - np.exp(-2.0 * lambd * T))


def HW_ZCB_CallPutPrice(CP, K, lambd, eta, P0T, T1, T2):
    """@author: Lech A. Grzelak"""
    B_r = HW_B(lambd, eta, T1, T2)
    A_r = HW_A(lambd, eta, P0T, T1, T2)

    mu_r = HW_Mu_FrwdMeasure(P0T, lambd, eta, T1)
    v_r = np.sqrt(HWVar_r(lambd, eta, T1))

    K_hat = K * np.exp(-A_r)

    a = (np.log(K_hat) - B_r * mu_r) / (B_r * v_r)

    d1 = a - B_r * v_r
    d2 = d1 + B_r * v_r

    term1 = np.exp(0.5 * B_r * B_r * v_r * v_r + B_r * mu_r) * st.norm.cdf(
        d1
    ) - K_hat * st.norm.cdf(d2)
    value = P0T(T1) * np.exp(A_r) * term1

    if CP == OptionType.CALL:
        return value
    else:
        return value - P0T(T2) + K * P0T(T1)


class OptionTypeSwap(enum.Enum):
    """@author: Lech A. Grzelak"""

    RECEIVER = 1.0
    PAYER = -1.0


def HW_SwapPrice(CP, notional, K, t, Ti, Tm, n, r_t, P0T, lambd, eta):
    """@author: Lech A. Grzelak"""
    # CP- payer of receiver
    # n- notional
    # K- strike
    # t- today's date
    # Ti- beginning of the swap
    # Tm- end of Swap
    # n- number of dates payments between Ti and Tm
    # r_t -interest rate at time t
    if n == 1:
        ti_grid = np.array([Ti, Tm])
    else:
        ti_grid = np.linspace(Ti, Tm, n)
    tau = ti_grid[1] - ti_grid[0]

    # overwrite Ti if t>Ti
    prevTi = ti_grid[np.where(ti_grid < t)]
    if np.size(prevTi) > 0:  # prevTi != []:
        Ti = prevTi[-1]

    # Now we need to handle the case when some payments are already done
    ti_grid = ti_grid[np.where(ti_grid > t)]

    temp = np.zeros(np.size(r_t))
    P_t_TiLambda = lambda Ti: HW_ZCB(lambd, eta, P0T, t, Ti, r_t)

    for idx, ti in enumerate(ti_grid):
        if ti > Ti:
            temp = temp + tau * P_t_TiLambda(ti)

    P_t_Ti = P_t_TiLambda(Ti)
    P_t_Tm = P_t_TiLambda(Tm)

    if CP == OptionTypeSwap.PAYER:
        swap = (P_t_Ti - P_t_Tm) - K * temp
    else:
        swap = K * temp - (P_t_Ti - P_t_Tm)

    return swap * notional


def GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T, lambd, eta, time_points=None):
    """
    Generate paths for Hull-White model using Euler scheme.

    Parameters:
    -----------
    NoOfPaths : int
        Number of Monte Carlo paths
    NoOfSteps : int
        Number of time steps (only used if time_points is None)
    T : float
        Final time (only used if time_points is None)
    P0T : function
        Zero-coupon bond price function
    lambd : float
        Mean reversion speed
    eta : float
        Volatility
    time_points : array-like, optional
        Custom time points for simulation (e.g., [0, 10/365, 0.5, 2.0])
        If provided, NoOfSteps and T are ignored

    Returns:
    --------
    dict with keys:
        'time': array of time points
        'R': array of interest rate paths (NoOfPaths x len(time))
    """
    from scipy.interpolate import CubicSpline

    # Create a smoothed yield curve for theta calculation to avoid kink issues
    # Sample the original curve at key points
    T_sample = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
    P_sample = np.array([float(P0T(t)) for t in T_sample])

    # Create cubic spline interpolation for smooth curve
    cs = CubicSpline(T_sample, P_sample, bc_type="natural")
    P0T_smooth = lambda t: cs(t)

    # time-step needed for differentiation
    dt_diff = 0.0001

    f0T = lambda t: -(
        np.log(P0T_smooth(t + dt_diff)) - np.log(P0T_smooth(t - dt_diff))
    ) / (2 * dt_diff)

    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = (
        lambda t: 1.0 / lambd * (f0T(t + dt_diff) - f0T(t - dt_diff)) / (2.0 * dt_diff)
        + f0T(t)
        + eta * eta / (2.0 * lambd * lambd) * (1.0 - np.exp(-2.0 * lambd * t))
    )

    # theta = lambda t: 0.1 +t -t
    # print("changed theta")

    # Set up time grid
    if time_points is not None:
        time = np.array(time_points)
        NoOfSteps = len(time) - 1
    else:
        time = np.linspace(0, T, NoOfSteps + 1)

    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    R = np.zeros([NoOfPaths, NoOfSteps + 1])
    R[:, 0] = r0

    for i in range(0, NoOfSteps):
        # Calculate dt for this specific step (supports non-uniform steps)
        dt = time[i + 1] - time[i]

        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        R[:, i + 1] = (
            R[:, i]
            + lambd * (theta(time[i]) - R[:, i]) * dt
            + eta * (W[:, i + 1] - W[:, i])
        )

    # Outputs
    paths = {"time": time, "R": R}
    return paths


def HW_ZCB(lambd, eta, P0T, T1, T2, rT1):
    """@author: Lech A. Grzelak"""
    B_r = HW_B(lambd, eta, T1, T2)
    A_r = HW_A(lambd, eta, P0T, T1, T2)
    return np.exp(A_r + B_r * rT1)


def HW_r_0(P0T, lambd, eta):
    """@author: Lech A. Grzelak"""
    # time-step needed for differentiation
    dt = 0.0001
    f0T = lambda t: -(np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    return r0
