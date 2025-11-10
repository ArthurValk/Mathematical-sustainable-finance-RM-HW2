"""Yield curves for Ex. 1"""

import numpy as np


def yield_curve() -> callable:
    """Function for the yield curve function P(0,T).
    The continuously compounded zero rates for P(0,T) are:
    (T,r(0,T)) = {(0.5, 3.5%), (1.0, 3.6%), (2.0, 3.8%), (5.0, 4.0%), (10.0, 4.2%)}.

    We interpolate linearly for maturities in between these points.

    Returns
    -------
    callable
        the yield curve function P(0,T)
    """
    # Define the maturities and dates
    maturities = np.array(
        [
            0.0,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ]
    )
    rates = np.array(
        [
            0.0,
            0.035,
            0.036,
            0.038,
            0.040,
            0.042,
        ]
    )

    def P0T(T: np.ndarray | float) -> np.ndarray | float:
        """Zero coupon bond price P(0,T) = exp(-r(0,T) * T)"""
        # Handle scalar or array input
        T_array = np.atleast_1d(T)
        r_T = np.interp(T_array, maturities, rates)
        result = np.exp(-r_T * T_array)
        # Return scalar if input was scalar
        return result if np.isscalar(T) or len(result) > 1 else result[0]

    return P0T


def parallel_shift_yield_curve(yield_curve: callable) -> callable:
    """Returns the yield_curve with parallel shift +150bps"""
    return lambda T: yield_curve(T) * np.exp(-0.0150 * T)


def bull_steep_yield_curve() -> callable:
    """Returns the yield_curve with bull steepener -25bps at 2y, +50bps at 10y"""
    # Define the maturities and rates
    maturities = np.array(
        [
            0.0,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
        ]
    )
    rates = np.array(
        [
            0.0,
            0.035,
            0.036,
            0.0355,
            0.040,
            0.047,
        ]
    )

    def P0T(T: np.ndarray | float) -> np.ndarray | float:
        """Zero coupon bond price P(0,T) = exp(-r(0,T) * T)"""
        # Handle scalar or array input
        T_array = np.atleast_1d(T)
        r_T = np.interp(T_array, maturities, rates)
        result = np.exp(-r_T * T_array)
        # Return scalar if input was scalar
        return result if np.isscalar(T) or len(result) > 1 else result[0]

    return P0T


def transition_risk_yield_curve(yield_curve: callable) -> callable:
    """Returns the yield_curve with transition risk: +75bps parallel shift"""
    return lambda T: yield_curve(T) * np.exp(-0.0075 * T)
