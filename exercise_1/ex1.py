"""exercise_1"""

from typing import Literal

import numpy as np

from exercise_1.HW_functions import (
    HW_SwapPrice,
    OptionTypeSwap,
    HW_r_0,
    HW_ZCB_CallPutPrice,
    OptionType,
    GeneratePathsHWEuler,
    HW_ZCB,
)
from exercise_1.yield_curves import (
    yield_curve,
    parallel_shift_yield_curve,
    bull_steep_yield_curve,
    transition_risk_yield_curve,
)

np.random.seed(1)

T = 10 / 252
a = 0.03
sigma = 0.01
scale_factor = 100e6
N = 10000

CURVE_TYPE = Literal[
    "default",
    "parallel_shift",
    "bull_steep",
    "transition_risk",
]


def libor_rate_t0(
    T_k_minus1: float,
    T_k: float,
    yield_curve: callable,
) -> float:
    """libor rate at t=0"""
    tau_k = T_k - T_k_minus1
    P0T_k_minus1 = yield_curve(T_k_minus1)
    P0T_k = yield_curve(T_k)
    return 1.0 / tau_k * (P0T_k_minus1 / P0T_k - 1.0)


def V_FRA_t0(
    T_k_minus1: float,
    T_k: float,
    K: float,
    yield_curve: callable,
) -> float:
    """FRA value at t=0"""
    L0 = libor_rate_t0(T_k_minus1, T_k, yield_curve)
    tau_k = T_k - T_k_minus1
    P0T_k = yield_curve(T_k)
    return P0T_k * tau_k * (K - L0)


def V_stepup_t0(
    yield_curve: callable,
    step_up_prob: float,
    a: float,
    sigma: float,
) -> float:
    """Step-up bond value at t=0 using swap pricing"""
    # Step-up is a payer swap (pay additional 25bps) starting at year 3.5
    # Equivalent to: p_ESG * swap with strike 25bps, Ti=3.5, Tm=10, n=14 payments
    # This represents the additional liability if ESG target is missed
    v = step_up_prob * HW_SwapPrice(
        CP=OptionTypeSwap.PAYER,
        notional=1,
        K=25e-4,
        t=0,
        Ti=3.5,
        Tm=10,
        n=14,
        r_t=HW_r_0(P0T=yield_curve, lambd=a, eta=sigma),
        P0T=yield_curve,
        lambd=a,
        eta=sigma,
    )
    return v


FRA_strike = 4e-2
FRA_T_K_minus1 = 0.5
FRA_T_K = 1

option_strike = 0.8  # Strike for call on ZCB

curve_scenarios = ["default", "parallel_shift", "bull_steep", "transition_risk"]

for curve_type in curve_scenarios:
    print(f"\n{'=' * 60}")
    print(f"Scenario: {curve_type.upper()}")
    print(f"{'=' * 60}")

    # <editor-fold desc="Task 4: Deterministic curve scenarios">
    # Set up yield curve and step-up probability
    if curve_type.lower() == "default":
        curve = yield_curve()
    elif curve_type.lower() == "parallel_shift":
        curve = parallel_shift_yield_curve(yield_curve())
    elif curve_type.lower() == "bull_steep":
        curve = bull_steep_yield_curve()
    else:
        curve = transition_risk_yield_curve(yield_curve())

    if curve_type.lower() != "transition_risk":
        step_up_prob = 40 / 100
    else:
        step_up_prob = 70 / 100
    # </editor-fold>

    # <editor-fold desc="Task 1: Compute initial portfolio value V0 and horizon values VT">
    # Analytical portfolio value at t=0
    V_t0 = (
        V_FRA_t0(
            T_k_minus1=FRA_T_K_minus1,
            T_k=FRA_T_K,
            K=FRA_strike,
            yield_curve=curve,
        )
        + HW_SwapPrice(
            CP=OptionTypeSwap.PAYER,
            notional=1,
            K=4.1 / 100,
            t=0,
            Ti=0,
            Tm=10,
            n=20,
            r_t=HW_r_0(P0T=curve, lambd=a, eta=sigma),
            P0T=curve,
            lambd=a,
            eta=sigma,
        )
        + V_stepup_t0(yield_curve=curve, step_up_prob=step_up_prob, a=a, sigma=sigma)
        + HW_ZCB_CallPutPrice(
            CP=OptionType.CALL,
            K=option_strike,
            lambd=a,
            eta=sigma,
            P0T=curve,
            T1=2,
            T2=10,
        )
    )

    # Monte Carlo simulation
    time_points = np.array([0, 10 / 252, 2])
    paths = GeneratePathsHWEuler(
        NoOfPaths=N,
        NoOfSteps=None,
        T=None,
        P0T=curve,
        lambd=a,
        eta=sigma,
        time_points=time_points,
    )

    r_paths = paths["R"]
    time_grid = paths["time"]

    t_val = 10 / 252
    r_at_10d = r_paths[:, 1]
    r_at_2y = r_paths[:, 2]

    # FRA value at t=10d
    P_10d_to_6m = HW_ZCB(lambd=a, eta=sigma, P0T=curve, T1=t_val, T2=0.5, rT1=r_at_10d)
    P_10d_to_1yr = HW_ZCB(lambd=a, eta=sigma, P0T=curve, T1=t_val, T2=1.0, rT1=r_at_10d)
    tau_k = 0.5
    L_forward = (P_10d_to_6m / P_10d_to_1yr - 1.0) / tau_k
    V_FRA_at_10d = (FRA_strike - L_forward) * tau_k * P_10d_to_1yr

    # Swap value at t=10d
    V_swap_at_10d = HW_SwapPrice(
        CP=OptionTypeSwap.PAYER,
        notional=1.0,
        K=4.1 / 100,
        t=t_val,
        Ti=t_val,
        Tm=10,
        n=20,
        r_t=r_at_10d,
        P0T=curve,
        lambd=a,
        eta=sigma,
    )

    # Bond option value at t=2y
    P_2y_to_10y = HW_ZCB(lambd=a, eta=sigma, P0T=curve, T1=2.0, T2=10.0, rT1=r_at_2y)
    V_option_at_2y = np.maximum(P_2y_to_10y - option_strike, 0.0)

    # Step-up value at t=10d
    step_up_occurs = np.random.binomial(n=1, p=step_up_prob, size=N)
    payment_dates = np.array([3.5 + i * 0.5 for i in range(14)])
    discount_factor_sum = np.zeros(N)
    for payment_date in payment_dates:
        P_10d_to_payment = HW_ZCB(
            lambd=a, eta=sigma, P0T=curve, T1=t_val, T2=payment_date, rT1=r_at_10d
        )
        discount_factor_sum += P_10d_to_payment

    V_stepup_at_10d = step_up_occurs * 25e-4 * 0.5 * discount_factor_sum

    # Portfolio value at t=10d
    P_10d_to_2y = HW_ZCB(lambd=a, eta=sigma, P0T=curve, T1=t_val, T2=2.0, rT1=r_at_10d)
    V_option_at_10d = V_option_at_2y * P_10d_to_2y

    V_portfolio_at_10d = (
        V_FRA_at_10d + V_swap_at_10d + V_stepup_at_10d + V_option_at_10d
    )

    # P&L calculation
    P_L_total = V_portfolio_at_10d - V_t0
    # </editor-fold>

    # <editor-fold desc="Task 2: Estimate VaR99% and ES97.5%">
    # Risk metrics - total portfolio
    VaR_99 = -np.percentile(P_L_total, 1) * scale_factor
    ES_97_5 = (
        -np.mean(P_L_total[P_L_total <= np.percentile(P_L_total, 2.5)]) * scale_factor
    )

    # Component values at t=0 for diagnostics
    V_FRA_t0_val = V_FRA_t0(
        T_k_minus1=FRA_T_K_minus1, T_k=FRA_T_K, K=FRA_strike, yield_curve=curve
    )
    V_swap_t0_val = HW_SwapPrice(
        CP=OptionTypeSwap.PAYER,
        notional=1,
        K=4.1 / 100,
        t=0,
        Ti=0,
        Tm=10,
        n=20,
        r_t=HW_r_0(P0T=curve, lambd=a, eta=sigma),
        P0T=curve,
        lambd=a,
        eta=sigma,
    )
    V_stepup_t0_val = V_stepup_t0(
        yield_curve=curve, step_up_prob=step_up_prob, a=a, sigma=sigma
    )
    V_option_t0_val = HW_ZCB_CallPutPrice(
        CP=OptionType.CALL, K=option_strike, lambd=a, eta=sigma, P0T=curve, T1=2, T2=10
    )

    # Convert arrays to scalars using item() or indexing
    v_t0_scalar = V_t0.item() if hasattr(V_t0, "item") else V_t0
    v_fra_scalar = (
        V_FRA_t0_val.item() if hasattr(V_FRA_t0_val, "item") else V_FRA_t0_val
    )
    v_swap_scalar = (
        V_swap_t0_val.item() if hasattr(V_swap_t0_val, "item") else V_swap_t0_val
    )
    v_stepup_scalar = (
        V_stepup_t0_val.item() if hasattr(V_stepup_t0_val, "item") else V_stepup_t0_val
    )
    v_option_scalar = (
        V_option_t0_val.item() if hasattr(V_option_t0_val, "item") else V_option_t0_val
    )

    print(f"Portfolio value at t=0: {v_t0_scalar * scale_factor:,.2f}")
    print(f"  FRA t=0: {v_fra_scalar * scale_factor:,.2f}")
    print(f"  Swap t=0: {v_swap_scalar * scale_factor:,.2f}")
    print(f"  Step-up t=0: {v_stepup_scalar * scale_factor:,.2f}")
    print(f"  Option t=0: {v_option_scalar * scale_factor:,.2f}")
    print(f"Mean P&L at t=10d: {np.mean(P_L_total) * scale_factor:,.2f}")
    print(f"VaR 99%: {VaR_99:,.2f}")
    print(f"ES 97.5%: {ES_97_5:,.2f}")
    # </editor-fold>

    # <editor-fold desc="Task 3: Attribute risk by instrument (leave-one-out)">
    # Only calculate incremental risk for default scenario
    if curve_type.lower() == "default":
        # Leave-one-out incremental VaR/ES
        # Instruments: FRA, Swap+Step-up (combined), Option

        # Portfolio without FRA
        V_t0_no_FRA = V_t0 - V_FRA_t0(
            T_k_minus1=FRA_T_K_minus1, T_k=FRA_T_K, K=FRA_strike, yield_curve=curve
        )
        V_portfolio_at_10d_no_FRA = V_portfolio_at_10d - V_FRA_at_10d
        P_L_no_FRA = V_portfolio_at_10d_no_FRA - V_t0_no_FRA
        VaR_99_no_FRA = -np.percentile(P_L_no_FRA, 1)
        ES_97_5_no_FRA = -np.mean(
            P_L_no_FRA[P_L_no_FRA <= np.percentile(P_L_no_FRA, 2.5)]
        )

        # Portfolio without Swap+Step-up
        V_swap_stepup_t0 = HW_SwapPrice(
            CP=OptionTypeSwap.PAYER,
            notional=1,
            K=4.1 / 100,
            t=0,
            Ti=0,
            Tm=10,
            n=20,
            r_t=HW_r_0(P0T=curve, lambd=a, eta=sigma),
            P0T=curve,
            lambd=a,
            eta=sigma,
        ) + V_stepup_t0(yield_curve=curve, step_up_prob=step_up_prob, a=a, sigma=sigma)
        V_t0_no_swap_stepup = V_t0 - V_swap_stepup_t0
        V_portfolio_at_10d_no_swap_stepup = (
            V_portfolio_at_10d - V_swap_at_10d - V_stepup_at_10d
        )
        P_L_no_swap_stepup = V_portfolio_at_10d_no_swap_stepup - V_t0_no_swap_stepup
        VaR_99_no_swap_stepup = -np.percentile(P_L_no_swap_stepup, 1)
        ES_97_5_no_swap_stepup = -np.mean(
            P_L_no_swap_stepup[
                P_L_no_swap_stepup <= np.percentile(P_L_no_swap_stepup, 2.5)
            ]
        )

        # Portfolio without Option
        V_option_t0 = HW_ZCB_CallPutPrice(
            CP=OptionType.CALL,
            K=option_strike,
            lambd=a,
            eta=sigma,
            P0T=curve,
            T1=2,
            T2=10,
        )
        V_t0_no_option = V_t0 - V_option_t0
        V_portfolio_at_10d_no_option = V_portfolio_at_10d - V_option_at_10d
        P_L_no_option = V_portfolio_at_10d_no_option - V_t0_no_option
        VaR_99_no_option = -np.percentile(P_L_no_option, 1)
        ES_97_5_no_option = -np.mean(
            P_L_no_option[P_L_no_option <= np.percentile(P_L_no_option, 2.5)]
        )

        print(f"\nLeave-One-Out Risk Attribution:")
        print(
            f"  Portfolio without FRA - VaR 99%: {VaR_99_no_FRA * scale_factor:,.2f}, ES 97.5%: {ES_97_5_no_FRA * scale_factor:,.2f}"
        )
        print(
            f"  Portfolio without Swap+Step-up - VaR 99%: {VaR_99_no_swap_stepup * scale_factor:,.2f}, ES 97.5%: {ES_97_5_no_swap_stepup * scale_factor:,.2f}"
        )
        print(
            f"  Portfolio without Option - VaR 99%: {VaR_99_no_option * scale_factor:,.2f}, ES 97.5%: {ES_97_5_no_option * scale_factor:,.2f}"
        )
    # </editor-fold>
