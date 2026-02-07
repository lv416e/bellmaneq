"""Financial model visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_exercise_boundary(
    time_steps: np.ndarray,
    exercise_boundary: np.ndarray,
    *,
    strike: float | None = None,
    title: str = "American Option Exercise Boundary",
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plots the early exercise boundary.

    Args:
        time_steps: Array of time step values.
        exercise_boundary: Stock price at the exercise boundary for each time step.
        strike: Strike price (displayed as a horizontal reference line).
        title: Plot title.
        ax: Existing Axes to draw on.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot only non-zero boundary values
    mask = exercise_boundary > 0
    ax.plot(
        time_steps[mask],
        exercise_boundary[mask],
        "r-",
        linewidth=2,
        label="Exercise Boundary",
    )

    if strike is not None:
        ax.axhline(strike, color="gray", linestyle="--", alpha=0.7, label=f"Strike = {strike}")

    # Shade the exercise and continuation regions
    if mask.any():
        ax.fill_between(
            time_steps[mask],
            0,
            exercise_boundary[mask],
            alpha=0.1,
            color="red",
            label="Exercise Region",
        )
        ax.fill_between(
            time_steps[mask],
            exercise_boundary[mask],
            exercise_boundary[mask].max() * 1.5,
            alpha=0.1,
            color="green",
            label="Continuation Region",
        )

    ax.set_xlabel("Time to Maturity")
    ax.set_ylabel("Stock Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_option_payoff(
    stock_prices: np.ndarray,
    strike: float,
    option_price: float,
    *,
    is_call: bool = False,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plots the option payoff diagram.

    Args:
        stock_prices: Array of stock prices.
        strike: Strike price.
        option_price: Option premium.
        is_call: True for a call option, False for a put.
        title: Plot title.
        ax: Existing Axes to draw on.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if is_call:
        intrinsic = np.maximum(stock_prices - strike, 0)
        label = "Call"
    else:
        intrinsic = np.maximum(strike - stock_prices, 0)
        label = "Put"

    profit = intrinsic - option_price

    ax.plot(stock_prices, intrinsic, "b--", linewidth=1.5, label=f"{label} Payoff")
    ax.plot(stock_prices, profit, "r-", linewidth=2, label=f"{label} Profit (premium={option_price:.2f})")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(strike, color="gray", linestyle=":", alpha=0.7, label=f"Strike = {strike}")

    if title is None:
        title = f"American {label} Option Payoff"
    ax.set_xlabel("Stock Price at Maturity")
    ax.set_ylabel("Payoff / Profit")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_price_convergence(
    steps_list: list[int],
    prices: list[float],
    *,
    reference_price: float | None = None,
    title: str = "Option Price Convergence",
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plots option price convergence as a function of the number of steps.

    Args:
        steps_list: List of step counts.
        prices: Option price computed at each step count.
        reference_price: Reference price (e.g., analytical solution) shown as a horizontal line.
        title: Plot title.
        ax: Existing Axes to draw on.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps_list, prices, "bo-", linewidth=1.5, label="Binomial Model")

    if reference_price is not None:
        ax.axhline(
            reference_price,
            color="red",
            linestyle="--",
            label=f"Reference = {reference_price:.4f}",
        )

    ax.set_xlabel("Number of Steps")
    ax.set_ylabel("Option Price")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_option_surface(
    spot: float = 100.0,
    strike: float = 100.0,
    rate: float = 0.05,
    volatility: float = 0.2,
    maturity: float = 1.0,
    steps: int = 100,
    *,
    is_call: bool = False,
    n_spot_points: int = 50,
    n_time_points: int = 50,
    title: str | None = None,
) -> Figure:
    """Plots a 3-D surface of option value vs. spot price and time to maturity.

    Args:
        spot: Reference spot price (sets the range centre).
        strike: Strike price.
        rate: Risk-free rate.
        volatility: Annualised volatility.
        maturity: Maximum time to maturity.
        steps: Number of binomial steps per pricing call.
        is_call: ``True`` for a call, ``False`` for a put.
        n_spot_points: Grid resolution along the spot axis.
        n_time_points: Grid resolution along the time axis.
        title: Plot title.

    Returns:
        matplotlib Figure with a 3-D surface plot.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import bellmaneq

    spot_range = np.linspace(spot * 0.5, spot * 1.5, n_spot_points)
    time_range = np.linspace(0.01, maturity, n_time_points)

    S, T = np.meshgrid(spot_range, time_range)
    V = np.zeros_like(S)

    for i in range(n_time_points):
        for j in range(n_spot_points):
            result = bellmaneq.price_american_option(
                spot=float(S[i, j]),
                strike=strike,
                rate=rate,
                volatility=volatility,
                maturity=float(T[i, j]),
                steps=steps,
                is_call=is_call,
            )
            V[i, j] = result.price

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S, T, V, cmap="viridis", alpha=0.8, edgecolor="none")

    ax.set_xlabel("Spot Price")
    ax.set_ylabel("Time to Maturity")
    ax.set_zlabel("Option Value")

    if title is None:
        option_type = "Call" if is_call else "Put"
        title = f"American {option_type} Option Value Surface (K={strike})"
    ax.set_title(title)

    return fig
