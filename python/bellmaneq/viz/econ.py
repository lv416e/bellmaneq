"""Economics model visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_value_function(
    grid: np.ndarray,
    values: np.ndarray,
    *,
    xlabel: str = "State",
    title: str = "Value Function",
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plots a 1D value function curve.

    Args:
        grid: State grid array.
        values: Value function array.
        xlabel: Label for the x-axis.
        title: Plot title.
        ax: Existing Axes to draw on.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(grid, values, "b-", linewidth=2, label="V(x)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_policy_function(
    grid: np.ndarray,
    policy: np.ndarray,
    *,
    xlabel: str = "State",
    ylabel: str = "Action",
    title: str = "Policy Function",
    forty_five: bool = False,
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plots a 1D policy function.

    Args:
        grid: State grid array.
        policy: Policy array.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.
        title: Plot title.
        forty_five: If True, add a 45-degree line.
        ax: Existing Axes to draw on.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(grid, policy, "r-", linewidth=2)

    if forty_five:
        lo = min(grid.min(), policy.min())
        hi = max(grid.max(), policy.max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="45-degree line")
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


def plot_cake_eating(
    cake_grid: np.ndarray,
    values: np.ndarray,
    policy: np.ndarray,
) -> Figure:
    """Plots Cake Eating Problem results (value function + consumption policy).

    Args:
        cake_grid: Cake size grid.
        values: Value function.
        policy: Optimal consumption policy.

    Returns:
        matplotlib Figure with two panels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plot_value_function(
        cake_grid, values,
        xlabel="Cake Size (x)",
        title="Value Function V(x)",
        ax=axes[0],
    )
    plot_policy_function(
        cake_grid, policy,
        xlabel="Cake Size (x)",
        ylabel="Consumption c*(x)",
        title="Optimal Consumption Policy",
        forty_five=True,
        ax=axes[1],
    )

    fig.suptitle("Cake Eating Problem", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_mccall(
    wage_grid: np.ndarray,
    values: np.ndarray,
    policy: np.ndarray,
    reservation_wage: float,
) -> Figure:
    """Plots McCall Job Search results.

    Args:
        wage_grid: Wage offer grid.
        values: Value function.
        policy: Accept/reject policy (1=accept, 0=reject).
        reservation_wage: Computed reservation wage.

    Returns:
        matplotlib Figure with three panels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Value function
    plot_value_function(
        wage_grid, values,
        xlabel="Wage Offer (w)",
        title="Value Function V(w)",
        ax=axes[0],
    )
    axes[0].axvline(reservation_wage, color="red", linestyle="--", alpha=0.7,
                    label=f"w* = {reservation_wage:.2f}")
    axes[0].legend()

    # Accept/reject regions
    accept_mask = policy.astype(bool)
    reject_mask = ~accept_mask
    axes[1].fill_between(wage_grid, 0, 1, where=reject_mask,
                         alpha=0.3, color="red", label="Reject")
    axes[1].fill_between(wage_grid, 0, 1, where=accept_mask,
                         alpha=0.3, color="green", label="Accept")
    axes[1].axvline(reservation_wage, color="black", linestyle="--",
                    label=f"w* = {reservation_wage:.2f}")
    axes[1].set_xlabel("Wage Offer (w)")
    axes[1].set_title("Accept/Reject Regions")
    axes[1].set_yticks([])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="x")

    # Reservation wage sensitivity placeholder: bar chart
    axes[2].barh(["Reservation\nWage"], [reservation_wage], color="steelblue")
    axes[2].set_xlabel("Wage")
    axes[2].set_title(f"Reservation Wage: w* = {reservation_wage:.2f}")
    axes[2].grid(True, alpha=0.3, axis="x")

    fig.suptitle("McCall Job Search Model", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_growth_model(
    capital_grid: np.ndarray,
    productivity_grid: np.ndarray,
    values: np.ndarray,
    policy: np.ndarray,
) -> Figure:
    """Plots Stochastic Growth Model results.

    Args:
        capital_grid: Capital grid (n_k,).
        productivity_grid: Productivity grid (n_z,).
        values: Value function (n_k, n_z).
        policy: Consumption policy (n_k, n_z).

    Returns:
        matplotlib Figure with three panels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    K, Z = np.meshgrid(capital_grid, productivity_grid, indexing="ij")

    # Value function heatmap
    im0 = axes[0].pcolormesh(capital_grid, productivity_grid, values.T, shading="auto", cmap="viridis")
    axes[0].set_xlabel("Capital (k)")
    axes[0].set_ylabel("Productivity (z)")
    axes[0].set_title("Value Function V(k, z)")
    fig.colorbar(im0, ax=axes[0])

    # Consumption policy for selected z levels
    z_indices = [0, len(productivity_grid) // 2, len(productivity_grid) - 1]
    for zi in z_indices:
        axes[1].plot(capital_grid, policy[:, zi], linewidth=1.5,
                     label=f"z = {productivity_grid[zi]:.3f}")
    axes[1].set_xlabel("Capital (k)")
    axes[1].set_ylabel("Consumption c*(k, z)")
    axes[1].set_title("Consumption Policy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Capital accumulation policy: k' = z*k^alpha + (1-delta)*k - c
    # Show policy as investment = output - consumption
    for zi in z_indices:
        investment = values[:, zi] * 0  # placeholder — show savings
        # k' approximation not directly available; show value slices instead
        axes[2].plot(capital_grid, values[:, zi], linewidth=1.5,
                     label=f"z = {productivity_grid[zi]:.3f}")
    axes[2].set_xlabel("Capital (k)")
    axes[2].set_ylabel("V(k, z)")
    axes[2].set_title("Value Function Slices")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Stochastic Growth Model", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_income_fluctuation(
    asset_grid: np.ndarray,
    income_grid: np.ndarray,
    values: np.ndarray,
    policy: np.ndarray,
    savings_policy: np.ndarray,
) -> Figure:
    """Plots Income Fluctuation Problem results.

    Args:
        asset_grid: Asset grid (n_a,).
        income_grid: Income grid (n_y,).
        values: Value function (n_a, n_y).
        policy: Consumption policy (n_a, n_y).
        savings_policy: Savings policy a' (n_a, n_y).

    Returns:
        matplotlib Figure with three panels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Value function heatmap
    im0 = axes[0].pcolormesh(asset_grid, income_grid, values.T, shading="auto", cmap="viridis")
    axes[0].set_xlabel("Assets (a)")
    axes[0].set_ylabel("Income (y)")
    axes[0].set_title("Value Function V(a, y)")
    fig.colorbar(im0, ax=axes[0])

    # Consumption policy for selected y levels
    y_indices = [0, len(income_grid) // 2, len(income_grid) - 1]
    for yi in y_indices:
        axes[1].plot(asset_grid, policy[:, yi], linewidth=1.5,
                     label=f"y = {income_grid[yi]:.3f}")
    axes[1].set_xlabel("Assets (a)")
    axes[1].set_ylabel("Consumption c*(a, y)")
    axes[1].set_title("Consumption Policy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Savings policy with 45-degree line
    for yi in y_indices:
        axes[2].plot(asset_grid, savings_policy[:, yi], linewidth=1.5,
                     label=f"y = {income_grid[yi]:.3f}")
    lo = asset_grid.min()
    hi = asset_grid.max()
    axes[2].plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="45-degree line")
    axes[2].set_xlabel("Current Assets (a)")
    axes[2].set_ylabel("Next-Period Assets (a')")
    axes[2].set_title("Savings Policy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Income Fluctuation Problem", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
