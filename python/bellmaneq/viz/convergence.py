"""Convergence curve visualization."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def plot_convergence(
    history: np.ndarray,
    *,
    title: str = "Value Iteration Convergence",
    ax: plt.Axes | None = None,
) -> Figure | None:
    """Plots the convergence history on a logarithmic scale.

    Args:
        history: Array of max|V_{k+1} - V_k| at each iteration.
        title: Plot title.
        ax: Existing Axes to draw on (creates a new figure if None).

    Returns:
        The newly created Figure, or None if an existing Axes was provided.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    iterations = np.arange(1, len(history) + 1)
    ax.semilogy(iterations, history, "b-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("max|V_{k+1} - V_k|")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


def plot_solver_comparison(
    results: dict[str, np.ndarray],
    *,
    title: str = "Solver Convergence Comparison",
) -> Figure:
    """Compares convergence curves of multiple solvers.

    Args:
        results: Mapping of solver name to its convergence history array.
        title: Plot title.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    for (name, history), color in zip(results.items(), colors):
        iterations = np.arange(1, len(history) + 1)
        ax.semilogy(iterations, history, linewidth=1.5, label=name, color=color)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("max|V_{k+1} - V_k|")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
