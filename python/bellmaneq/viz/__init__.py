"""Visualization helper module."""

from bellmaneq.viz.convergence import plot_convergence, plot_solver_comparison
from bellmaneq.viz.games import plot_tictactoe, plot_connect_four, plot_minimax_tree
from bellmaneq.viz.finance import (
    plot_exercise_boundary,
    plot_option_payoff,
    plot_price_convergence,
    plot_option_surface,
)
from bellmaneq.viz.econ import (
    plot_value_function,
    plot_policy_function,
    plot_cake_eating,
    plot_mccall,
    plot_growth_model,
    plot_income_fluctuation,
)

__all__ = [
    "plot_convergence",
    "plot_solver_comparison",
    "plot_tictactoe",
    "plot_connect_four",
    "plot_minimax_tree",
    "plot_exercise_boundary",
    "plot_option_payoff",
    "plot_price_convergence",
    "plot_option_surface",
    "plot_value_function",
    "plot_policy_function",
    "plot_cake_eating",
    "plot_mccall",
    "plot_growth_model",
    "plot_income_fluctuation",
]
