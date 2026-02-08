"""bellmaneq: A general-purpose dynamic programming solver built on the Bellman equation.

High-performance implementation powered by a Rust core with PyO3 bindings.
"""

from bellmaneq._core import (
    PySolverResult,
    solve_value_iteration,
    solve_policy_iteration,
    PyTicTacToe as TicTacToe,
    PyConnectFour as ConnectFour,
    PyAmericanOptionResult,
    price_american_option,
    PyCakeEatingResult,
    PyMcCallResult,
    PyGrowthResult,
    PyIncomeFluctuationResult,
    solve_cake_eating,
    solve_mccall,
    solve_growth,
    solve_income_fluctuation,
)

__all__ = [
    "PySolverResult",
    "solve_value_iteration",
    "solve_policy_iteration",
    "TicTacToe",
    "ConnectFour",
    "PyAmericanOptionResult",
    "price_american_option",
    "PyCakeEatingResult",
    "PyMcCallResult",
    "PyGrowthResult",
    "PyIncomeFluctuationResult",
    "solve_cake_eating",
    "solve_mccall",
    "solve_growth",
    "solve_income_fluctuation",
]
