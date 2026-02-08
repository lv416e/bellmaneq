"""Type stubs for the bellmaneq._core Rust extension module."""

import numpy as np
from numpy.typing import NDArray

class PySolverResult:
    """Result returned by MDP solvers."""

    iterations: int
    converged: bool

    def get_values(self) -> NDArray[np.float64]: ...
    def get_policy(self) -> NDArray[np.int64]: ...
    def get_convergence_history(self) -> NDArray[np.float64]: ...

def solve_value_iteration(
    rewards: NDArray[np.float64],
    transitions: NDArray[np.float64],
    gamma: float,
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> PySolverResult: ...

def solve_policy_iteration(
    rewards: NDArray[np.float64],
    transitions: NDArray[np.float64],
    gamma: float,
) -> PySolverResult: ...

class PyTicTacToe:
    """Tic-Tac-Toe game engine with minimax search."""

    def __init__(self) -> None: ...
    def minimax(self, board: list[int], next_player: int, depth: int) -> float: ...
    def best_move(self, board: list[int], next_player: int, depth: int) -> int | None: ...
    def legal_actions(self, board: list[int], next_player: int) -> list[int]: ...

class PyConnectFour:
    """Connect Four game engine with minimax search."""

    def __init__(self) -> None: ...
    def minimax(self, board: list[list[int]], next_player: int, depth: int) -> float: ...
    def best_move(self, board: list[list[int]], next_player: int, depth: int) -> int | None: ...
    def legal_actions(self, board: list[list[int]], next_player: int) -> list[int]: ...
    def apply_move(self, board: list[list[int]], next_player: int, col: int) -> list[list[int]]: ...
    def check_winner(self, board: list[list[int]]) -> int: ...
    @staticmethod
    def empty_board() -> list[list[int]]: ...

class PyAmericanOptionResult:
    """Result of American option pricing."""

    @property
    def price(self) -> float: ...
    def get_exercise_boundary(self) -> NDArray[np.float64]: ...
    def get_time_steps(self) -> NDArray[np.float64]: ...
    def get_stock_prices_at_maturity(self) -> NDArray[np.float64]: ...

def price_american_option(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    steps: int,
    is_call: bool = False,
) -> PyAmericanOptionResult: ...

# ---------------------------------------------------------------------------
# Economics models
# ---------------------------------------------------------------------------

class PyCakeEatingResult:
    """Result of solving the Cake Eating problem."""

    iterations: int
    converged: bool

    def get_values(self) -> NDArray[np.float64]: ...
    def get_policy(self) -> NDArray[np.float64]: ...
    def get_cake_grid(self) -> NDArray[np.float64]: ...
    def get_convergence_history(self) -> NDArray[np.float64]: ...

def solve_cake_eating(
    discount: float = 0.95,
    risk_aversion: float = 1.0,
    x_max: float = 1.0,
    n_cake: int = 50,
    n_consumption: int = 50,
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> PyCakeEatingResult: ...

class PyMcCallResult:
    """Result of solving the McCall Job Search model."""

    reservation_wage: float
    iterations: int
    converged: bool

    def get_values(self) -> NDArray[np.float64]: ...
    def get_policy(self) -> NDArray[np.int64]: ...
    def get_wage_grid(self) -> NDArray[np.float64]: ...
    def get_convergence_history(self) -> NDArray[np.float64]: ...

def solve_mccall(
    discount: float = 0.95,
    unemployment_comp: float = 25.0,
    w_min: float = 10.0,
    w_max: float = 60.0,
    n_wages: int = 50,
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> PyMcCallResult: ...

class PyGrowthResult:
    """Result of solving the Stochastic Growth model."""

    iterations: int
    converged: bool

    def get_values(self) -> NDArray[np.float64]: ...
    def get_policy(self) -> NDArray[np.float64]: ...
    def get_capital_grid(self) -> NDArray[np.float64]: ...
    def get_productivity_grid(self) -> NDArray[np.float64]: ...
    def get_convergence_history(self) -> NDArray[np.float64]: ...

def solve_growth(
    discount: float = 0.95,
    risk_aversion: float = 2.0,
    alpha: float = 0.36,
    delta: float = 0.1,
    rho: float = 0.9,
    sigma_z: float = 0.02,
    n_k: int = 50,
    n_z: int = 7,
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> PyGrowthResult: ...

class PyIncomeFluctuationResult:
    """Result of solving the Income Fluctuation problem."""

    iterations: int
    converged: bool

    def get_values(self) -> NDArray[np.float64]: ...
    def get_policy(self) -> NDArray[np.float64]: ...
    def get_savings_policy(self) -> NDArray[np.float64]: ...
    def get_asset_grid(self) -> NDArray[np.float64]: ...
    def get_income_grid(self) -> NDArray[np.float64]: ...
    def get_convergence_history(self) -> NDArray[np.float64]: ...

def solve_income_fluctuation(
    discount: float = 0.95,
    risk_aversion: float = 2.0,
    interest_rate: float = 0.03,
    borrowing_limit: float = 0.0,
    rho: float = 0.9,
    sigma_y: float = 0.1,
    n_a: int = 50,
    n_y: int = 7,
    tol: float = 1e-10,
    max_iter: int = 10000,
) -> PyIncomeFluctuationResult: ...
