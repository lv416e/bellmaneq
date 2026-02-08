"""Smoke tests for visualization functions."""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import bellmaneq  # noqa: E402
from bellmaneq.viz import (  # noqa: E402
    plot_convergence,
    plot_solver_comparison,
    plot_tictactoe,
    plot_connect_four,
    plot_minimax_tree,
    plot_exercise_boundary,
    plot_option_payoff,
    plot_price_convergence,
    plot_option_surface,
    plot_cake_eating,
    plot_mccall,
    plot_growth_model,
    plot_income_fluctuation,
)


class TestConvergenceViz:
    """Smoke tests for convergence visualizations."""

    def test_plot_convergence_returns_figure(self):
        history = np.array([1.0, 0.5, 0.1, 0.01, 0.001])
        fig = plot_convergence(history)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_convergence_with_existing_axes(self):
        _, ax = plt.subplots()
        history = np.array([1.0, 0.5, 0.1])
        result = plot_convergence(history, ax=ax)
        assert result is None
        plt.close("all")

    def test_plot_solver_comparison(self):
        results = {
            "VI": np.array([1.0, 0.5, 0.1]),
            "PI": np.array([0.8, 0.2]),
        }
        fig = plot_solver_comparison(results)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestGamesViz:
    """Smoke tests for game board visualizations."""

    def test_plot_tictactoe_empty(self):
        fig = plot_tictactoe([0] * 9)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_tictactoe_with_values(self):
        board = [1, 2, 0, 0, 1, 0, 0, 0, 2]
        values = [0.5, -0.3, 0.1, 0.2, 0.8, -0.1, 0.0, 0.3, -0.5]
        fig = plot_tictactoe(board, values=values)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_connect_four_empty(self):
        board = [[0] * 7 for _ in range(6)]
        fig = plot_connect_four(board)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_connect_four_with_last_move(self):
        board = [[0] * 7 for _ in range(6)]
        board[5][3] = 1
        fig = plot_connect_four(board, last_move=(5, 3))
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_minimax_tree(self):
        game = bellmaneq.TicTacToe()
        board = [0] * 9
        fig = plot_minimax_tree(game, board, next_player=1, depth=2)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestFinanceViz:
    """Smoke tests for financial visualizations."""

    def test_plot_exercise_boundary(self):
        time_steps = np.linspace(0, 1, 51)
        boundary = np.concatenate([np.zeros(10), np.linspace(90, 95, 41)])
        fig = plot_exercise_boundary(time_steps, boundary, strike=100.0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_option_payoff(self):
        stock_prices = np.linspace(50, 150, 100)
        fig = plot_option_payoff(stock_prices, strike=100.0, option_price=5.0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_price_convergence(self):
        steps_list = [50, 100, 200, 400]
        prices = [5.1, 5.05, 5.02, 5.01]
        fig = plot_price_convergence(steps_list, prices, reference_price=5.0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_option_surface(self):
        fig = plot_option_surface(
            spot=100.0,
            strike=100.0,
            steps=20,
            n_spot_points=5,
            n_time_points=5,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestEconViz:
    """Smoke tests for economics visualizations."""

    def test_plot_cake_eating(self):
        result = bellmaneq.solve_cake_eating(discount=0.95, n_cake=20, n_consumption=20)
        fig = plot_cake_eating(
            result.get_cake_grid(), result.get_values(), result.get_policy(),
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_mccall(self):
        result = bellmaneq.solve_mccall(discount=0.95, n_wages=20)
        fig = plot_mccall(
            result.get_wage_grid(), result.get_values(),
            result.get_policy(), result.reservation_wage,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_growth_model(self):
        result = bellmaneq.solve_growth(discount=0.95, n_k=20, n_z=5)
        fig = plot_growth_model(
            result.get_capital_grid(), result.get_productivity_grid(),
            result.get_values(), result.get_policy(),
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_income_fluctuation(self):
        result = bellmaneq.solve_income_fluctuation(discount=0.95, n_a=20, n_y=5)
        fig = plot_income_fluctuation(
            result.get_asset_grid(), result.get_income_grid(),
            result.get_values(), result.get_policy(), result.get_savings_policy(),
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
