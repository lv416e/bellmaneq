"""Python tests for the economics models."""

import numpy as np
import bellmaneq


class TestCakeEating:
    """Tests for the Cake Eating Problem."""

    def test_converges(self):
        result = bellmaneq.solve_cake_eating(discount=0.95)
        assert result.converged

    def test_value_function_monotone(self):
        result = bellmaneq.solve_cake_eating(discount=0.95, n_cake=30, n_consumption=30)
        values = result.get_values()
        cake_grid = result.get_cake_grid()
        # V(x) should be non-decreasing in x (skip terminal state at index 0)
        for i in range(1, len(cake_grid) - 1):
            assert values[i + 1] >= values[i] - 1e-6

    def test_policy_positive(self):
        result = bellmaneq.solve_cake_eating(discount=0.95)
        policy = result.get_policy()
        cake_grid = result.get_cake_grid()
        # For positive cake sizes, consumption should be positive
        for i, x in enumerate(cake_grid):
            if x > 0:
                assert policy[i] > 0, f"Expected positive consumption at cake size {x}"


class TestMcCall:
    """Tests for the McCall Job Search Model."""

    def test_converges(self):
        result = bellmaneq.solve_mccall(discount=0.95, unemployment_comp=25.0)
        assert result.converged

    def test_reservation_wage_exists(self):
        result = bellmaneq.solve_mccall(discount=0.95, unemployment_comp=25.0)
        assert result.reservation_wage > 0

    def test_reservation_wage_comparative_statics(self):
        """Higher unemployment compensation should increase the reservation wage."""
        r_low = bellmaneq.solve_mccall(discount=0.95, unemployment_comp=20.0)
        r_high = bellmaneq.solve_mccall(discount=0.95, unemployment_comp=30.0)
        assert r_high.reservation_wage >= r_low.reservation_wage - 1e-6


class TestGrowthModel:
    """Tests for the Stochastic Growth Model."""

    def test_converges(self):
        result = bellmaneq.solve_growth(discount=0.95, n_k=30, n_z=5)
        assert result.converged

    def test_value_shape(self):
        n_k, n_z = 30, 5
        result = bellmaneq.solve_growth(discount=0.95, n_k=n_k, n_z=n_z)
        values = result.get_values()
        assert values.shape == (n_k, n_z)


class TestIncomeFluctuation:
    """Tests for the Income Fluctuation Problem."""

    def test_converges(self):
        result = bellmaneq.solve_income_fluctuation(discount=0.95, n_a=30, n_y=5)
        assert result.converged

    def test_borrowing_constraint_effect(self):
        """Relaxing the borrowing constraint should weakly improve utility."""
        r_tight = bellmaneq.solve_income_fluctuation(
            discount=0.95, borrowing_limit=0.0, n_a=30, n_y=5,
        )
        r_loose = bellmaneq.solve_income_fluctuation(
            discount=0.95, borrowing_limit=1.0, n_a=30, n_y=5,
        )
        v_tight = r_tight.get_values()
        v_loose = r_loose.get_values()
        # Average value should be weakly higher with looser constraint
        assert np.mean(v_loose) >= np.mean(v_tight) - 1e-4
