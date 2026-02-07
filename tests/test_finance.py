"""Python tests for the financial models."""

import numpy as np
import bellmaneq


class TestAmericanOption:
    """Tests for American option pricing."""

    def test_put_price_positive(self):
        result = bellmaneq.price_american_option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            steps=200,
        )
        assert result.price > 0

    def test_put_geq_intrinsic(self):
        """American put price must be at least as large as the intrinsic value."""
        spot, strike = 95.0, 100.0
        result = bellmaneq.price_american_option(
            spot=spot,
            strike=strike,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            steps=200,
        )
        intrinsic = max(strike - spot, 0)
        assert result.price >= intrinsic - 1e-10

    def test_call_option(self):
        """Call options can also be priced."""
        result = bellmaneq.price_american_option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            steps=200,
            is_call=True,
        )
        assert result.price > 0

    def test_exercise_boundary(self):
        result = bellmaneq.price_american_option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            steps=100,
        )
        boundary = result.get_exercise_boundary()
        assert len(boundary) > 0
        # For a put, the exercise boundary should be at or below the strike price
        valid = boundary[boundary > 0]
        assert all(b <= 100.0 + 1e-6 for b in valid)

    def test_time_steps(self):
        steps = 50
        result = bellmaneq.price_american_option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            steps=steps,
        )
        time_steps = result.get_time_steps()
        assert len(time_steps) == steps + 1
        assert abs(time_steps[0]) < 1e-10  # t=0
        assert abs(time_steps[-1] - 1.0) < 1e-10  # t=T

    def test_convergence_with_steps(self):
        """Option price should converge as the number of steps increases."""
        prices = []
        for steps in [50, 100, 200, 400]:
            result = bellmaneq.price_american_option(
                spot=100.0,
                strike=100.0,
                rate=0.05,
                volatility=0.2,
                maturity=1.0,
                steps=steps,
            )
            prices.append(result.price)

        # Later price differences should be smaller
        diff_early = abs(prices[1] - prices[0])
        diff_late = abs(prices[3] - prices[2])
        assert diff_late < diff_early

    def test_stock_prices_at_maturity(self):
        steps = 100
        result = bellmaneq.price_american_option(
            spot=100.0,
            strike=100.0,
            rate=0.05,
            volatility=0.2,
            maturity=1.0,
            steps=steps,
        )
        stock_prices = result.get_stock_prices_at_maturity()
        assert len(stock_prices) == steps + 1
        # All stock prices must be positive
        assert all(s > 0 for s in stock_prices)
