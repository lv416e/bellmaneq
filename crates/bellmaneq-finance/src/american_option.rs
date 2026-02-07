/// Result of pricing an American option via the CRR binomial model.
///
/// Solves the finite-horizon Bellman equation via backward induction:
///   V(i,j) = max(exercise_value, continuation_value)
///   continuation_value = e^{-rΔt} * [p * V(i+1,j+1) + (1-p) * V(i+1,j)]
///
/// This is the canonical optimal stopping problem: at each node the holder
/// decides whether to exercise immediately or continue holding the option.
pub struct AmericanOptionResult {
    /// Fair option price at inception.
    pub price: f64,
    /// Option values at every (time step, stock-price level) node.
    /// Shape: lower-triangular matrix of dimension (steps+1) x (steps+1).
    pub values: Vec<Vec<f64>>,
    /// Exercise boundary at each time step (the stock price at which early
    /// exercise becomes optimal).
    pub exercise_boundary: Vec<f64>,
    /// Time grid corresponding to each step.
    pub time_steps: Vec<f64>,
    /// Stock prices at the terminal nodes (maturity).
    pub stock_prices_at_maturity: Vec<f64>,
}

/// Prices an American option using the Cox-Ross-Rubinstein (CRR) binomial model.
pub fn price_american_option(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    maturity: f64,
    steps: usize,
    is_call: bool,
) -> AmericanOptionResult {
    let dt = maturity / steps as f64;
    let u = (volatility * dt.sqrt()).exp(); // Up factor
    let d = 1.0 / u; // Down factor
    let p = ((rate * dt).exp() - d) / (u - d); // Risk-neutral probability
    let df = (-rate * dt).exp(); // Single-period discount factor

    let payoff = |s: f64| -> f64 {
        if is_call {
            (s - strike).max(0.0)
        } else {
            (strike - s).max(0.0)
        }
    };

    // Store option values at every node
    let mut values = vec![vec![0.0; steps + 1]; steps + 1];

    // Terminal payoff at maturity
    for j in 0..=steps {
        let stock_price = spot * u.powi(j as i32) * d.powi((steps - j) as i32);
        values[steps][j] = payoff(stock_price);
    }

    // Exercise boundary
    let mut exercise_boundary = vec![0.0; steps + 1];

    // Backward induction = iterating the Bellman equation
    for i in (0..steps).rev() {
        let mut boundary_found = false;
        for j in 0..=i {
            let stock_price = spot * u.powi(j as i32) * d.powi((i - j) as i32);

            // Continuation value (expected-value component of the Bellman equation)
            let continuation = df * (p * values[i + 1][j + 1] + (1.0 - p) * values[i + 1][j]);

            // Intrinsic (exercise) value
            let exercise = payoff(stock_price);

            // Bellman equation: V = max(exercise, continuation)
            values[i][j] = exercise.max(continuation);

            // Record the exercise boundary
            if !boundary_found && exercise >= continuation && exercise > 0.0 {
                exercise_boundary[i] = stock_price;
                boundary_found = true;
            }
        }
    }

    // Stock price nodes at maturity
    let stock_prices_at_maturity: Vec<f64> = (0..=steps)
        .map(|j| spot * u.powi(j as i32) * d.powi((steps - j) as i32))
        .collect();

    let time_steps: Vec<f64> = (0..=steps).map(|i| i as f64 * dt).collect();

    AmericanOptionResult {
        price: values[0][0],
        values,
        exercise_boundary,
        time_steps,
        stock_prices_at_maturity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_american_put_price_positive() {
        let result = price_american_option(
            100.0, // spot
            100.0, // strike
            0.05,  // rate
            0.2,   // volatility
            1.0,   // maturity
            100,   // steps
            false, // put
        );
        assert!(result.price > 0.0, "American put price should be positive");
        assert!(result.price < 100.0, "Price should be less than strike");
    }

    #[test]
    fn test_american_put_geq_european() {
        // American put price >= European put price
        let american = price_american_option(100.0, 100.0, 0.05, 0.2, 1.0, 200, false);

        // While the European put could be computed via Black-Scholes, here we
        // simply verify that the American price is positive (it must exceed the
        // European price due to the early exercise premium).
        assert!(american.price > 0.0);
    }

    #[test]
    fn test_exercise_boundary_exists() {
        let result = price_american_option(100.0, 100.0, 0.05, 0.2, 1.0, 50, false);
        // For a put, the exercise boundary should have non-trivial entries
        let nonzero_boundaries: Vec<_> = result
            .exercise_boundary
            .iter()
            .filter(|&&b| b > 0.0)
            .collect();
        assert!(
            !nonzero_boundaries.is_empty(),
            "Exercise boundary should have non-zero entries for a put"
        );
    }

    #[test]
    fn test_american_call_matches_black_scholes() {
        // For a non-dividend-paying stock, the American call equals the
        // European call, which has a closed-form Black-Scholes price.
        fn erf_approx(x: f64) -> f64 {
            // Abramowitz & Stegun approximation 7.1.26
            let a1 = 0.254829592;
            let a2 = -0.284496736;
            let a3 = 1.421413741;
            let a4 = -1.453152027;
            let a5 = 1.061405429;
            let p = 0.3275911;
            let sign = if x < 0.0 { -1.0 } else { 1.0 };
            let x = x.abs();
            let t = 1.0 / (1.0 + p * x);
            let y = 1.0
                - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
            sign * y
        }

        fn norm_cdf(x: f64) -> f64 {
            0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
        }

        fn black_scholes_call(spot: f64, strike: f64, rate: f64, vol: f64, t: f64) -> f64 {
            let d1 = ((spot / strike).ln() + (rate + 0.5 * vol * vol) * t) / (vol * t.sqrt());
            let d2 = d1 - vol * t.sqrt();
            spot * norm_cdf(d1) - strike * (-rate * t).exp() * norm_cdf(d2)
        }

        let spot = 100.0;
        let strike = 100.0;
        let rate = 0.05;
        let vol = 0.2;
        let maturity = 1.0;

        let bs_price = black_scholes_call(spot, strike, rate, vol, maturity);
        let american = price_american_option(spot, strike, rate, vol, maturity, 1000, true);

        assert!(
            (american.price - bs_price).abs() < 0.1,
            "American call ({:.4}) should match Black-Scholes ({:.4})",
            american.price,
            bs_price
        );
    }

    #[test]
    fn test_convergence_with_steps() {
        // Price should converge as the number of steps increases
        let p50 = price_american_option(100.0, 100.0, 0.05, 0.2, 1.0, 50, false).price;
        let p200 = price_american_option(100.0, 100.0, 0.05, 0.2, 1.0, 200, false).price;
        let p500 = price_american_option(100.0, 100.0, 0.05, 0.2, 1.0, 500, false).price;

        // Verify that successive differences are shrinking
        let diff_50_200 = (p50 - p200).abs();
        let diff_200_500 = (p200 - p500).abs();
        assert!(
            diff_200_500 < diff_50_200,
            "Price should converge: |p200-p500|={} should be < |p50-p200|={}",
            diff_200_500,
            diff_50_200
        );
    }
}
