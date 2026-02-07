/// American option pricing via the explicit finite difference method.
///
/// Discretises the Black-Scholes PDE on a (stock price, time) grid and
/// solves backward in time with the early-exercise constraint:
///
///   V(S, t) = max( payoff(S),  continuation(S, t) )
///
/// The PDE (for a non-dividend-paying stock):
///   ∂V/∂t + ½σ²S² ∂²V/∂S² + rS ∂V/∂S − rV = 0
///
/// This provides an alternative to the CRR binomial model and converges
/// to the same price as the grid is refined.

/// Result of finite difference pricing.
pub struct FiniteDifferenceResult {
    /// Fair option price at the current spot.
    pub price: f64,
    /// Full grid of option values.  Shape: `(n_time_steps + 1) × (n_price_steps + 1)`.
    /// `grid_values[t][j]` is the option value at time index `t` and stock-price index `j`.
    pub grid_values: Vec<Vec<f64>>,
    /// Stock-price grid points.
    pub stock_grid: Vec<f64>,
    /// Time grid points.
    pub time_grid: Vec<f64>,
}

/// Prices an American option using the explicit finite difference scheme.
///
/// # Parameters
///
/// * `spot` – current stock price S₀.
/// * `strike` – option strike price K.
/// * `rate` – risk-free interest rate r.
/// * `volatility` – annualised volatility σ.
/// * `maturity` – time to maturity T (in years).
/// * `n_price_steps` – number of stock-price grid intervals.
/// * `n_time_steps` – number of time steps.
/// * `s_max_factor` – determines S_max = s_max_factor × spot.
/// * `is_call` – `true` for a call, `false` for a put.
///
/// # Stability
///
/// The explicit scheme has a CFL stability condition.  If the grid is too
/// coarse in time relative to the spatial grid, oscillations may appear.
/// Increasing `n_time_steps` resolves this.
pub fn price_american_fd(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    maturity: f64,
    n_price_steps: usize,
    n_time_steps: usize,
    s_max_factor: f64,
    is_call: bool,
) -> FiniteDifferenceResult {
    let s_max = spot * s_max_factor;
    let ds = s_max / n_price_steps as f64;
    let dt = maturity / n_time_steps as f64;
    let sigma2 = volatility * volatility;

    let stock_grid: Vec<f64> = (0..=n_price_steps).map(|j| j as f64 * ds).collect();
    let time_grid: Vec<f64> = (0..=n_time_steps).map(|i| i as f64 * dt).collect();

    let payoff = |s: f64| -> f64 {
        if is_call {
            (s - strike).max(0.0)
        } else {
            (strike - s).max(0.0)
        }
    };

    // Terminal condition: V(S, T) = payoff(S).
    let mut v: Vec<f64> = stock_grid.iter().map(|&s| payoff(s)).collect();
    let mut grid_values = vec![vec![0.0; n_price_steps + 1]; n_time_steps + 1];
    grid_values[n_time_steps] = v.clone();

    // Step backward through time.
    for n in (0..n_time_steps).rev() {
        let mut v_new = vec![0.0; n_price_steps + 1];

        // Interior nodes: j = 1 .. n_price_steps - 1.
        for j in 1..n_price_steps {
            let s_j = j as f64 * ds;
            let s_j2 = s_j * s_j;

            // Explicit finite difference coefficients.
            let alpha = 0.5 * dt * (sigma2 * s_j2 / (ds * ds) - rate * s_j / ds);
            let beta = 1.0 - dt * (sigma2 * s_j2 / (ds * ds) + rate);
            let gamma_coeff = 0.5 * dt * (sigma2 * s_j2 / (ds * ds) + rate * s_j / ds);

            let continuation = alpha * v[j - 1] + beta * v[j] + gamma_coeff * v[j + 1];

            // Early-exercise constraint: V = max(payoff, continuation).
            v_new[j] = continuation.max(payoff(s_j));
        }

        // Boundary conditions.
        let remaining_time = (n_time_steps - n) as f64 * dt;
        if is_call {
            v_new[0] = 0.0;
            v_new[n_price_steps] =
                (s_max - strike * (-rate * remaining_time).exp()).max(0.0);
        } else {
            v_new[0] = (strike * (-rate * remaining_time).exp()).max(0.0);
            v_new[n_price_steps] = 0.0;
        }

        grid_values[n] = v_new.clone();
        v = v_new;
    }

    // Linear interpolation to find the price at the exact spot.
    let j_spot = ((spot / ds) as usize).min(n_price_steps - 1);
    let weight = (spot - stock_grid[j_spot]) / ds;
    let price = v[j_spot] * (1.0 - weight) + v[j_spot + 1] * weight;

    FiniteDifferenceResult {
        price,
        grid_values,
        stock_grid,
        time_grid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::american_option::price_american_option;

    #[test]
    fn test_fd_price_positive() {
        let result =
            price_american_fd(100.0, 100.0, 0.05, 0.2, 1.0, 200, 2000, 3.0, false);
        assert!(result.price > 0.0, "FD put price should be positive");
        assert!(result.price < 100.0, "FD put price should be less than strike");
    }

    #[test]
    fn test_fd_vs_binomial() {
        let fd = price_american_fd(100.0, 100.0, 0.05, 0.2, 1.0, 200, 2000, 4.0, false);
        let binom = price_american_option(100.0, 100.0, 0.05, 0.2, 1.0, 500, false);
        assert!(
            (fd.price - binom.price).abs() < 0.5,
            "FD price ({:.4}) and binomial price ({:.4}) should be close",
            fd.price,
            binom.price
        );
    }

    #[test]
    fn test_fd_call_price() {
        let result =
            price_american_fd(100.0, 100.0, 0.05, 0.2, 1.0, 200, 2000, 3.0, true);
        assert!(result.price > 0.0, "FD call price should be positive");
    }
}
