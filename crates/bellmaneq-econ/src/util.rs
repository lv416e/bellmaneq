/// Approximation of the error function using the Abramowitz & Stegun
/// formula (7.1.26).
fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Standard normal cumulative distribution function.
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

/// Result of the Tauchen (1986) discretization of an AR(1) process.
pub struct TauchenResult {
    /// Grid of state values (length `n_states`).
    pub states: Vec<f64>,
    /// Transition probability matrix `P[i][j] = Pr(z_j | z_i)`.
    pub transition_matrix: Vec<Vec<f64>>,
}

/// Discretizes an AR(1) process `z' = rho * z + sigma * epsilon`
/// into a finite Markov chain using the Tauchen (1986) method.
///
/// # Parameters
/// * `rho` — Persistence parameter (0 < rho < 1).
/// * `sigma` — Standard deviation of the innovation epsilon.
/// * `n_states` — Number of grid points.
/// * `m` — Number of unconditional standard deviations for the grid range (typically 3).
pub fn tauchen(rho: f64, sigma: f64, n_states: usize, m: f64) -> TauchenResult {
    assert!(n_states >= 2, "tauchen requires at least 2 states");

    // Unconditional standard deviation of z
    let sigma_z = sigma / (1.0 - rho * rho).sqrt();
    let z_max = m * sigma_z;
    let z_min = -z_max;

    let states = linspace(z_min, z_max, n_states);
    let step = (z_max - z_min) / (n_states - 1) as f64;
    let half_step = step / 2.0;

    let mut transition_matrix = vec![vec![0.0; n_states]; n_states];

    for i in 0..n_states {
        let mean = rho * states[i];
        for j in 0..n_states {
            if j == 0 {
                transition_matrix[i][j] = norm_cdf((states[j] - mean + half_step) / sigma);
            } else if j == n_states - 1 {
                transition_matrix[i][j] =
                    1.0 - norm_cdf((states[j] - mean - half_step) / sigma);
            } else {
                transition_matrix[i][j] = norm_cdf((states[j] - mean + half_step) / sigma)
                    - norm_cdf((states[j] - mean - half_step) / sigma);
            }
        }
    }

    TauchenResult {
        states,
        transition_matrix,
    }
}

/// CRRA (Constant Relative Risk Aversion) utility function.
///
/// * `u(c) = c^{1-sigma} / (1-sigma)` for `sigma != 1`
/// * `u(c) = ln(c)` for `sigma == 1`
///
/// Returns `f64::NEG_INFINITY` for `c <= 0`.
pub fn crra_utility(c: f64, sigma: f64) -> f64 {
    if c <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if (sigma - 1.0).abs() < 1e-12 {
        c.ln()
    } else {
        c.powf(1.0 - sigma) / (1.0 - sigma)
    }
}

/// Constructs a linearly spaced grid of `n` points in `[lo, hi]`.
pub fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![lo];
    }
    let step = (hi - lo) / (n - 1) as f64;
    (0..n).map(|i| lo + i as f64 * step).collect()
}

/// Returns the index of the nearest grid point to `value`.
///
/// Assumes `grid` is sorted in ascending order.
pub fn nearest_index(grid: &[f64], value: f64) -> usize {
    if grid.is_empty() {
        return 0;
    }
    let mut best = 0;
    let mut best_dist = (grid[0] - value).abs();
    for (i, &g) in grid.iter().enumerate().skip(1) {
        let dist = (g - value).abs();
        if dist < best_dist {
            best = i;
            best_dist = dist;
        } else {
            // Grid is sorted, so once distance starts increasing we can stop
            break;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tauchen_rows_sum_to_one() {
        let result = tauchen(0.9, 0.1, 7, 3.0);
        for (i, row) in result.transition_matrix.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Row {} sums to {}, expected 1.0",
                i,
                sum
            );
        }
    }

    #[test]
    fn test_tauchen_grid_symmetric() {
        let result = tauchen(0.9, 0.1, 7, 3.0);
        let n = result.states.len();
        for i in 0..n / 2 {
            assert!(
                (result.states[i] + result.states[n - 1 - i]).abs() < 1e-10,
                "Grid not symmetric: {} vs {}",
                result.states[i],
                result.states[n - 1 - i]
            );
        }
    }

    #[test]
    fn test_crra_log_case() {
        let c = 2.5;
        let u = crra_utility(c, 1.0);
        assert!(
            (u - c.ln()).abs() < 1e-12,
            "CRRA(sigma=1) should equal ln(c)"
        );
    }

    #[test]
    fn test_crra_negative() {
        assert!(crra_utility(-1.0, 2.0) == f64::NEG_INFINITY);
        assert!(crra_utility(0.0, 2.0) == f64::NEG_INFINITY);
    }

    #[test]
    fn test_linspace_endpoints() {
        let grid = linspace(1.0, 5.0, 5);
        assert_eq!(grid.len(), 5);
        assert!((grid[0] - 1.0).abs() < 1e-12);
        assert!((grid[4] - 5.0).abs() < 1e-12);
        // Spacing should be uniform
        for i in 0..4 {
            assert!((grid[i + 1] - grid[i] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_nearest_index() {
        let grid = linspace(0.0, 10.0, 11); // [0, 1, 2, ..., 10]
        assert_eq!(nearest_index(&grid, 0.0), 0);
        assert_eq!(nearest_index(&grid, 3.2), 3);
        assert_eq!(nearest_index(&grid, 3.8), 4);
        assert_eq!(nearest_index(&grid, 10.0), 10);
        assert_eq!(nearest_index(&grid, -1.0), 0);
        assert_eq!(nearest_index(&grid, 11.0), 10);
    }
}
