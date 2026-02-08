use bellmaneq_core::mdp::MDP;

use crate::util::{crra_utility, linspace, nearest_index};

/// Result of solving the Cake Eating problem.
pub struct CakeEatingResult {
    /// Value function V(x) for each cake grid point.
    pub values: Vec<f64>,
    /// Optimal consumption c*(x) for each cake grid point.
    pub policy: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the solver converged.
    pub converged: bool,
    /// Convergence history: max|V_{k+1} - V_k| at each iteration.
    pub convergence_history: Vec<f64>,
}

/// Analytical value function for the Cake Eating problem with CRRA utility.
///
/// For σ = 1 (log): V(x) = A + ln(x)/(1-β)
///   where A = [ln(1-β^{1/σ}) + β·ln(β)/((1-β)·σ)] / (1-β)
/// For σ ≠ 1: V(x) = (1-β^{1/σ})^{-σ} · u(x)
fn analytical_cake_value(x: f64, discount: f64, risk_aversion: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let sigma = risk_aversion;
    let alpha = 1.0 - discount.powf(1.0 / sigma);
    if alpha <= 1e-15 {
        return f64::NEG_INFINITY;
    }

    if (sigma - 1.0).abs() < 1e-12 {
        let one_m_b = 1.0 - discount;
        let b = 1.0 / one_m_b;
        let a = (alpha.ln() + discount * b * discount.ln()) / one_m_b;
        a + b * x.ln()
    } else {
        alpha.powf(-sigma) * crra_utility(x, sigma)
    }
}

/// The Cake Eating Problem.
///
/// An agent has a cake of initial size x and must decide how much to
/// consume each period. The Bellman equation is:
///
///   V(x) = max_c { u(c) + β · V(x - c) }
///
/// This is the simplest infinite-horizon dynamic programming model:
/// deterministic transitions, one state variable, one control variable.
pub struct CakeEating {
    /// Discount factor β ∈ (0, 1).
    pub discount: f64,
    /// CRRA risk aversion parameter σ.
    pub risk_aversion: f64,
    /// Grid of cake sizes.
    cake_grid: Vec<f64>,
    /// Grid of possible consumption levels.
    consumption_grid: Vec<f64>,
}

impl CakeEating {
    /// Creates a new Cake Eating problem with default grid parameters.
    ///
    /// Defaults: `x_max = 1.0`, `n_cake = 50`, `n_consumption = 50`.
    pub fn new(discount: f64, risk_aversion: f64) -> Self {
        Self::with_grid(discount, risk_aversion, 1.0, 50, 50)
    }

    /// Creates a new Cake Eating problem with explicit grid parameters.
    pub fn with_grid(
        discount: f64,
        risk_aversion: f64,
        x_max: f64,
        n_cake: usize,
        n_consumption: usize,
    ) -> Self {
        let cake_grid = linspace(0.0, x_max, n_cake);
        let consumption_grid = linspace(0.0, x_max, n_consumption);
        Self {
            discount,
            risk_aversion,
            cake_grid,
            consumption_grid,
        }
    }

    /// Returns the cake size for a given state index.
    pub fn cake_size(&self, state: &usize) -> f64 {
        self.cake_grid[*state]
    }

    /// Returns the consumption level for a given action index.
    pub fn consumption(&self, action: &usize) -> f64 {
        self.consumption_grid[*action]
    }

    /// Returns the cake grid.
    pub fn cake_grid(&self) -> &[f64] {
        &self.cake_grid
    }

    /// Returns the consumption grid.
    pub fn consumption_grid(&self) -> &[f64] {
        &self.consumption_grid
    }

    /// Solves the Cake Eating problem using value iteration with an
    /// analytical warm-start and a fine internal consumption grid.
    ///
    /// This dedicated solver correctly handles the boundary condition
    /// V(0) = -∞ for σ ≥ 1, which the generic MDP solver cannot represent
    /// (it sets V(terminal) = 0, making "eat everything" artificially optimal).
    pub fn solve(&self, discount: f64, tol: f64, max_iter: usize) -> CakeEatingResult {
        let n = self.cake_grid.len();
        if n == 0 {
            return CakeEatingResult {
                values: vec![],
                policy: vec![],
                iterations: 0,
                converged: true,
                convergence_history: vec![],
            };
        }

        let x_max = *self.cake_grid.last().unwrap();

        // Use a fine internal consumption grid. This ensures that even at the
        // smallest cake state, there exist consumption choices small enough to
        // keep the remaining cake at a neighboring (non-zero) grid point.
        let n_fine = (n * 10).max(500);
        let fine_grid = linspace(0.0, x_max, n_fine);

        // --- Warm-start with the analytical solution ---
        let mut values = vec![0.0; n];
        for i in 1..n {
            values[i] = analytical_cake_value(self.cake_grid[i], discount, self.risk_aversion);
            // Guard against non-finite values from extreme parameters
            if !values[i].is_finite() {
                values[i] = -1e15;
            }
        }
        // V(0) should be -∞. We set it to a large negative finite value
        // that is significantly worse than any reachable state.
        if n > 1 {
            let v1 = values[1];
            values[0] = v1 - (v1.abs() + 1.0) * 10.0;
        }

        let mut policy_consumption = vec![0.0_f64; n];
        let mut history = Vec::new();
        let mut converged = false;
        let mut final_iter = max_iter;

        for iter in 0..max_iter {
            let mut new_values = vec![0.0; n];
            new_values[0] = values[0]; // V(0) stays fixed

            for i in 1..n {
                let x = self.cake_grid[i];
                let mut best_val = f64::NEG_INFINITY;
                let mut best_c = 0.0;

                for &c in &fine_grid {
                    if c < 1e-12 || c > x + 1e-12 {
                        continue;
                    }
                    let remaining = (x - c).max(0.0);
                    let next_idx = nearest_index(&self.cake_grid, remaining);
                    let val = crra_utility(c, self.risk_aversion) + discount * values[next_idx];
                    if val > best_val {
                        best_val = val;
                        best_c = c;
                    }
                }

                new_values[i] = best_val;
                policy_consumption[i] = best_c;
            }

            // Convergence check over non-zero cake states only
            let delta = (1..n)
                .map(|i| (new_values[i] - values[i]).abs())
                .fold(0.0_f64, f64::max);
            history.push(delta);
            values = new_values;

            if delta < tol {
                converged = true;
                final_iter = iter + 1;
                break;
            }
        }

        CakeEatingResult {
            values,
            policy: policy_consumption,
            iterations: final_iter,
            converged,
            convergence_history: history,
        }
    }
}

impl MDP for CakeEating {
    type State = usize;
    type Action = usize;

    fn states(&self) -> Vec<usize> {
        (0..self.cake_grid.len()).collect()
    }

    fn actions(&self, state: &usize) -> Vec<usize> {
        let x = self.cake_grid[*state];
        self.consumption_grid
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 1e-12 && c <= x + 1e-12)
            .map(|(i, _)| i)
            .collect()
    }

    fn transitions(&self, state: &usize, action: &usize) -> Vec<(f64, usize, f64)> {
        let x = self.cake_grid[*state];
        let c = self.consumption_grid[*action];
        let x_prime = (x - c).max(0.0);
        let reward = crra_utility(c, self.risk_aversion);
        let next_state = nearest_index(&self.cake_grid, x_prime);
        vec![(1.0, next_state, reward)]
    }

    fn is_terminal(&self, state: &usize) -> bool {
        *state == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converges() {
        let model = CakeEating::new(0.95, 1.0);
        let result = model.solve(0.95, 1e-10, 10000);
        assert!(result.converged, "Cake eating should converge");
    }

    #[test]
    fn test_value_monotone() {
        let model = CakeEating::with_grid(0.95, 1.0, 1.0, 30, 30);
        let result = model.solve(0.95, 1e-10, 10000);

        // V(x) should be non-decreasing in x (skip state 0)
        for i in 1..model.cake_grid.len() - 1 {
            assert!(
                result.values[i + 1] >= result.values[i] - 1e-6,
                "V({}) = {} > V({}) = {}",
                model.cake_grid[i + 1],
                result.values[i + 1],
                model.cake_grid[i],
                result.values[i],
            );
        }
    }

    #[test]
    fn test_consumption_monotone() {
        let model = CakeEating::with_grid(0.95, 1.0, 1.0, 30, 30);
        let result = model.solve(0.95, 1e-10, 10000);

        // c*(x) should be non-decreasing in x (skip states 0 and 1)
        for i in 2..model.cake_grid.len() - 1 {
            assert!(
                result.policy[i + 1] >= result.policy[i] - 1e-8,
                "c*({}) = {} > c*({}) = {}",
                model.cake_grid[i + 1],
                result.policy[i + 1],
                model.cake_grid[i],
                result.policy[i],
            );
        }
    }

    #[test]
    fn test_policy_fraction() {
        // For log utility (σ=1) with β=0.95, the analytical c*(x) = 0.05·x.
        // The grid-discretized solution should be approximately correct.
        let model = CakeEating::with_grid(0.95, 1.0, 1.0, 50, 50);
        let result = model.solve(0.95, 1e-10, 10000);

        // Check that c*(x)/x is much closer to 0.05 than to 1.0 (the old bug)
        let n = model.cake_grid.len();
        let x = model.cake_grid[n - 1];
        let c = result.policy[n - 1];
        let ratio = c / x;
        let analytical = 0.05;
        assert!(
            (ratio - analytical).abs() < 0.05,
            "At x={}, c/x = {} but analytical = {}",
            x,
            ratio,
            analytical
        );
    }
}
