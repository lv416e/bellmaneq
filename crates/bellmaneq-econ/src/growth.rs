use bellmaneq_core::mdp::MDP;

use crate::util::{crra_utility, linspace, nearest_index, tauchen};

/// Stochastic Neoclassical Growth Model (Brock-Mirman).
///
/// A representative agent chooses consumption to maximize discounted
/// lifetime utility subject to a Cobb-Douglas production technology
/// with stochastic productivity shocks.
///
/// Bellman equation:
///
///   V(k, z) = max_c { u(c) + β · E[V(k', z') | z] }
///
/// where:
///   k' = z · k^α + (1 - δ) · k - c
///   z follows a discretized AR(1): ln(z') = ρ · ln(z) + ε
///
/// State: (capital, productivity shock).
/// Action: consumption.
pub struct GrowthModel {
    /// Discount factor β ∈ (0, 1).
    pub discount: f64,
    /// CRRA risk aversion parameter σ.
    pub risk_aversion: f64,
    /// Capital share (Cobb-Douglas exponent) α.
    pub alpha: f64,
    /// Depreciation rate δ.
    pub delta: f64,
    /// Capital grid.
    capital_grid: Vec<f64>,
    /// Productivity shock grid (from Tauchen).
    productivity_grid: Vec<f64>,
    /// Productivity transition matrix P[i][j] = Pr(z_j | z_i).
    productivity_transitions: Vec<Vec<f64>>,
    /// Consumption grid.
    consumption_grid: Vec<f64>,
}

impl GrowthModel {
    /// Creates a new Growth Model with standard calibration.
    ///
    /// Defaults: α=0.36, δ=0.1, ρ=0.9, σ_z=0.02, n_k=50, n_z=7.
    pub fn new(discount: f64, risk_aversion: f64) -> Self {
        Self::build(discount, risk_aversion, 0.36, 0.1, 0.9, 0.02, 50, 7)
    }

    /// Full-parameter constructor.
    pub fn build(
        discount: f64,
        risk_aversion: f64,
        alpha: f64,
        delta: f64,
        rho: f64,
        sigma_z: f64,
        n_k: usize,
        n_z: usize,
    ) -> Self {
        let tauchen_result = tauchen(rho, sigma_z, n_z, 3.0);
        // Exponentiate the log-productivity grid so z > 0
        let productivity_grid: Vec<f64> =
            tauchen_result.states.iter().map(|&s| s.exp()).collect();
        let productivity_transitions = tauchen_result.transition_matrix;

        // Deterministic steady-state capital: k_ss = (α/(1/β - 1 + δ))^{1/(1-α)}
        let k_ss = (alpha / (1.0 / discount - 1.0 + delta)).powf(1.0 / (1.0 - alpha));
        let k_min = 0.1 * k_ss;
        let k_max = 2.5 * k_ss;
        let capital_grid = linspace(k_min, k_max, n_k);

        // Maximum possible output (highest z, highest k)
        let z_max = *productivity_grid.last().unwrap_or(&1.0);
        let y_max = z_max * k_max.powf(alpha) + (1.0 - delta) * k_max;
        let n_c = n_k; // Same resolution as capital grid
        let consumption_grid = linspace(1e-6, y_max, n_c);

        Self {
            discount,
            risk_aversion,
            alpha,
            delta,
            capital_grid,
            productivity_grid,
            productivity_transitions,
            consumption_grid,
        }
    }

    /// Builder: set production parameters.
    pub fn with_production(mut self, alpha: f64, delta: f64) -> Self {
        self = Self::build(
            self.discount,
            self.risk_aversion,
            alpha,
            delta,
            0.9,
            0.02,
            self.capital_grid.len(),
            self.productivity_grid.len(),
        );
        self
    }

    /// Builder: set capital grid resolution.
    pub fn with_capital_grid(mut self, n_k: usize) -> Self {
        self = Self::build(
            self.discount,
            self.risk_aversion,
            self.alpha,
            self.delta,
            0.9,
            0.02,
            n_k,
            self.productivity_grid.len(),
        );
        self
    }

    /// Builder: set productivity process parameters.
    pub fn with_productivity(mut self, rho: f64, sigma: f64, n_z: usize) -> Self {
        self = Self::build(
            self.discount,
            self.risk_aversion,
            self.alpha,
            self.delta,
            rho,
            sigma,
            self.capital_grid.len(),
            n_z,
        );
        self
    }

    /// Decode a flat state index into (capital_index, productivity_index).
    fn decode_state(&self, state: &usize) -> (usize, usize) {
        let n_z = self.productivity_grid.len();
        (*state / n_z, *state % n_z)
    }

    /// Encode (capital_index, productivity_index) into a flat state index.
    fn encode_state(&self, k_idx: usize, z_idx: usize) -> usize {
        k_idx * self.productivity_grid.len() + z_idx
    }

    /// Returns the capital grid.
    pub fn capital_grid(&self) -> &[f64] {
        &self.capital_grid
    }

    /// Returns the productivity grid.
    pub fn productivity_grid(&self) -> &[f64] {
        &self.productivity_grid
    }

    /// Returns the consumption grid.
    pub fn consumption_grid(&self) -> &[f64] {
        &self.consumption_grid
    }

    /// Number of capital grid points.
    pub fn n_k(&self) -> usize {
        self.capital_grid.len()
    }

    /// Number of productivity grid points.
    pub fn n_z(&self) -> usize {
        self.productivity_grid.len()
    }

    /// Total number of states.
    pub fn n_states(&self) -> usize {
        self.n_k() * self.n_z()
    }
}

impl MDP for GrowthModel {
    type State = usize;
    type Action = usize;

    fn states(&self) -> Vec<usize> {
        (0..self.n_states()).collect()
    }

    fn actions(&self, state: &usize) -> Vec<usize> {
        let (k_idx, z_idx) = self.decode_state(state);
        let k = self.capital_grid[k_idx];
        let z = self.productivity_grid[z_idx];
        let y = z * k.powf(self.alpha) + (1.0 - self.delta) * k;
        let k_min = self.capital_grid[0];

        self.consumption_grid
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 1e-12 && y - c >= k_min - 1e-12)
            .map(|(i, _)| i)
            .collect()
    }

    fn transitions(&self, state: &usize, action: &usize) -> Vec<(f64, usize, f64)> {
        let (k_idx, z_idx) = self.decode_state(state);
        let k = self.capital_grid[k_idx];
        let z = self.productivity_grid[z_idx];
        let c = self.consumption_grid[*action];

        let k_prime = z * k.powf(self.alpha) + (1.0 - self.delta) * k - c;
        let k_prime_idx = nearest_index(&self.capital_grid, k_prime);
        let reward = crra_utility(c, self.risk_aversion);

        self.productivity_transitions[z_idx]
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 1e-15)
            .map(|(z_prime_idx, &p)| {
                let next_state = self.encode_state(k_prime_idx, z_prime_idx);
                (p, next_state, reward)
            })
            .collect()
    }

    fn is_terminal(&self, _state: &usize) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bellmaneq_core::solver::value_iteration::ValueIteration;

    fn small_model() -> GrowthModel {
        GrowthModel::build(0.95, 2.0, 0.36, 0.1, 0.9, 0.02, 20, 5)
    }

    #[test]
    fn test_converges() {
        let model = small_model();
        let solver = ValueIteration::new(model.discount).with_tolerance(1e-8);
        let result = solver.solve(&model);
        assert!(result.converged, "Growth model should converge");
    }

    #[test]
    fn test_value_increasing_in_capital() {
        let model = small_model();
        let solver = ValueIteration::new(model.discount).with_tolerance(1e-8);
        let result = solver.solve(&model);

        // Fix z at the middle productivity level
        let z_idx = model.n_z() / 2;
        for k_idx in 0..model.n_k() - 1 {
            let s1 = model.encode_state(k_idx, z_idx);
            let s2 = model.encode_state(k_idx + 1, z_idx);
            assert!(
                result.values[&s2] >= result.values[&s1] - 1e-6,
                "V(k={:.3}, z) = {:.4} should be <= V(k={:.3}, z) = {:.4}",
                model.capital_grid[k_idx],
                result.values[&s1],
                model.capital_grid[k_idx + 1],
                result.values[&s2],
            );
        }
    }

    #[test]
    fn test_value_increasing_in_productivity() {
        let model = small_model();
        let solver = ValueIteration::new(model.discount).with_tolerance(1e-8);
        let result = solver.solve(&model);

        // Fix k at the middle capital level
        let k_idx = model.n_k() / 2;
        for z_idx in 0..model.n_z() - 1 {
            let s1 = model.encode_state(k_idx, z_idx);
            let s2 = model.encode_state(k_idx, z_idx + 1);
            assert!(
                result.values[&s2] >= result.values[&s1] - 1e-6,
                "V(k, z={:.3}) = {:.4} should be <= V(k, z={:.3}) = {:.4}",
                model.productivity_grid[z_idx],
                result.values[&s1],
                model.productivity_grid[z_idx + 1],
                result.values[&s2],
            );
        }
    }

    #[test]
    fn test_consumption_positive() {
        let model = small_model();
        let solver = ValueIteration::new(model.discount).with_tolerance(1e-8);
        let result = solver.solve(&model);

        for (&s, &a) in &result.policy {
            let c = model.consumption_grid[a];
            assert!(
                c > 0.0,
                "Consumption should be positive at state {}",
                s
            );
        }
    }
}
