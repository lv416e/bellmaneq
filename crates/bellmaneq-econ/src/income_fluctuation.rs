use bellmaneq_core::mdp::MDP;

use crate::util::{crra_utility, linspace, nearest_index, tauchen};

/// Income Fluctuation Problem (Bewley/Aiyagari).
///
/// A household receives stochastic income and chooses consumption subject
/// to a borrowing constraint. This is the building block of heterogeneous
/// agent models (Aiyagari 1994, Huggett 1993).
///
/// Bellman equation:
///
///   V(a, y) = max_c { u(c) + β · E[V(a', y') | y] }
///
/// where:
///   a' = (1 + r) · a + y - c
///   a' >= -b   (borrowing constraint)
///   y follows a discretized AR(1) Markov chain
///
/// State: (assets, income).
/// Action: consumption.
pub struct IncomeFluctuation {
    /// Discount factor β ∈ (0, 1).
    pub discount: f64,
    /// CRRA risk aversion parameter σ.
    pub risk_aversion: f64,
    /// Risk-free interest rate r.
    pub interest_rate: f64,
    /// Borrowing limit b >= 0 (agent cannot hold debt below -b).
    pub borrowing_limit: f64,
    /// Asset grid.
    asset_grid: Vec<f64>,
    /// Income grid (from Tauchen).
    income_grid: Vec<f64>,
    /// Income transition matrix P[i][j] = Pr(y_j | y_i).
    income_transitions: Vec<Vec<f64>>,
    /// Consumption grid.
    consumption_grid: Vec<f64>,
}

impl IncomeFluctuation {
    /// Creates a new Income Fluctuation problem with standard calibration.
    ///
    /// Defaults: r=0.03, b=0, ρ=0.9, σ_y=0.1, n_a=50, n_y=7.
    pub fn new(discount: f64, risk_aversion: f64) -> Self {
        Self::build(discount, risk_aversion, 0.03, 0.0, 0.9, 0.1, 50, 7)
    }

    /// Full-parameter constructor.
    pub fn build(
        discount: f64,
        risk_aversion: f64,
        interest_rate: f64,
        borrowing_limit: f64,
        rho: f64,
        sigma_y: f64,
        n_a: usize,
        n_y: usize,
    ) -> Self {
        let tauchen_result = tauchen(rho, sigma_y, n_y, 3.0);
        // Exponentiate the log-income grid so y > 0
        let income_grid: Vec<f64> = tauchen_result.states.iter().map(|&s| s.exp()).collect();
        let income_transitions = tauchen_result.transition_matrix;

        let a_min = -borrowing_limit;
        // Set a_max to allow enough room for savings
        let y_max = *income_grid.last().unwrap_or(&1.0);
        let a_max = 4.0 * y_max / interest_rate.max(0.01);
        let asset_grid = linspace(a_min, a_max, n_a);

        // Max consumption = max cash on hand
        let c_max = (1.0 + interest_rate) * a_max + y_max;
        let n_c = n_a;
        let consumption_grid = linspace(1e-6, c_max, n_c);

        Self {
            discount,
            risk_aversion,
            interest_rate,
            borrowing_limit,
            asset_grid,
            income_grid,
            income_transitions,
            consumption_grid,
        }
    }

    /// Builder: set interest rate.
    pub fn with_interest_rate(self, r: f64) -> Self {
        Self::build(
            self.discount,
            self.risk_aversion,
            r,
            self.borrowing_limit,
            0.9,
            0.1,
            self.asset_grid.len(),
            self.income_grid.len(),
        )
    }

    /// Builder: set borrowing limit.
    pub fn with_borrowing_limit(self, b: f64) -> Self {
        Self::build(
            self.discount,
            self.risk_aversion,
            self.interest_rate,
            b,
            0.9,
            0.1,
            self.asset_grid.len(),
            self.income_grid.len(),
        )
    }

    /// Builder: set income process parameters.
    pub fn with_income_process(self, rho: f64, sigma: f64, n_y: usize) -> Self {
        Self::build(
            self.discount,
            self.risk_aversion,
            self.interest_rate,
            self.borrowing_limit,
            rho,
            sigma,
            self.asset_grid.len(),
            n_y,
        )
    }

    /// Builder: set asset grid resolution.
    pub fn with_asset_grid_size(self, n_a: usize) -> Self {
        Self::build(
            self.discount,
            self.risk_aversion,
            self.interest_rate,
            self.borrowing_limit,
            0.9,
            0.1,
            n_a,
            self.income_grid.len(),
        )
    }

    /// Decode a flat state index into (asset_index, income_index).
    fn decode_state(&self, state: &usize) -> (usize, usize) {
        let n_y = self.income_grid.len();
        (*state / n_y, *state % n_y)
    }

    /// Encode (asset_index, income_index) into a flat state index.
    fn encode_state(&self, a_idx: usize, y_idx: usize) -> usize {
        a_idx * self.income_grid.len() + y_idx
    }

    /// Returns the asset grid.
    pub fn asset_grid(&self) -> &[f64] {
        &self.asset_grid
    }

    /// Returns the income grid.
    pub fn income_grid(&self) -> &[f64] {
        &self.income_grid
    }

    /// Returns the consumption grid.
    pub fn consumption_grid(&self) -> &[f64] {
        &self.consumption_grid
    }

    /// Number of asset grid points.
    pub fn n_a(&self) -> usize {
        self.asset_grid.len()
    }

    /// Number of income grid points.
    pub fn n_y(&self) -> usize {
        self.income_grid.len()
    }

    /// Total number of states.
    pub fn n_states(&self) -> usize {
        self.n_a() * self.n_y()
    }
}

impl MDP for IncomeFluctuation {
    type State = usize;
    type Action = usize;

    fn states(&self) -> Vec<usize> {
        (0..self.n_states()).collect()
    }

    fn actions(&self, state: &usize) -> Vec<usize> {
        let (a_idx, y_idx) = self.decode_state(state);
        let a = self.asset_grid[a_idx];
        let y = self.income_grid[y_idx];
        let cash_on_hand = (1.0 + self.interest_rate) * a + y;
        let a_min = self.asset_grid[0]; // = -borrowing_limit
        let c_max = cash_on_hand - a_min;

        self.consumption_grid
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 1e-12 && c <= c_max + 1e-12)
            .map(|(i, _)| i)
            .collect()
    }

    fn transitions(&self, state: &usize, action: &usize) -> Vec<(f64, usize, f64)> {
        let (a_idx, y_idx) = self.decode_state(state);
        let a = self.asset_grid[a_idx];
        let y = self.income_grid[y_idx];
        let c = self.consumption_grid[*action];

        let a_prime = (1.0 + self.interest_rate) * a + y - c;
        let a_prime_idx = nearest_index(&self.asset_grid, a_prime);
        let reward = crra_utility(c, self.risk_aversion);

        self.income_transitions[y_idx]
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 1e-15)
            .map(|(y_prime_idx, &p)| {
                let next_state = self.encode_state(a_prime_idx, y_prime_idx);
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

    fn small_model() -> IncomeFluctuation {
        IncomeFluctuation::build(0.95, 2.0, 0.03, 0.0, 0.9, 0.1, 20, 5)
    }

    #[test]
    fn test_converges() {
        let model = small_model();
        let solver = ValueIteration::new(model.discount).with_tolerance(1e-8);
        let result = solver.solve(&model);
        assert!(
            result.converged,
            "Income fluctuation model should converge"
        );
    }

    #[test]
    fn test_value_increasing_in_assets() {
        let model = small_model();
        let solver = ValueIteration::new(model.discount).with_tolerance(1e-8);
        let result = solver.solve(&model);

        // Fix y at the middle income level
        let y_idx = model.n_y() / 2;
        for a_idx in 0..model.n_a() - 1 {
            let s1 = model.encode_state(a_idx, y_idx);
            let s2 = model.encode_state(a_idx + 1, y_idx);
            assert!(
                result.values[&s2] >= result.values[&s1] - 1e-6,
                "V(a={:.3}, y) = {:.4} should be <= V(a={:.3}, y) = {:.4}",
                model.asset_grid[a_idx],
                result.values[&s1],
                model.asset_grid[a_idx + 1],
                result.values[&s2],
            );
        }
    }

    #[test]
    fn test_borrowing_constraint_binds() {
        // With tight borrowing constraint (b=0), agents at low assets
        // should choose the lowest feasible consumption
        let model = IncomeFluctuation::build(0.95, 2.0, 0.03, 0.0, 0.9, 0.1, 20, 5);
        let solver = ValueIteration::new(model.discount).with_tolerance(1e-8);
        let result = solver.solve(&model);

        // At the lowest asset level, savings a' should be near a_min
        let y_idx = 0; // lowest income
        let s = model.encode_state(0, y_idx);
        let a = model.asset_grid[0];
        let y = model.income_grid[y_idx];
        let c = model.consumption_grid[result.policy[&s]];
        let a_prime = (1.0 + model.interest_rate) * a + y - c;

        // a' should be close to a_min (the borrowing constraint)
        let a_min = model.asset_grid[0];
        assert!(
            (a_prime - a_min).abs() < model.asset_grid[1] - model.asset_grid[0] + 1e-6,
            "At low assets/income, borrowing constraint should nearly bind: a'={:.4}, a_min={:.4}",
            a_prime,
            a_min,
        );
    }
}
