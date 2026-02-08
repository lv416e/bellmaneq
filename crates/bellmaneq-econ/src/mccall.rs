use std::collections::HashMap;

use bellmaneq_core::mdp::MDP;

use crate::util::linspace;

/// The McCall (1970) Job Search Model.
///
/// An unemployed worker samples wage offers from a known distribution
/// each period and must decide whether to accept (receive the wage forever)
/// or reject (receive unemployment compensation and sample again).
///
/// Bellman equation:
///
///   V(w) = max { w/(1-β),  c + β · E[V(w')] }
///
/// The optimal policy has a reservation wage w*: accept iff w >= w*.
pub struct McCall {
    /// Discount factor β ∈ (0, 1).
    pub discount: f64,
    /// Unemployment compensation per period.
    pub unemployment_comp: f64,
    /// Grid of wage offers.
    wage_grid: Vec<f64>,
    /// Probability of each wage offer (sums to 1).
    wage_probs: Vec<f64>,
}

impl McCall {
    /// Creates a new McCall model with uniform wage distribution.
    ///
    /// Defaults: `w_min = 10`, `w_max = 60`, `n_wages = 50`.
    pub fn new(discount: f64, unemployment_comp: f64) -> Self {
        Self::with_uniform_wages(discount, unemployment_comp, 10.0, 60.0, 50)
    }

    /// Creates a McCall model with a uniform wage distribution on `[w_min, w_max]`.
    pub fn with_uniform_wages(
        discount: f64,
        unemployment_comp: f64,
        w_min: f64,
        w_max: f64,
        n_wages: usize,
    ) -> Self {
        let wage_grid = linspace(w_min, w_max, n_wages);
        let p = 1.0 / n_wages as f64;
        let wage_probs = vec![p; n_wages];
        Self {
            discount,
            unemployment_comp,
            wage_grid,
            wage_probs,
        }
    }

    /// Creates a McCall model with a custom wage distribution.
    pub fn with_custom_wages(
        discount: f64,
        unemployment_comp: f64,
        wage_grid: Vec<f64>,
        wage_probs: Vec<f64>,
    ) -> Self {
        Self {
            discount,
            unemployment_comp,
            wage_grid,
            wage_probs,
        }
    }

    /// Returns the wage for a given state index (non-terminal).
    pub fn wage(&self, state: &usize) -> f64 {
        self.wage_grid[*state]
    }

    /// Returns the wage grid.
    pub fn wage_grid(&self) -> &[f64] {
        &self.wage_grid
    }

    /// Number of wage states (excludes the terminal absorbing state).
    pub fn n_wages(&self) -> usize {
        self.wage_grid.len()
    }

    /// Computes the reservation wage from the solved policy.
    ///
    /// The reservation wage w* is the lowest wage at which acceptance is optimal.
    /// Returns `f64::INFINITY` if rejection is always optimal.
    pub fn reservation_wage(&self, policy: &HashMap<usize, usize>) -> f64 {
        for (i, &w) in self.wage_grid.iter().enumerate() {
            if let Some(&action) = policy.get(&i) {
                if action == 1 {
                    return w;
                }
            }
        }
        f64::INFINITY
    }
}

impl MDP for McCall {
    type State = usize;
    type Action = usize;

    fn states(&self) -> Vec<usize> {
        // 0..n_wages are wage-offer states; n_wages is the absorbing terminal state
        (0..=self.wage_grid.len()).collect()
    }

    fn actions(&self, state: &usize) -> Vec<usize> {
        if self.is_terminal(state) {
            vec![]
        } else {
            vec![0, 1] // 0 = reject, 1 = accept
        }
    }

    fn transitions(&self, state: &usize, action: &usize) -> Vec<(f64, usize, f64)> {
        let n = self.wage_grid.len();

        match action {
            1 => {
                // Accept: get perpetuity w/(1-β) as immediate reward, go to terminal
                let w = self.wage_grid[*state];
                let perpetuity = w / (1.0 - self.discount);
                vec![(1.0, n, perpetuity)]
            }
            _ => {
                // Reject: get unemployment comp, draw new wage next period
                self.wage_probs
                    .iter()
                    .enumerate()
                    .filter(|(_, &p)| p > 1e-15)
                    .map(|(w_idx, &p)| (p, w_idx, self.unemployment_comp))
                    .collect()
            }
        }
    }

    fn is_terminal(&self, state: &usize) -> bool {
        *state == self.wage_grid.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bellmaneq_core::solver::value_iteration::ValueIteration;

    #[test]
    fn test_converges() {
        let model = McCall::new(0.95, 25.0);
        let solver = ValueIteration::new(model.discount);
        let result = solver.solve(&model);
        assert!(result.converged, "McCall model should converge");
    }

    #[test]
    fn test_reservation_wage_exists() {
        let model = McCall::new(0.95, 25.0);
        let solver = ValueIteration::new(model.discount);
        let result = solver.solve(&model);

        let w_star = model.reservation_wage(&result.policy);
        assert!(
            w_star < f64::INFINITY,
            "Reservation wage should exist (not all reject)"
        );
        assert!(w_star > 0.0, "Reservation wage should be positive");
    }

    #[test]
    fn test_reservation_wage_increases_with_comp() {
        let m1 = McCall::with_uniform_wages(0.95, 20.0, 10.0, 60.0, 50);
        let m2 = McCall::with_uniform_wages(0.95, 35.0, 10.0, 60.0, 50);

        let s1 = ValueIteration::new(m1.discount).solve(&m1);
        let s2 = ValueIteration::new(m2.discount).solve(&m2);

        let w1 = m1.reservation_wage(&s1.policy);
        let w2 = m2.reservation_wage(&s2.policy);

        assert!(
            w2 >= w1,
            "Higher unemployment comp ({}) should raise reservation wage: w1={}, w2={}",
            35.0,
            w1,
            w2
        );
    }

    #[test]
    fn test_value_increasing() {
        let model = McCall::new(0.95, 25.0);
        let solver = ValueIteration::new(model.discount);
        let result = solver.solve(&model);

        // V(w) should be non-decreasing in w
        for i in 0..model.n_wages() - 1 {
            let v_curr = result.values[&i];
            let v_next = result.values[&(i + 1)];
            assert!(
                v_next >= v_curr - 1e-8,
                "V(w={}) = {} > V(w={}) = {}",
                model.wage(&(i + 1)),
                v_next,
                model.wage(&i),
                v_curr,
            );
        }
    }
}
