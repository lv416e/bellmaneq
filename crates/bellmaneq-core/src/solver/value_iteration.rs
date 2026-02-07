use std::collections::HashMap;

use crate::mdp::MDP;
use crate::operator::{bellman_operator, extract_policy};
use super::SolverResult;

/// Value Iteration solver.
///
/// Repeatedly applies the Bellman optimality operator T until convergence
/// to the fixed point V*.
/// V_{k+1} = TV_k
///
/// Convergence rate: ||V_k - V*|| ≤ γ^k / (1-γ) * ||V_1 - V_0||
pub struct ValueIteration {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Convergence threshold (converged when max|V_{k+1} - V_k| < tolerance).
    pub tolerance: f64,
    /// Discount factor γ ∈ [0, 1).
    pub discount: f64,
}

impl ValueIteration {
    pub fn new(discount: f64) -> Self {
        Self {
            max_iterations: 10_000,
            tolerance: 1e-10,
            discount,
        }
    }

    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Runs Value Iteration on the given MDP.
    pub fn solve<M: MDP>(&self, mdp: &M) -> SolverResult<M::State, M::Action> {
        let states = mdp.states();

        // Initialize V_0 = 0
        let mut values: HashMap<M::State, f64> = HashMap::with_capacity(states.len());
        for s in &states {
            values.insert(s.clone(), 0.0);
        }

        let mut convergence_history = Vec::new();
        let mut converged = false;

        for iteration in 0..self.max_iterations {
            // V_{k+1} = T(V_k)
            let new_values = bellman_operator(mdp, &values, self.discount);

            // δ = max_s |V_{k+1}(s) - V_k(s)|
            let delta = states
                .iter()
                .map(|s| {
                    let old = values.get(s).unwrap_or(&0.0);
                    let new = new_values.get(s).unwrap_or(&0.0);
                    (new - old).abs()
                })
                .fold(0.0_f64, f64::max);

            convergence_history.push(delta);
            values = new_values;

            if delta < self.tolerance {
                converged = true;
                // iteration is 0-indexed, so add 1
                let policy = extract_policy(mdp, &values, self.discount);
                return SolverResult {
                    values,
                    policy,
                    iterations: iteration + 1,
                    converged,
                    convergence_history,
                };
            }
        }

        let policy = extract_policy(mdp, &values, self.discount);
        SolverResult {
            values,
            policy,
            iterations: self.max_iterations,
            converged,
            convergence_history,
        }
    }
}
