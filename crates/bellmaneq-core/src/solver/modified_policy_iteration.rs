use std::collections::HashMap;

use crate::mdp::MDP;
use crate::operator::{bellman_operator, extract_policy};
use super::SolverResult;

/// Modified Policy Iteration solver.
///
/// A hybrid of Value Iteration and Policy Iteration. Truncates policy
/// evaluation after m sweeps before proceeding to policy improvement.
/// - m = 1: equivalent to Value Iteration
/// - m = ∞: equivalent to Policy Iteration
pub struct ModifiedPolicyIteration {
    /// Maximum number of outer iterations.
    pub max_iterations: usize,
    /// Number of policy evaluation sweeps m before truncation.
    pub eval_steps: usize,
    /// Convergence threshold.
    pub tolerance: f64,
    /// Discount factor γ.
    pub discount: f64,
}

impl ModifiedPolicyIteration {
    pub fn new(discount: f64, eval_steps: usize) -> Self {
        Self {
            max_iterations: 10_000,
            eval_steps,
            tolerance: 1e-10,
            discount,
        }
    }

    pub fn solve<M: MDP>(&self, mdp: &M) -> SolverResult<M::State, M::Action> {
        let states = mdp.states();

        let mut values: HashMap<M::State, f64> = HashMap::with_capacity(states.len());
        for s in &states {
            values.insert(s.clone(), 0.0);
        }

        let mut convergence_history = Vec::new();

        for iteration in 0..self.max_iterations {
            // Policy improvement: compute greedy policy with respect to V
            let policy = extract_policy(mdp, &values, self.discount);

            // Partial policy evaluation: iterate only m sweeps
            for _ in 0..self.eval_steps {
                for s in &states {
                    if mdp.is_terminal(s) {
                        continue;
                    }
                    let Some(action) = policy.get(s) else {
                        continue;
                    };
                    let new_value: f64 = mdp
                        .transitions(s, action)
                        .iter()
                        .map(|(prob, next_state, reward)| {
                            prob * (reward
                                + self.discount * values.get(next_state).unwrap_or(&0.0))
                        })
                        .sum();
                    values.insert(s.clone(), new_value);
                }
            }

            // Apply the Bellman operator once and check convergence
            let new_values = bellman_operator(mdp, &values, self.discount);
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
                let policy = extract_policy(mdp, &values, self.discount);
                return SolverResult {
                    values,
                    policy,
                    iterations: iteration + 1,
                    converged: true,
                    convergence_history,
                };
            }
        }

        let policy = extract_policy(mdp, &values, self.discount);
        SolverResult {
            values,
            policy,
            iterations: self.max_iterations,
            converged: false,
            convergence_history,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::value_iteration::ValueIteration;
    use crate::solver::policy_iteration::PolicyIteration;

    struct ThreeStateMDP;

    impl MDP for ThreeStateMDP {
        type State = usize;
        type Action = usize;

        fn states(&self) -> Vec<usize> {
            vec![0, 1, 2]
        }

        fn actions(&self, _state: &usize) -> Vec<usize> {
            vec![0, 1]
        }

        fn transitions(&self, s: &usize, a: &usize) -> Vec<(f64, usize, f64)> {
            match (s, a) {
                (0, 0) => vec![(0.7, 0, 1.0), (0.3, 1, 0.0)],
                (0, 1) => vec![(0.4, 1, 0.5), (0.6, 2, 0.2)],
                (1, 0) => vec![(0.5, 0, 0.0), (0.5, 2, 1.5)],
                (1, 1) => vec![(1.0, 1, 0.8)],
                (2, 0) => vec![(0.3, 0, 0.0), (0.7, 2, 0.5)],
                (2, 1) => vec![(0.6, 1, 0.3), (0.4, 2, 0.7)],
                _ => vec![],
            }
        }

        fn is_terminal(&self, _state: &usize) -> bool {
            false
        }
    }

    #[test]
    fn test_mpi_matches_vi_and_pi() {
        let mdp = ThreeStateMDP;
        let gamma = 0.95;

        let vi = ValueIteration::new(gamma).with_tolerance(1e-12).solve(&mdp);
        let pi = PolicyIteration::new(gamma).solve(&mdp);
        let mpi = ModifiedPolicyIteration::new(gamma, 5).solve(&mdp);

        for s in mdp.states() {
            let vi_v = vi.values[&s];
            let pi_v = pi.values[&s];
            let mpi_v = mpi.values[&s];

            assert!(
                (vi_v - mpi_v).abs() < 1e-8,
                "State {}: VI = {}, MPI = {}",
                s, vi_v, mpi_v
            );
            assert!(
                (pi_v - mpi_v).abs() < 1e-8,
                "State {}: PI = {}, MPI = {}",
                s, pi_v, mpi_v
            );
        }

        // Policies must also agree.
        for s in mdp.states() {
            assert_eq!(
                vi.policy[&s], mpi.policy[&s],
                "Policy mismatch at state {}",
                s
            );
        }
    }
}
