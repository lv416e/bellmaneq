use std::collections::HashMap;

use crate::mdp::MDP;
use crate::operator::{extract_policy, q_value};
use super::SolverResult;

/// Policy Iteration solver.
///
/// Alternates between policy evaluation (solving the Bellman expectation
/// equation) and policy improvement. Guaranteed to converge to the exact
/// optimal policy in a finite number of steps for finite MDPs.
pub struct PolicyIteration {
    /// Maximum number of outer iterations.
    pub max_iterations: usize,
    /// Maximum number of iterations for policy evaluation.
    pub eval_max_iterations: usize,
    /// Convergence threshold for policy evaluation.
    pub eval_tolerance: f64,
    /// Discount factor γ.
    pub discount: f64,
}

impl PolicyIteration {
    pub fn new(discount: f64) -> Self {
        Self {
            max_iterations: 1_000,
            eval_max_iterations: 10_000,
            eval_tolerance: 1e-12,
            discount,
        }
    }

    /// Policy evaluation: iteratively computes V^π for a fixed policy π.
    ///
    /// V^π(s) = Σ_{s'} P(s'|s,π(s)) * [R(s,π(s),s') + γ * V^π(s')]
    fn evaluate_policy<M: MDP>(
        &self,
        mdp: &M,
        policy: &HashMap<M::State, M::Action>,
        values: &mut HashMap<M::State, f64>,
    ) {
        let states = mdp.states();

        for _ in 0..self.eval_max_iterations {
            let mut delta = 0.0_f64;

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
                        prob * (reward + self.discount * values.get(next_state).unwrap_or(&0.0))
                    })
                    .sum();

                let old_value = *values.get(s).unwrap_or(&0.0);
                delta = delta.max((new_value - old_value).abs());
                values.insert(s.clone(), new_value);
            }

            if delta < self.eval_tolerance {
                break;
            }
        }
    }

    /// Runs Policy Iteration on the given MDP.
    pub fn solve<M: MDP>(&self, mdp: &M) -> SolverResult<M::State, M::Action> {
        let states = mdp.states();

        // V_0 = 0
        let mut values: HashMap<M::State, f64> = HashMap::with_capacity(states.len());
        for s in &states {
            values.insert(s.clone(), 0.0);
        }

        // Initial policy: select the first available action for each state
        let mut policy: HashMap<M::State, M::Action> = HashMap::new();
        for s in &states {
            if mdp.is_terminal(s) {
                continue;
            }
            let actions = mdp.actions(s);
            if let Some(a) = actions.into_iter().next() {
                policy.insert(s.clone(), a);
            }
        }

        let mut convergence_history = Vec::new();

        for iteration in 0..self.max_iterations {
            // 1. Policy evaluation: compute V^π
            self.evaluate_policy(mdp, &policy, &mut values);

            // 2. Policy improvement: π'(s) = argmax_a Q(s,a)
            let new_policy = extract_policy(mdp, &values, self.discount);

            // Check whether the policy has changed
            let mut policy_stable = true;
            let mut max_value_change = 0.0_f64;

            for s in &states {
                if mdp.is_terminal(s) {
                    continue;
                }
                let old_action = policy.get(s);
                let new_action = new_policy.get(s);
                if old_action != new_action {
                    policy_stable = false;
                }
                // Also record the magnitude of value change
                if let (Some(old_a), Some(new_a)) = (old_action, new_action) {
                    let q_old = q_value(mdp, &values, s, old_a, self.discount);
                    let q_new = q_value(mdp, &values, s, new_a, self.discount);
                    max_value_change = max_value_change.max((q_new - q_old).abs());
                }
            }

            convergence_history.push(max_value_change);
            policy = new_policy;

            if policy_stable {
                return SolverResult {
                    values,
                    policy,
                    iterations: iteration + 1,
                    converged: true,
                    convergence_history,
                };
            }
        }

        SolverResult {
            values,
            policy,
            iterations: self.max_iterations,
            converged: false,
            convergence_history,
        }
    }
}
