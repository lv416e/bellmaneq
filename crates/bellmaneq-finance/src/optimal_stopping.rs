/// Generic optimal stopping framework.
///
/// An optimal stopping problem asks: at each time step, should the agent
/// "stop" (collect a payoff) or "continue" (wait for a potentially better
/// opportunity)?  The Bellman equation is:
///
///   V(s, t) = max( stop_payoff(s, t),
///                  −holding_cost(s, t) + δ · Σ P(s'|s,t) V(s', t+1) )
///
/// This module provides a trait capturing the problem structure, a generic
/// backward-induction solver, and the Secretary Problem as a concrete
/// example.
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// Defines an optimal stopping problem over a finite horizon.
pub trait OptimalStopping {
    type State: Clone + Hash + Eq + Debug;

    /// Immediate payoff received upon stopping in `state` at `time`.
    fn stop_payoff(&self, state: &Self::State, time: usize) -> f64;

    /// Per-period cost of continuing (often zero).
    fn holding_cost(&self, state: &Self::State, time: usize) -> f64;

    /// Transition probabilities from `state` at `time`: `(prob, next_state)`.
    fn transitions(&self, state: &Self::State, time: usize) -> Vec<(f64, Self::State)>;

    /// All possible states at the given time step.
    fn states_at_time(&self, time: usize) -> Vec<Self::State>;

    /// Number of decision periods (0-indexed: times 0 .. horizon).
    fn horizon(&self) -> usize;

    /// Per-period discount factor δ.
    fn discount(&self) -> f64;
}

/// Result of solving an optimal stopping problem.
pub struct OptimalStoppingResult<S: Hash + Eq> {
    /// `values[t]` maps state -> V(state, t).
    pub values: Vec<HashMap<S, f64>>,
    /// `stop_flags[t]` maps state -> true if stopping is optimal.
    pub stop_flags: Vec<HashMap<S, bool>>,
}

/// Solves an optimal stopping problem via backward induction.
pub fn solve_optimal_stopping<P: OptimalStopping>(
    problem: &P,
) -> OptimalStoppingResult<P::State> {
    let horizon = problem.horizon();
    let mut values: Vec<HashMap<P::State, f64>> = vec![HashMap::new(); horizon + 1];
    let mut stop_flags: Vec<HashMap<P::State, bool>> = vec![HashMap::new(); horizon + 1];

    // At the terminal time, the agent must stop.
    for state in problem.states_at_time(horizon) {
        let payoff = problem.stop_payoff(&state, horizon);
        values[horizon].insert(state.clone(), payoff);
        stop_flags[horizon].insert(state, true);
    }

    // Backward induction.
    for t in (0..horizon).rev() {
        for state in problem.states_at_time(t) {
            let stop_val = problem.stop_payoff(&state, t);
            let hold_cost = problem.holding_cost(&state, t);

            let continuation: f64 = problem
                .transitions(&state, t)
                .iter()
                .map(|(prob, next_s)| prob * values[t + 1].get(next_s).unwrap_or(&0.0))
                .sum();

            let continue_val = -hold_cost + problem.discount() * continuation;
            let optimal_stop = stop_val >= continue_val;
            let value = stop_val.max(continue_val);

            values[t].insert(state.clone(), value);
            stop_flags[t].insert(state, optimal_stop);
        }
    }

    OptimalStoppingResult { values, stop_flags }
}

// ---------------------------------------------------------------------------
// Example: The Secretary Problem (Best Choice Problem)
// ---------------------------------------------------------------------------

/// The classical Secretary (Best Choice) Problem.
///
/// N candidates are interviewed in sequence. After each interview the
/// agent observes whether the current candidate is the best seen so far
/// and must immediately accept or reject. The goal is to maximise the
/// probability of hiring the overall best candidate.
///
/// The optimal policy is to reject the first ⌊N/e⌋ candidates and then
/// accept the next one who is the best so far.
pub struct SecretaryProblem {
    pub n_candidates: usize,
}

impl OptimalStopping for SecretaryProblem {
    type State = (usize, bool); // (candidates_seen, is_current_best_so_far)

    fn stop_payoff(&self, state: &Self::State, _time: usize) -> f64 {
        let (seen, is_best) = *state;
        if is_best {
            // Probability that the best among the first `seen` is the
            // overall best out of all N.
            seen as f64 / self.n_candidates as f64
        } else {
            0.0
        }
    }

    fn holding_cost(&self, _state: &Self::State, _time: usize) -> f64 {
        0.0
    }

    fn transitions(&self, state: &Self::State, _time: usize) -> Vec<(f64, Self::State)> {
        let (seen, _) = *state;
        let next_seen = seen + 1;
        if next_seen > self.n_candidates {
            return vec![];
        }
        // The (next_seen)-th candidate is the best so far with probability
        // 1 / next_seen.
        let p_best = 1.0 / next_seen as f64;
        vec![
            (p_best, (next_seen, true)),
            (1.0 - p_best, (next_seen, false)),
        ]
    }

    fn states_at_time(&self, time: usize) -> Vec<Self::State> {
        let seen = time + 1;
        if seen > self.n_candidates {
            return vec![];
        }
        vec![(seen, true), (seen, false)]
    }

    fn horizon(&self) -> usize {
        // Decision times 0 .. n-1 (one per candidate).
        self.n_candidates - 1
    }

    fn discount(&self) -> f64 {
        1.0 // No discounting.
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secretary_threshold() {
        // The optimal stopping threshold should be approximately N/e.
        let n = 100;
        let problem = SecretaryProblem { n_candidates: n };
        let result = solve_optimal_stopping(&problem);

        // Find the first time the agent would stop on a best-so-far candidate.
        let mut threshold = n;
        for t in 0..n {
            let state = (t + 1, true);
            if let Some(&should_stop) = result.stop_flags[t].get(&state) {
                if should_stop {
                    threshold = t + 1;
                    break;
                }
            }
        }

        let expected = (n as f64 / std::f64::consts::E).round() as usize;
        assert!(
            (threshold as i64 - expected as i64).unsigned_abs() <= 2,
            "Threshold {} should be near N/e = {}",
            threshold,
            expected
        );
    }

    #[test]
    fn test_secretary_value() {
        // For large N the optimal win probability converges to 1/e ≈ 0.368.
        let n = 200;
        let problem = SecretaryProblem { n_candidates: n };
        let result = solve_optimal_stopping(&problem);

        // The initial state is "1 candidate seen, and they are the best so far".
        let initial_value = *result.values[0].get(&(1, true)).unwrap_or(&0.0);
        let expected = 1.0 / std::f64::consts::E;
        assert!(
            (initial_value - expected).abs() < 0.02,
            "Initial value {:.4} should be near 1/e = {:.4}",
            initial_value,
            expected
        );
    }
}
