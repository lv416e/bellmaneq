use std::collections::HashMap;

use rayon::prelude::*;

use crate::mdp::MDP;

/// Applies the Bellman optimality operator T.
///
/// (TV)(s) = max_a [ Σ_{s'} P(s'|s,a) * (R(s,a,s') + γ * V(s')) ]
///
/// By the Banach contraction mapping theorem, T is a contraction for γ < 1,
/// guaranteeing the existence of a unique fixed point V* = TV*.
pub fn bellman_operator<M: MDP>(
    mdp: &M,
    values: &HashMap<M::State, f64>,
    discount: f64,
) -> HashMap<M::State, f64> {
    let states = mdp.states();
    let mut new_values = HashMap::with_capacity(states.len());

    for s in &states {
        if mdp.is_terminal(s) {
            new_values.insert(s.clone(), 0.0);
            continue;
        }

        let actions = mdp.actions(s);
        if actions.is_empty() {
            new_values.insert(s.clone(), 0.0);
            continue;
        }

        let best_value = actions
            .iter()
            .map(|a| q_value(mdp, values, s, a, discount))
            .fold(f64::NEG_INFINITY, f64::max);

        new_values.insert(s.clone(), best_value);
    }

    new_values
}

/// Computes the Q-value (action-value) for a state-action pair.
///
/// Q(s,a) = Σ_{s'} P(s'|s,a) * (R(s,a,s') + γ * V(s'))
pub fn q_value<M: MDP>(
    mdp: &M,
    values: &HashMap<M::State, f64>,
    state: &M::State,
    action: &M::Action,
    discount: f64,
) -> f64 {
    mdp.transitions(state, action)
        .iter()
        .map(|(prob, next_state, reward)| {
            prob * (reward + discount * values.get(next_state).unwrap_or(&0.0))
        })
        .sum()
}

/// Extracts the greedy policy from a value function.
///
/// π*(s) = argmax_a Q(s,a)
pub fn extract_policy<M: MDP>(
    mdp: &M,
    values: &HashMap<M::State, f64>,
    discount: f64,
) -> HashMap<M::State, M::Action> {
    let mut policy = HashMap::new();

    for s in mdp.states() {
        if mdp.is_terminal(&s) {
            continue;
        }

        let actions = mdp.actions(&s);
        if actions.is_empty() {
            continue;
        }

        let best_action = actions
            .into_iter()
            .max_by(|a1, a2| {
                let q1 = q_value(mdp, values, &s, a1, discount);
                let q2 = q_value(mdp, values, &s, a2, discount);
                q1.partial_cmp(&q2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();

        policy.insert(s, best_action);
    }

    policy
}

/// Parallel version of the Bellman optimality operator.
///
/// Evaluates states concurrently using rayon. Beneficial when the state
/// space is large enough for the parallelism overhead to pay off.
pub fn bellman_operator_par<M: MDP + Sync>(
    mdp: &M,
    values: &HashMap<M::State, f64>,
    discount: f64,
) -> HashMap<M::State, f64>
where
    M::State: Send + Sync,
    M::Action: Send + Sync,
{
    let states = mdp.states();
    states
        .par_iter()
        .map(|s| {
            if mdp.is_terminal(s) {
                return (s.clone(), 0.0);
            }

            let actions = mdp.actions(s);
            if actions.is_empty() {
                return (s.clone(), 0.0);
            }

            let best_value = actions
                .iter()
                .map(|a| q_value(mdp, values, s, a, discount))
                .fold(f64::NEG_INFINITY, f64::max);

            (s.clone(), best_value)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small stochastic MDP used across operator tests.
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
    fn test_contraction_mapping() {
        // The Bellman operator T is a γ-contraction in the sup-norm:
        //   ||TV1 - TV2||_∞ ≤ γ · ||V1 - V2||_∞
        let mdp = ThreeStateMDP;
        let gamma = 0.9;

        let v1: HashMap<usize, f64> = [(0, 1.0), (1, 2.0), (2, 0.5)].into();
        let v2: HashMap<usize, f64> = [(0, 3.0), (1, 0.0), (2, 1.5)].into();

        let tv1 = bellman_operator(&mdp, &v1, gamma);
        let tv2 = bellman_operator(&mdp, &v2, gamma);

        let dist_before: f64 = mdp
            .states()
            .iter()
            .map(|s| (v1[s] - v2[s]).abs())
            .fold(0.0_f64, f64::max);

        let dist_after: f64 = mdp
            .states()
            .iter()
            .map(|s| (tv1[s] - tv2[s]).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            dist_after <= gamma * dist_before + 1e-12,
            "Contraction violated: ||TV1 - TV2|| = {}, γ · ||V1 - V2|| = {}",
            dist_after,
            gamma * dist_before
        );
    }

    #[test]
    fn test_fixed_point() {
        // V* = TV*: the optimal value function is a fixed point of T.
        use crate::solver::value_iteration::ValueIteration;

        let mdp = ThreeStateMDP;
        let gamma = 0.9;

        let solver = ValueIteration::new(gamma).with_tolerance(1e-12);
        let result = solver.solve(&mdp);

        let tv_star = bellman_operator(&mdp, &result.values, gamma);

        let max_diff: f64 = mdp
            .states()
            .iter()
            .map(|s| (tv_star[s] - result.values[s]).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            max_diff < 1e-10,
            "Fixed point violated: max|TV* - V*| = {}",
            max_diff
        );
    }

    #[test]
    fn test_parallel_matches_sequential() {
        let mdp = ThreeStateMDP;
        let gamma = 0.9;
        let values: HashMap<usize, f64> = [(0, 1.0), (1, 2.0), (2, 0.5)].into();

        let seq = bellman_operator(&mdp, &values, gamma);
        let par = bellman_operator_par(&mdp, &values, gamma);

        for s in mdp.states() {
            assert!(
                (seq[&s] - par[&s]).abs() < 1e-12,
                "State {}: seq = {}, par = {}",
                s,
                seq[&s],
                par[&s]
            );
        }
    }
}
