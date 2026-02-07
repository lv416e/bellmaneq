use std::fmt::Debug;
use std::hash::Hash;

/// Trait defining a Markov Decision Process (MDP).
///
/// Provides all information required to solve the Bellman equation:
/// V(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]
pub trait MDP {
    /// State type.
    type State: Clone + Hash + Eq + Debug;
    /// Action type.
    type Action: Clone + Hash + Eq + Debug;

    /// Enumerates all states in the MDP.
    fn states(&self) -> Vec<Self::State>;

    /// Returns the list of available actions from a given state.
    fn actions(&self, state: &Self::State) -> Vec<Self::Action>;

    /// Returns transition tuples for a state-action pair.
    /// Each element is (probability, next_state, immediate_reward).
    /// Probabilities must sum to 1.0.
    fn transitions(
        &self,
        state: &Self::State,
        action: &Self::Action,
    ) -> Vec<(f64, Self::State, f64)>;

    /// Returns whether the given state is terminal.
    fn is_terminal(&self, state: &Self::State) -> bool;
}
