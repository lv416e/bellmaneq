pub mod value_iteration;
pub mod policy_iteration;
pub mod modified_policy_iteration;

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

/// Stores the result of an MDP solver.
#[derive(Debug, Clone)]
pub struct SolverResult<S: Hash + Eq, A: Hash + Eq> {
    /// Optimal value function V*.
    pub values: HashMap<S, f64>,
    /// Optimal policy π*.
    pub policy: HashMap<S, A>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the solver converged within the tolerance.
    pub converged: bool,
    /// Convergence history: max|V_{k+1} - V_k| at each iteration.
    pub convergence_history: Vec<f64>,
}
