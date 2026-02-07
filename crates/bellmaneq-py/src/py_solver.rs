use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use bellmaneq_core::mdp::MDP;
use bellmaneq_core::solver::value_iteration::ValueIteration;
use bellmaneq_core::solver::policy_iteration::PolicyIteration;
use bellmaneq_core::solver::SolverResult;

/// A tabular MDP defined by NumPy arrays.
struct TabularMDP {
    n_states: usize,
    n_actions: usize,
    /// Reward matrix R(s,a) with shape (n_states, n_actions).
    rewards: Vec<Vec<f64>>,
    /// Transition probabilities P(s'|s,a) with shape (n_states, n_actions, n_states).
    transitions: Vec<Vec<Vec<f64>>>,
}

impl MDP for TabularMDP {
    type State = usize;
    type Action = usize;

    fn states(&self) -> Vec<usize> {
        (0..self.n_states).collect()
    }

    fn actions(&self, _state: &usize) -> Vec<usize> {
        (0..self.n_actions).collect()
    }

    fn transitions(&self, state: &usize, action: &usize) -> Vec<(f64, usize, f64)> {
        let reward = self.rewards[*state][*action];
        self.transitions[*state][*action]
            .iter()
            .enumerate()
            .filter(|(_, &p)| p > 0.0)
            .map(|(s_next, &p)| (p, s_next, reward))
            .collect()
    }

    fn is_terminal(&self, _state: &usize) -> bool {
        false
    }
}

/// Solver result exposed to Python.
#[pyclass]
pub struct PySolverResult {
    values: Vec<f64>,
    policy: Vec<i64>,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    converged: bool,
    convergence_history: Vec<f64>,
}

#[pymethods]
impl PySolverResult {
    fn get_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.values.clone().into_pyarray(py)
    }

    fn get_policy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.policy.clone().into_pyarray(py)
    }

    fn get_convergence_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.convergence_history.clone().into_pyarray(py)
    }
}

fn convert_result(result: SolverResult<usize, usize>, n_states: usize) -> PySolverResult {
    let mut values = vec![0.0; n_states];
    let mut policy = vec![-1i64; n_states];

    for (s, v) in &result.values {
        values[*s] = *v;
    }
    for (s, a) in &result.policy {
        policy[*s] = *a as i64;
    }

    PySolverResult {
        values,
        policy,
        iterations: result.iterations,
        converged: result.converged,
        convergence_history: result.convergence_history,
    }
}

/// Solves a discrete MDP using Value Iteration.
///
/// Args:
///     rewards: Reward matrix of shape (n_states, n_actions).
///     transitions: Transition probability tensor of shape (n_states, n_actions, n_states).
///     gamma: Discount factor.
///     tol: Convergence threshold (default: 1e-10).
///     max_iter: Maximum number of iterations (default: 10000).
#[pyfunction]
#[pyo3(signature = (rewards, transitions, gamma, tol=1e-10, max_iter=10000))]
fn solve_value_iteration(
    rewards: PyReadonlyArray2<'_, f64>,
    transitions: PyReadonlyArray1<'_, f64>,
    gamma: f64,
    tol: f64,
    max_iter: usize,
) -> PyResult<PySolverResult> {
    let r = rewards.as_array();
    let n_states = r.shape()[0];
    let n_actions = r.shape()[1];

    // transitions is a flat array of shape (n_states * n_actions * n_states)
    let t_raw = transitions.as_slice()?;
    if t_raw.len() != n_states * n_actions * n_states {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!(
                "transitions must have {} elements, got {}",
                n_states * n_actions * n_states,
                t_raw.len()
            ),
        ));
    }

    let mut rewards_vec = vec![vec![0.0; n_actions]; n_states];
    let mut trans_vec = vec![vec![vec![0.0; n_states]; n_actions]; n_states];

    for s in 0..n_states {
        for a in 0..n_actions {
            rewards_vec[s][a] = r[[s, a]];
            for sp in 0..n_states {
                trans_vec[s][a][sp] = t_raw[s * n_actions * n_states + a * n_states + sp];
            }
        }
    }

    let mdp = TabularMDP {
        n_states,
        n_actions,
        rewards: rewards_vec,
        transitions: trans_vec,
    };

    let solver = ValueIteration::new(gamma)
        .with_tolerance(tol)
        .with_max_iterations(max_iter);

    let result = solver.solve(&mdp);
    Ok(convert_result(result, n_states))
}

/// Solves a discrete MDP using Policy Iteration.
#[pyfunction]
#[pyo3(signature = (rewards, transitions, gamma))]
fn solve_policy_iteration(
    rewards: PyReadonlyArray2<'_, f64>,
    transitions: PyReadonlyArray1<'_, f64>,
    gamma: f64,
) -> PyResult<PySolverResult> {
    let r = rewards.as_array();
    let n_states = r.shape()[0];
    let n_actions = r.shape()[1];

    let t_raw = transitions.as_slice()?;
    if t_raw.len() != n_states * n_actions * n_states {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "transitions shape mismatch",
        ));
    }

    let mut rewards_vec = vec![vec![0.0; n_actions]; n_states];
    let mut trans_vec = vec![vec![vec![0.0; n_states]; n_actions]; n_states];

    for s in 0..n_states {
        for a in 0..n_actions {
            rewards_vec[s][a] = r[[s, a]];
            for sp in 0..n_states {
                trans_vec[s][a][sp] = t_raw[s * n_actions * n_states + a * n_states + sp];
            }
        }
    }

    let mdp = TabularMDP {
        n_states,
        n_actions,
        rewards: rewards_vec,
        transitions: trans_vec,
    };

    let solver = PolicyIteration::new(gamma);
    let result = solver.solve(&mdp);
    Ok(convert_result(result, n_states))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySolverResult>()?;
    m.add_function(wrap_pyfunction!(solve_value_iteration, m)?)?;
    m.add_function(wrap_pyfunction!(solve_policy_iteration, m)?)?;
    Ok(())
}
