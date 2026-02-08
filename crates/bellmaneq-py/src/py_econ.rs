use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2};

use bellmaneq_core::solver::value_iteration::ValueIteration;
use bellmaneq_econ::{cake_eating, growth, income_fluctuation, mccall};

/// Convert a flat Vec into a Vec<Vec> with the given number of columns.
fn to_vec2(flat: &[f64], n_rows: usize, n_cols: usize) -> Vec<Vec<f64>> {
    (0..n_rows)
        .map(|i| flat[i * n_cols..(i + 1) * n_cols].to_vec())
        .collect()
}

// ---------------------------------------------------------------------------
// Cake Eating
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PyCakeEatingResult {
    values: Vec<f64>,
    policy: Vec<f64>,
    cake_grid: Vec<f64>,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    converged: bool,
    convergence_history: Vec<f64>,
}

#[pymethods]
impl PyCakeEatingResult {
    fn get_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.values.clone().into_pyarray(py)
    }

    fn get_policy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.policy.clone().into_pyarray(py)
    }

    fn get_cake_grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.cake_grid.clone().into_pyarray(py)
    }

    fn get_convergence_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.convergence_history.clone().into_pyarray(py)
    }
}

#[pyfunction]
#[pyo3(signature = (
    discount=0.95, risk_aversion=1.0, x_max=1.0,
    n_cake=50, n_consumption=50, tol=1e-10, max_iter=10000
))]
fn solve_cake_eating(
    discount: f64,
    risk_aversion: f64,
    x_max: f64,
    n_cake: usize,
    n_consumption: usize,
    tol: f64,
    max_iter: usize,
) -> PyCakeEatingResult {
    let model = cake_eating::CakeEating::with_grid(
        discount, risk_aversion, x_max, n_cake, n_consumption,
    );
    let result = model.solve(discount, tol, max_iter);

    PyCakeEatingResult {
        values: result.values,
        policy: result.policy,
        cake_grid: model.cake_grid().to_vec(),
        iterations: result.iterations,
        converged: result.converged,
        convergence_history: result.convergence_history,
    }
}

// ---------------------------------------------------------------------------
// McCall Job Search
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PyMcCallResult {
    values: Vec<f64>,
    policy: Vec<i64>,
    wage_grid: Vec<f64>,
    #[pyo3(get)]
    reservation_wage: f64,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    converged: bool,
    convergence_history: Vec<f64>,
}

#[pymethods]
impl PyMcCallResult {
    fn get_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.values.clone().into_pyarray(py)
    }

    fn get_policy<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.policy.clone().into_pyarray(py)
    }

    fn get_wage_grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.wage_grid.clone().into_pyarray(py)
    }

    fn get_convergence_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.convergence_history.clone().into_pyarray(py)
    }
}

#[pyfunction]
#[pyo3(signature = (
    discount=0.95, unemployment_comp=25.0, w_min=10.0, w_max=60.0,
    n_wages=50, tol=1e-10, max_iter=10000
))]
fn solve_mccall(
    discount: f64,
    unemployment_comp: f64,
    w_min: f64,
    w_max: f64,
    n_wages: usize,
    tol: f64,
    max_iter: usize,
) -> PyMcCallResult {
    let model = mccall::McCall::with_uniform_wages(
        discount, unemployment_comp, w_min, w_max, n_wages,
    );
    let solver = ValueIteration::new(discount)
        .with_tolerance(tol)
        .with_max_iterations(max_iter);
    let result = solver.solve(&model);

    let wage_grid = model.wage_grid().to_vec();
    let n = wage_grid.len();
    let mut values = vec![0.0; n];
    let mut policy = vec![0i64; n];

    for s in 0..n {
        values[s] = *result.values.get(&s).unwrap_or(&0.0);
        if let Some(&a) = result.policy.get(&s) {
            policy[s] = a as i64;
        }
    }

    let reservation_wage = model.reservation_wage(&result.policy);

    PyMcCallResult {
        values,
        policy,
        wage_grid,
        reservation_wage,
        iterations: result.iterations,
        converged: result.converged,
        convergence_history: result.convergence_history,
    }
}

// ---------------------------------------------------------------------------
// Stochastic Growth Model
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PyGrowthResult {
    values: Vec<f64>,
    policy: Vec<f64>,
    capital_grid: Vec<f64>,
    productivity_grid: Vec<f64>,
    n_k: usize,
    n_z: usize,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    converged: bool,
    convergence_history: Vec<f64>,
}

#[pymethods]
impl PyGrowthResult {
    /// Value function as 2D array (n_k, n_z).
    fn get_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let vec2 = to_vec2(&self.values, self.n_k, self.n_z);
        Ok(PyArray2::from_vec2(py, &vec2)?)
    }

    /// Consumption policy as 2D array (n_k, n_z).
    fn get_policy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let vec2 = to_vec2(&self.policy, self.n_k, self.n_z);
        Ok(PyArray2::from_vec2(py, &vec2)?)
    }

    fn get_capital_grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.capital_grid.clone().into_pyarray(py)
    }

    fn get_productivity_grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.productivity_grid.clone().into_pyarray(py)
    }

    fn get_convergence_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.convergence_history.clone().into_pyarray(py)
    }
}

#[pyfunction]
#[pyo3(signature = (
    discount=0.95, risk_aversion=2.0, alpha=0.36, delta=0.1,
    rho=0.9, sigma_z=0.02, n_k=50, n_z=7, tol=1e-10, max_iter=10000
))]
fn solve_growth(
    discount: f64,
    risk_aversion: f64,
    alpha: f64,
    delta: f64,
    rho: f64,
    sigma_z: f64,
    n_k: usize,
    n_z: usize,
    tol: f64,
    max_iter: usize,
) -> PyGrowthResult {
    let model = growth::GrowthModel::build(
        discount, risk_aversion, alpha, delta, rho, sigma_z, n_k, n_z,
    );
    let solver = ValueIteration::new(discount)
        .with_tolerance(tol)
        .with_max_iterations(max_iter);
    let result = solver.solve(&model);

    let total = model.n_states();
    let mut values = vec![0.0; total];
    let mut policy = vec![0.0; total];

    for s in 0..total {
        values[s] = *result.values.get(&s).unwrap_or(&0.0);
        if let Some(&a) = result.policy.get(&s) {
            policy[s] = model.consumption_grid()[a];
        }
    }

    PyGrowthResult {
        values,
        policy,
        capital_grid: model.capital_grid().to_vec(),
        productivity_grid: model.productivity_grid().to_vec(),
        n_k: model.n_k(),
        n_z: model.n_z(),
        iterations: result.iterations,
        converged: result.converged,
        convergence_history: result.convergence_history,
    }
}

// ---------------------------------------------------------------------------
// Income Fluctuation
// ---------------------------------------------------------------------------

#[pyclass]
pub struct PyIncomeFluctuationResult {
    values: Vec<f64>,
    policy: Vec<f64>,
    savings_policy: Vec<f64>,
    asset_grid: Vec<f64>,
    income_grid: Vec<f64>,
    n_a: usize,
    n_y: usize,
    #[pyo3(get)]
    iterations: usize,
    #[pyo3(get)]
    converged: bool,
    convergence_history: Vec<f64>,
}

#[pymethods]
impl PyIncomeFluctuationResult {
    /// Value function as 2D array (n_a, n_y).
    fn get_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let vec2 = to_vec2(&self.values, self.n_a, self.n_y);
        Ok(PyArray2::from_vec2(py, &vec2)?)
    }

    /// Consumption policy as 2D array (n_a, n_y).
    fn get_policy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let vec2 = to_vec2(&self.policy, self.n_a, self.n_y);
        Ok(PyArray2::from_vec2(py, &vec2)?)
    }

    /// Savings policy a' as 2D array (n_a, n_y).
    fn get_savings_policy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let vec2 = to_vec2(&self.savings_policy, self.n_a, self.n_y);
        Ok(PyArray2::from_vec2(py, &vec2)?)
    }

    fn get_asset_grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.asset_grid.clone().into_pyarray(py)
    }

    fn get_income_grid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.income_grid.clone().into_pyarray(py)
    }

    fn get_convergence_history<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.convergence_history.clone().into_pyarray(py)
    }
}

#[pyfunction]
#[pyo3(signature = (
    discount=0.95, risk_aversion=2.0, interest_rate=0.03, borrowing_limit=0.0,
    rho=0.9, sigma_y=0.1, n_a=50, n_y=7, tol=1e-10, max_iter=10000
))]
fn solve_income_fluctuation(
    discount: f64,
    risk_aversion: f64,
    interest_rate: f64,
    borrowing_limit: f64,
    rho: f64,
    sigma_y: f64,
    n_a: usize,
    n_y: usize,
    tol: f64,
    max_iter: usize,
) -> PyIncomeFluctuationResult {
    let model = income_fluctuation::IncomeFluctuation::build(
        discount, risk_aversion, interest_rate, borrowing_limit, rho, sigma_y, n_a, n_y,
    );
    let solver = ValueIteration::new(discount)
        .with_tolerance(tol)
        .with_max_iterations(max_iter);
    let result = solver.solve(&model);

    let total = model.n_states();
    let mut values = vec![0.0; total];
    let mut policy = vec![0.0; total];
    let mut savings_policy = vec![0.0; total];

    for s in 0..total {
        values[s] = *result.values.get(&s).unwrap_or(&0.0);
        if let Some(&a_idx) = result.policy.get(&s) {
            let c = model.consumption_grid()[a_idx];
            policy[s] = c;

            let n_y_inner = model.n_y();
            let a_idx_s = s / n_y_inner;
            let y_idx_s = s % n_y_inner;
            let a = model.asset_grid()[a_idx_s];
            let y = model.income_grid()[y_idx_s];
            savings_policy[s] = (1.0 + interest_rate) * a + y - c;
        }
    }

    PyIncomeFluctuationResult {
        values,
        policy,
        savings_policy,
        asset_grid: model.asset_grid().to_vec(),
        income_grid: model.income_grid().to_vec(),
        n_a: model.n_a(),
        n_y: model.n_y(),
        iterations: result.iterations,
        converged: result.converged,
        convergence_history: result.convergence_history,
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCakeEatingResult>()?;
    m.add_class::<PyMcCallResult>()?;
    m.add_class::<PyGrowthResult>()?;
    m.add_class::<PyIncomeFluctuationResult>()?;
    m.add_function(wrap_pyfunction!(solve_cake_eating, m)?)?;
    m.add_function(wrap_pyfunction!(solve_mccall, m)?)?;
    m.add_function(wrap_pyfunction!(solve_growth, m)?)?;
    m.add_function(wrap_pyfunction!(solve_income_fluctuation, m)?)?;
    Ok(())
}
