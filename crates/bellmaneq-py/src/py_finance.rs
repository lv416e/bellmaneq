use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1};

use bellmaneq_finance::american_option;

#[pyclass]
pub struct PyAmericanOptionResult {
    inner: american_option::AmericanOptionResult,
}

#[pymethods]
impl PyAmericanOptionResult {
    #[getter]
    fn price(&self) -> f64 {
        self.inner.price
    }

    fn get_exercise_boundary<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.exercise_boundary.clone().into_pyarray(py)
    }

    fn get_time_steps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.time_steps.clone().into_pyarray(py)
    }

    fn get_stock_prices_at_maturity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.stock_prices_at_maturity.clone().into_pyarray(py)
    }
}

/// Prices an American option using the binomial model.
///
/// Bellman equation: V(i,j) = max(intrinsic value, discounted expectation).
#[pyfunction]
#[pyo3(signature = (spot, strike, rate, volatility, maturity, steps, is_call=false))]
fn price_american_option(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    maturity: f64,
    steps: usize,
    is_call: bool,
) -> PyAmericanOptionResult {
    let result = american_option::price_american_option(
        spot, strike, rate, volatility, maturity, steps, is_call,
    );
    PyAmericanOptionResult { inner: result }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAmericanOptionResult>()?;
    m.add_function(wrap_pyfunction!(price_american_option, m)?)?;
    Ok(())
}
