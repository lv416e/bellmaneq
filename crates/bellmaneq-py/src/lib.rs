mod py_solver;
mod py_games;
mod py_finance;
mod py_econ;

use pyo3::prelude::*;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    py_solver::register(m)?;
    py_games::register(m)?;
    py_finance::register(m)?;
    py_econ::register(m)?;
    Ok(())
}
