use pyo3::prelude::*;

// -----------------------------
// Basic statistics
// -----------------------------

#[pyfunction]
fn mean(values: Vec<f64>) -> PyResult<f64> {
    if values.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("values cannot be empty"));
    }

    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

#[pyfunction]
fn variance(values: Vec<f64>, sample: bool) -> PyResult<f64> {
    if values.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "at least two values are required",
        ));
    }

    let avg = values.iter().sum::<f64>() / values.len() as f64;
    let squared_diffs: f64 = values.iter().map(|x| (x - avg).powi(2)).sum();

    let denominator = if sample {
        (values.len() - 1) as f64
    } else {
        values.len() as f64
    };

    Ok(squared_diffs / denominator)
}

#[pyfunction]
fn std_dev(values: Vec<f64>, sample: bool) -> PyResult<f64> {
    Ok(variance(values, sample)?.sqrt())
}

#[pyfunction]
fn simple_returns(prices: Vec<f64>) -> PyResult<Vec<f64>> {
    if prices.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "at least two prices are required",
        ));
    }

    if prices.iter().any(|&p| p <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "prices must be positive",
        ));
    }

    Ok(prices.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect())
}

#[pyfunction]
fn log_returns(prices: Vec<f64>) -> PyResult<Vec<f64>> {
    if prices.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "at least two prices are required",
        ));
    }

    if prices.iter().any(|&p| p <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "prices must be positive",
        ));
    }

    Ok(prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect())
}

#[pyfunction]
fn covariance(x: Vec<f64>, y: Vec<f64>, sample: bool) -> PyResult<f64> {
    if x.len() != y.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "x and y must have the same length",
        ));
    }

    if x.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "at least two values are required",
        ));
    }

    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;

    let cov_sum: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - mean_x) * (b - mean_y))
        .sum();

    let denominator = if sample {
        (x.len() - 1) as f64
    } else {
        x.len() as f64
    };

    Ok(cov_sum / denominator)
}

#[pyfunction]
fn correlation(x: Vec<f64>, y: Vec<f64>) -> PyResult<f64> {
    let cov = covariance(x.clone(), y.clone(), true)?;
    let sx = std_dev(x, true)?;
    let sy = std_dev(y, true)?;

    if sx == 0.0 || sy == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "standard deviation cannot be zero",
        ));
    }

    Ok(cov / (sx * sy))
}

// -----------------------------
// Black-Scholes
// -----------------------------

fn validate_option_inputs(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<()> {
    if spot <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("spot must be positive"));
    }
    if strike <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("strike must be positive"));
    }
    if volatility <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("volatility must be positive"));
    }
    if time <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("time must be positive"));
    }
    if !rate.is_finite() {
        return Err(pyo3::exceptions::PyValueError::new_err("rate must be finite"));
    }

    Ok(())
}

fn normal_cdf(x: f64) -> f64 {
    let a1 = 0.319381530;
    let a2 = -0.356563782;
    let a3 = 1.781477937;
    let a4 = -1.821255978;
    let a5 = 1.330274429;

    let k = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = a1 * k
        + a2 * k.powi(2)
        + a3 * k.powi(3)
        + a4 * k.powi(4)
        + a5 * k.powi(5);

    let pdf = (-(x * x) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cdf = 1.0 - pdf * poly;

    if x < 0.0 {
        1.0 - cdf
    } else {
        cdf
    }
}

fn d1_value(spot: f64, strike: f64, rate: f64, volatility: f64, time: f64) -> f64 {
    ((spot / strike).ln() + (rate + 0.5 * volatility.powi(2)) * time)
        / (volatility * time.sqrt())
}

fn d2_value(spot: f64, strike: f64, rate: f64, volatility: f64, time: f64) -> f64 {
    d1_value(spot, strike, rate, volatility, time) - volatility * time.sqrt()
}

#[pyfunction]
fn black_scholes_call(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;

    let d1 = d1_value(spot, strike, rate, volatility, time);
    let d2 = d2_value(spot, strike, rate, volatility, time);

    Ok(spot * normal_cdf(d1) - strike * (-rate * time).exp() * normal_cdf(d2))
}

#[pyfunction]
fn black_scholes_put(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;

    let d1 = d1_value(spot, strike, rate, volatility, time);
    let d2 = d2_value(spot, strike, rate, volatility, time);

    Ok(strike * (-rate * time).exp() * normal_cdf(-d2) - spot * normal_cdf(-d1))
}

#[pyfunction]
fn black_scholes_d1(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;
    Ok(d1_value(spot, strike, rate, volatility, time))
}

#[pyfunction]
fn black_scholes_d2(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;
    Ok(d2_value(spot, strike, rate, volatility, time))
}

// -----------------------------
// Module export
// -----------------------------

#[pymodule]
fn larp_quantmath(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(variance, m)?)?;
    m.add_function(wrap_pyfunction!(std_dev, m)?)?;
    m.add_function(wrap_pyfunction!(simple_returns, m)?)?;
    m.add_function(wrap_pyfunction!(log_returns, m)?)?;
    m.add_function(wrap_pyfunction!(covariance, m)?)?;
    m.add_function(wrap_pyfunction!(correlation, m)?)?;

    m.add_function(wrap_pyfunction!(black_scholes_call, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_put, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_d1, m)?)?;
    m.add_function(wrap_pyfunction!(black_scholes_d2, m)?)?;

    Ok(())
}