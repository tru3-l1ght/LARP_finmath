use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
// Black-Scholes helpers
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

fn normal_pdf(x: f64) -> f64 {
    (-(x * x) / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt()
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

    let cdf = 1.0 - normal_pdf(x) * poly;

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

// -----------------------------
// Black-Scholes pricing
// -----------------------------

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
// Greeks
// -----------------------------

#[pyfunction]
fn delta_call(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;
    let d1 = d1_value(spot, strike, rate, volatility, time);
    Ok(normal_cdf(d1))
}

#[pyfunction]
fn delta_put(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;
    let d1 = d1_value(spot, strike, rate, volatility, time);
    Ok(normal_cdf(d1) - 1.0)
}

#[pyfunction]
fn gamma(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;
    let d1 = d1_value(spot, strike, rate, volatility, time);
    Ok(normal_pdf(d1) / (spot * volatility * time.sqrt()))
}

#[pyfunction]
fn vega(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;
    let d1 = d1_value(spot, strike, rate, volatility, time);
    Ok(spot * normal_pdf(d1) * time.sqrt())
}

#[pyfunction]
fn theta_call(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;

    let d1 = d1_value(spot, strike, rate, volatility, time);
    let d2 = d2_value(spot, strike, rate, volatility, time);

    let first_term = -(spot * normal_pdf(d1) * volatility) / (2.0 * time.sqrt());
    let second_term = rate * strike * (-rate * time).exp() * normal_cdf(d2);

    Ok(first_term - second_term)
}

#[pyfunction]
fn theta_put(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;

    let d1 = d1_value(spot, strike, rate, volatility, time);
    let d2 = d2_value(spot, strike, rate, volatility, time);

    let first_term = -(spot * normal_pdf(d1) * volatility) / (2.0 * time.sqrt());
    let second_term = rate * strike * (-rate * time).exp() * normal_cdf(-d2);

    Ok(first_term + second_term)
}

#[pyfunction]
fn rho_call(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;
    let d2 = d2_value(spot, strike, rate, volatility, time);
    Ok(strike * time * (-rate * time).exp() * normal_cdf(d2))
}

#[pyfunction]
fn rho_put(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;
    let d2 = d2_value(spot, strike, rate, volatility, time);
    Ok(-strike * time * (-rate * time).exp() * normal_cdf(-d2))
}

// -----------------------------
// Monte Carlo option pricing
// -----------------------------

fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.r#gen::<f64>().max(1e-12);
    let u2: f64 = rng.r#gen::<f64>();

    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[pyfunction]
fn monte_carlo_call(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
    simulations: usize,
    seed: u64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;

    if simulations == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "simulations must be greater than zero",
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let drift = (rate - 0.5 * volatility.powi(2)) * time;
    let diffusion = volatility * time.sqrt();

    let mut payoff_sum = 0.0;

    for _ in 0..simulations {
        let z = standard_normal(&mut rng);
        let final_price = spot * (drift + diffusion * z).exp();
        let payoff = (final_price - strike).max(0.0);
        payoff_sum += payoff;
    }

    Ok((-rate * time).exp() * payoff_sum / simulations as f64)
}

#[pyfunction]
fn monte_carlo_put(
    spot: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    time: f64,
    simulations: usize,
    seed: u64,
) -> PyResult<f64> {
    validate_option_inputs(spot, strike, rate, volatility, time)?;

    if simulations == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "simulations must be greater than zero",
        ));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let drift = (rate - 0.5 * volatility.powi(2)) * time;
    let diffusion = volatility * time.sqrt();

    let mut payoff_sum = 0.0;

    for _ in 0..simulations {
        let z = standard_normal(&mut rng);
        let final_price = spot * (drift + diffusion * z).exp();
        let payoff = (strike - final_price).max(0.0);
        payoff_sum += payoff;
    }

    Ok((-rate * time).exp() * payoff_sum / simulations as f64)
}

// -----------------------------
// Portfolio and risk metrics
// -----------------------------

#[pyfunction]
fn portfolio_return(expected_returns: Vec<f64>, weights: Vec<f64>) -> PyResult<f64> {
    if expected_returns.len() != weights.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected_returns and weights must have the same length",
        ));
    }

    if expected_returns.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "inputs cannot be empty",
        ));
    }

    Ok(expected_returns
        .iter()
        .zip(weights.iter())
        .map(|(r, w)| r * w)
        .sum())
}

#[pyfunction]
fn portfolio_volatility(covariance_matrix: Vec<Vec<f64>>, weights: Vec<f64>) -> PyResult<f64> {
    let n = weights.len();

    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "weights cannot be empty",
        ));
    }

    if covariance_matrix.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "covariance matrix row count must match weights length",
        ));
    }

    for row in covariance_matrix.iter() {
        if row.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "covariance matrix must be square",
            ));
        }
    }

    let mut variance = 0.0;

    for i in 0..n {
        for j in 0..n {
            variance += weights[i] * weights[j] * covariance_matrix[i][j];
        }
    }

    Ok(variance.sqrt())
}

#[pyfunction]
fn sharpe_ratio(portfolio_return: f64, risk_free_rate: f64, portfolio_volatility: f64) -> PyResult<f64> {
    if portfolio_volatility <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "portfolio_volatility must be positive",
        ));
    }

    Ok((portfolio_return - risk_free_rate) / portfolio_volatility)
}

#[pyfunction]
fn historical_var(returns: Vec<f64>, confidence_level: f64) -> PyResult<f64> {
    if returns.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "returns cannot be empty",
        ));
    }

    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "confidence_level must be between 0 and 1",
        ));
    }

    let mut sorted_returns = returns;
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let percentile = 1.0 - confidence_level;
    let index = ((sorted_returns.len() as f64) * percentile).floor() as usize;
    let capped_index = index.min(sorted_returns.len() - 1);

    Ok(-sorted_returns[capped_index])
}

#[pyfunction]
fn max_drawdown(prices: Vec<f64>) -> PyResult<f64> {
    if prices.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "prices cannot be empty",
        ));
    }

    if prices.iter().any(|&p| p <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "prices must be positive",
        ));
    }

    let mut peak = prices[0];
    let mut max_dd = 0.0;

    for price in prices {
        if price > peak {
            peak = price;
        }

        let drawdown = (peak - price) / peak;

        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }

    Ok(max_dd)
}

// -----------------------------
// Implied volatility
// -----------------------------

fn implied_volatility_bisection(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    time: f64,
    is_call: bool,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<f64> {
    if market_price <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "market_price must be positive",
        ));
    }

    if tolerance <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tolerance must be positive",
        ));
    }

    if max_iterations == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max_iterations must be greater than zero",
        ));
    }

    if spot <= 0.0 || strike <= 0.0 || time <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "spot, strike, and time must be positive",
        ));
    }

    let mut low = 1e-6;
    let mut high = 5.0;

    for _ in 0..max_iterations {
        let mid = (low + high) / 2.0;

        let price = if is_call {
            black_scholes_call(spot, strike, rate, mid, time)?
        } else {
            black_scholes_put(spot, strike, rate, mid, time)?
        };

        let diff = price - market_price;

        if diff.abs() < tolerance {
            return Ok(mid);
        }

        if price > market_price {
            high = mid;
        } else {
            low = mid;
        }
    }

    Ok((low + high) / 2.0)
}

#[pyfunction]
fn implied_volatility_call(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    time: f64,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<f64> {
    implied_volatility_bisection(
        market_price,
        spot,
        strike,
        rate,
        time,
        true,
        tolerance,
        max_iterations,
    )
}

#[pyfunction]
fn implied_volatility_put(
    market_price: f64,
    spot: f64,
    strike: f64,
    rate: f64,
    time: f64,
    tolerance: f64,
    max_iterations: usize,
) -> PyResult<f64> {
    implied_volatility_bisection(
        market_price,
        spot,
        strike,
        rate,
        time,
        false,
        tolerance,
        max_iterations,
    )
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

    m.add_function(wrap_pyfunction!(delta_call, m)?)?;
    m.add_function(wrap_pyfunction!(delta_put, m)?)?;
    m.add_function(wrap_pyfunction!(gamma, m)?)?;
    m.add_function(wrap_pyfunction!(vega, m)?)?;
    m.add_function(wrap_pyfunction!(theta_call, m)?)?;
    m.add_function(wrap_pyfunction!(theta_put, m)?)?;
    m.add_function(wrap_pyfunction!(rho_call, m)?)?;
    m.add_function(wrap_pyfunction!(rho_put, m)?)?;

    m.add_function(wrap_pyfunction!(monte_carlo_call, m)?)?;
    m.add_function(wrap_pyfunction!(monte_carlo_put, m)?)?;

    m.add_function(wrap_pyfunction!(portfolio_return, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio_volatility, m)?)?;
    m.add_function(wrap_pyfunction!(sharpe_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(historical_var, m)?)?;
    m.add_function(wrap_pyfunction!(max_drawdown, m)?)?;

    m.add_function(wrap_pyfunction!(implied_volatility_call, m)?)?;
    m.add_function(wrap_pyfunction!(implied_volatility_put, m)?)?;

    Ok(())
}