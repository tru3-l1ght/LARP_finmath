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
    Ok(normal_cdf(d1_value(spot, strike, rate, volatility, time)))
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
    Ok(normal_cdf(d1_value(spot, strike, rate, volatility, time)) - 1.0)
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
        payoff_sum += (final_price - strike).max(0.0);
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
        payoff_sum += (strike - final_price).max(0.0);
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
        return Err(pyo3::exceptions::PyValueError::new_err("inputs cannot be empty"));
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
        return Err(pyo3::exceptions::PyValueError::new_err("weights cannot be empty"));
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

    let mut variance_value = 0.0;

    for i in 0..n {
        for j in 0..n {
            variance_value += weights[i] * weights[j] * covariance_matrix[i][j];
        }
    }

    Ok(variance_value.sqrt())
}

#[pyfunction]
fn sharpe_ratio(
    portfolio_return: f64,
    risk_free_rate: f64,
    portfolio_volatility: f64,
) -> PyResult<f64> {
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
        return Err(pyo3::exceptions::PyValueError::new_err("returns cannot be empty"));
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
        return Err(pyo3::exceptions::PyValueError::new_err("prices cannot be empty"));
    }

    if prices.iter().any(|&p| p <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err("prices must be positive"));
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
// Fixed income
// -----------------------------

#[pyfunction]
fn bond_price(
    face_value: f64,
    coupon_rate: f64,
    yield_rate: f64,
    years_to_maturity: f64,
    payments_per_year: usize,
) -> PyResult<f64> {
    if face_value <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("face_value must be positive"));
    }

    if coupon_rate < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("coupon_rate cannot be negative"));
    }

    if yield_rate <= -1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "yield_rate must be greater than -1",
        ));
    }

    if years_to_maturity <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "years_to_maturity must be positive",
        ));
    }

    if payments_per_year == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "payments_per_year must be greater than zero",
        ));
    }

    let total_periods = (years_to_maturity * payments_per_year as f64).round() as usize;
    let coupon_payment = face_value * coupon_rate / payments_per_year as f64;
    let period_yield = yield_rate / payments_per_year as f64;

    let mut price = 0.0;

    for t in 1..=total_periods {
        price += coupon_payment / (1.0 + period_yield).powi(t as i32);
    }

    price += face_value / (1.0 + period_yield).powi(total_periods as i32);

    Ok(price)
}

#[pyfunction]
fn current_yield(face_value: f64, coupon_rate: f64, market_price: f64) -> PyResult<f64> {
    if face_value <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("face_value must be positive"));
    }

    if coupon_rate < 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("coupon_rate cannot be negative"));
    }

    if market_price <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("market_price must be positive"));
    }

    Ok((face_value * coupon_rate) / market_price)
}

#[pyfunction]
fn macaulay_duration(
    face_value: f64,
    coupon_rate: f64,
    yield_rate: f64,
    years_to_maturity: f64,
    payments_per_year: usize,
) -> PyResult<f64> {
    let price = bond_price(
        face_value,
        coupon_rate,
        yield_rate,
        years_to_maturity,
        payments_per_year,
    )?;

    let total_periods = (years_to_maturity * payments_per_year as f64).round() as usize;
    let coupon_payment = face_value * coupon_rate / payments_per_year as f64;
    let period_yield = yield_rate / payments_per_year as f64;

    let mut weighted_sum = 0.0;

    for t in 1..=total_periods {
        let time_years = t as f64 / payments_per_year as f64;
        let cash_flow = if t == total_periods {
            coupon_payment + face_value
        } else {
            coupon_payment
        };

        let present_value = cash_flow / (1.0 + period_yield).powi(t as i32);
        weighted_sum += time_years * present_value;
    }

    Ok(weighted_sum / price)
}

#[pyfunction]
fn modified_duration(
    face_value: f64,
    coupon_rate: f64,
    yield_rate: f64,
    years_to_maturity: f64,
    payments_per_year: usize,
) -> PyResult<f64> {
    let mac_duration = macaulay_duration(
        face_value,
        coupon_rate,
        yield_rate,
        years_to_maturity,
        payments_per_year,
    )?;

    Ok(mac_duration / (1.0 + yield_rate / payments_per_year as f64))
}

#[pyfunction]
fn bond_convexity(
    face_value: f64,
    coupon_rate: f64,
    yield_rate: f64,
    years_to_maturity: f64,
    payments_per_year: usize,
) -> PyResult<f64> {
    let price = bond_price(
        face_value,
        coupon_rate,
        yield_rate,
        years_to_maturity,
        payments_per_year,
    )?;

    let total_periods = (years_to_maturity * payments_per_year as f64).round() as usize;
    let coupon_payment = face_value * coupon_rate / payments_per_year as f64;
    let period_yield = yield_rate / payments_per_year as f64;

    let mut convexity_sum = 0.0;

    for t in 1..=total_periods {
        let cash_flow = if t == total_periods {
            coupon_payment + face_value
        } else {
            coupon_payment
        };

        let t_f64 = t as f64;
        let present_value = cash_flow / (1.0 + period_yield).powi(t as i32);
        convexity_sum += t_f64 * (t_f64 + 1.0) * present_value;
    }

    let periods_per_year_squared = (payments_per_year as f64).powi(2);

    Ok(convexity_sum / (price * (1.0 + period_yield).powi(2) * periods_per_year_squared))
}

// -----------------------------
// Technical indicators
// -----------------------------

#[pyfunction]
fn sma(values: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    if window == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be greater than zero",
        ));
    }

    if values.len() < window {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "values length must be at least window size",
        ));
    }

    let mut result = Vec::new();

    for i in 0..=(values.len() - window) {
        let slice = &values[i..i + window];
        result.push(slice.iter().sum::<f64>() / window as f64);
    }

    Ok(result)
}

#[pyfunction]
fn ema(values: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    if window == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be greater than zero",
        ));
    }

    if values.len() < window {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "values length must be at least window size",
        ));
    }

    let multiplier = 2.0 / (window as f64 + 1.0);
    let mut result = Vec::new();

    let first_ema = values[..window].iter().sum::<f64>() / window as f64;
    result.push(first_ema);

    let mut previous_ema = first_ema;

    for price in values.iter().skip(window) {
        let current_ema = (price - previous_ema) * multiplier + previous_ema;
        result.push(current_ema);
        previous_ema = current_ema;
    }

    Ok(result)
}

#[pyfunction]
fn rsi(prices: Vec<f64>, window: usize) -> PyResult<Vec<f64>> {
    if window == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be greater than zero",
        ));
    }

    if prices.len() <= window {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "prices length must be greater than window size",
        ));
    }

    if prices.iter().any(|&p| p <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err("prices must be positive"));
    }

    let mut gains = Vec::new();
    let mut losses = Vec::new();

    for pair in prices.windows(2) {
        let change = pair[1] - pair[0];

        if change >= 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(change.abs());
        }
    }

    let mut result = Vec::new();

    let mut avg_gain = gains[..window].iter().sum::<f64>() / window as f64;
    let mut avg_loss = losses[..window].iter().sum::<f64>() / window as f64;

    if avg_loss == 0.0 {
        result.push(100.0);
    } else {
        let rs = avg_gain / avg_loss;
        result.push(100.0 - (100.0 / (1.0 + rs)));
    }

    for i in window..gains.len() {
        avg_gain = ((avg_gain * (window as f64 - 1.0)) + gains[i]) / window as f64;
        avg_loss = ((avg_loss * (window as f64 - 1.0)) + losses[i]) / window as f64;

        if avg_loss == 0.0 {
            result.push(100.0);
        } else {
            let rs = avg_gain / avg_loss;
            result.push(100.0 - (100.0 / (1.0 + rs)));
        }
    }

    Ok(result)
}

#[pyfunction]
fn rolling_volatility(
    returns: Vec<f64>,
    window: usize,
    annualization_factor: f64,
) -> PyResult<Vec<f64>> {
    if window == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be greater than zero",
        ));
    }

    if annualization_factor <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "annualization_factor must be positive",
        ));
    }

    if returns.len() < window {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "returns length must be at least window size",
        ));
    }

    let mut result = Vec::new();

    for i in 0..=(returns.len() - window) {
        let slice = returns[i..i + window].to_vec();
        result.push(std_dev(slice, true)? * annualization_factor.sqrt());
    }

    Ok(result)
}

// -----------------------------
// Advanced technical indicators
// -----------------------------

#[pyfunction]
fn macd(
    prices: Vec<f64>,
    fast_window: usize,
    slow_window: usize,
    signal_window: usize,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if fast_window == 0 || slow_window == 0 || signal_window == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "windows must be greater than zero",
        ));
    }

    if fast_window >= slow_window {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "fast_window must be less than slow_window",
        ));
    }

    if prices.len() < slow_window + signal_window {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "prices length is too short for MACD calculation",
        ));
    }

    let fast_ema = ema(prices.clone(), fast_window)?;
    let slow_ema = ema(prices.clone(), slow_window)?;

    let offset = slow_window - fast_window;
    let aligned_fast = &fast_ema[offset..];

    let macd_line: Vec<f64> = aligned_fast
        .iter()
        .zip(slow_ema.iter())
        .map(|(fast, slow)| fast - slow)
        .collect();

    let signal_line = ema(macd_line.clone(), signal_window)?;

    let histogram_offset = macd_line.len() - signal_line.len();
    let aligned_macd = &macd_line[histogram_offset..];

    let histogram: Vec<f64> = aligned_macd
        .iter()
        .zip(signal_line.iter())
        .map(|(m, s)| m - s)
        .collect();

    Ok((macd_line, signal_line, histogram))
}

#[pyfunction]
fn bollinger_bands(
    prices: Vec<f64>,
    window: usize,
    num_std_dev: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if window == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be greater than zero",
        ));
    }

    if num_std_dev <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_std_dev must be positive",
        ));
    }

    if prices.len() < window {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "prices length must be at least window size",
        ));
    }

    let mut middle_band = Vec::new();
    let mut upper_band = Vec::new();
    let mut lower_band = Vec::new();

    for i in 0..=(prices.len() - window) {
        let slice = prices[i..i + window].to_vec();
        let avg = mean(slice.clone())?;
        let sd = std_dev(slice, true)?;

        middle_band.push(avg);
        upper_band.push(avg + num_std_dev * sd);
        lower_band.push(avg - num_std_dev * sd);
    }

    Ok((middle_band, upper_band, lower_band))
}

#[pyfunction]
fn z_score(values: Vec<f64>, value: f64) -> PyResult<f64> {
    if values.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "at least two values are required",
        ));
    }

    let avg = mean(values.clone())?;
    let sd = std_dev(values, true)?;

    if sd == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "standard deviation cannot be zero",
        ));
    }

    Ok((value - avg) / sd)
}

// -----------------------------
// Backtesting metrics
// -----------------------------

#[pyfunction]
fn total_return(prices: Vec<f64>) -> PyResult<f64> {
    if prices.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "at least two prices are required",
        ));
    }

    if prices.iter().any(|&p| p <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err("prices must be positive"));
    }

    Ok((prices[prices.len() - 1] / prices[0]) - 1.0)
}

#[pyfunction]
fn cagr(prices: Vec<f64>, years: f64) -> PyResult<f64> {
    if prices.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "at least two prices are required",
        ));
    }

    if prices.iter().any(|&p| p <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err("prices must be positive"));
    }

    if years <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("years must be positive"));
    }

    Ok((prices[prices.len() - 1] / prices[0]).powf(1.0 / years) - 1.0)
}

#[pyfunction]
fn drawdown_series(prices: Vec<f64>) -> PyResult<Vec<f64>> {
    if prices.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("prices cannot be empty"));
    }

    if prices.iter().any(|&p| p <= 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err("prices must be positive"));
    }

    let mut peak = prices[0];
    let mut result = Vec::new();

    for price in prices {
        if price > peak {
            peak = price;
        }

        result.push((price - peak) / peak);
    }

    Ok(result)
}

#[pyfunction]
fn calmar_ratio(prices: Vec<f64>, years: f64) -> PyResult<f64> {
    let annual_return = cagr(prices.clone(), years)?;
    let max_dd = max_drawdown(prices)?;

    if max_dd == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "max drawdown cannot be zero",
        ));
    }

    Ok(annual_return / max_dd)
}

#[pyfunction]
fn win_rate(returns: Vec<f64>) -> PyResult<f64> {
    if returns.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("returns cannot be empty"));
    }

    let wins = returns.iter().filter(|&&r| r > 0.0).count();

    Ok(wins as f64 / returns.len() as f64)
}

#[pyfunction]
fn profit_factor(returns: Vec<f64>) -> PyResult<f64> {
    if returns.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("returns cannot be empty"));
    }

    let gross_profit: f64 = returns.iter().filter(|&&r| r > 0.0).sum();
    let gross_loss: f64 = returns
        .iter()
        .filter(|&&r| r < 0.0)
        .map(|r| r.abs())
        .sum();

    if gross_loss == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "gross loss cannot be zero",
        ));
    }

    Ok(gross_profit / gross_loss)
}

#[pyfunction]
fn average_return(returns: Vec<f64>) -> PyResult<f64> {
    mean(returns)
}

#[pyfunction]
fn annualized_return(returns: Vec<f64>, periods_per_year: f64) -> PyResult<f64> {
    if returns.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("returns cannot be empty"));
    }

    if periods_per_year <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "periods_per_year must be positive",
        ));
    }

    let compounded: f64 = returns.iter().fold(1.0, |acc, r| acc * (1.0 + r));

    Ok(compounded.powf(periods_per_year / returns.len() as f64) - 1.0)
}

#[pyfunction]
fn annualized_volatility(returns: Vec<f64>, periods_per_year: f64) -> PyResult<f64> {
    if returns.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "at least two returns are required",
        ));
    }

    if periods_per_year <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "periods_per_year must be positive",
        ));
    }

    Ok(std_dev(returns, true)? * periods_per_year.sqrt())
}

// -----------------------------
// Professional risk analytics
// -----------------------------

#[pyfunction]
fn downside_deviation(
    returns: Vec<f64>,
    minimum_acceptable_return: f64,
    periods_per_year: f64,
) -> PyResult<f64> {
    if returns.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("returns cannot be empty"));
    }

    if periods_per_year <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "periods_per_year must be positive",
        ));
    }

    let downside_sum: f64 = returns
        .iter()
        .map(|r| {
            let diff = (r - minimum_acceptable_return).min(0.0);
            diff.powi(2)
        })
        .sum();

    Ok((downside_sum / returns.len() as f64).sqrt() * periods_per_year.sqrt())
}

#[pyfunction]
fn sortino_ratio(
    portfolio_return: f64,
    minimum_acceptable_return: f64,
    downside_deviation_value: f64,
) -> PyResult<f64> {
    if downside_deviation_value <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "downside_deviation must be positive",
        ));
    }

    Ok((portfolio_return - minimum_acceptable_return) / downside_deviation_value)
}

#[pyfunction]
fn beta(asset_returns: Vec<f64>, market_returns: Vec<f64>) -> PyResult<f64> {
    if asset_returns.len() != market_returns.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "asset_returns and market_returns must have the same length",
        ));
    }

    if asset_returns.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "at least two returns are required",
        ));
    }

    let market_variance = variance(market_returns.clone(), true)?;

    if market_variance == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "market variance cannot be zero",
        ));
    }

    Ok(covariance(asset_returns, market_returns, true)? / market_variance)
}

#[pyfunction]
fn alpha(
    asset_return: f64,
    risk_free_rate: f64,
    beta_value: f64,
    market_return: f64,
) -> PyResult<f64> {
    Ok(asset_return - (risk_free_rate + beta_value * (market_return - risk_free_rate)))
}

#[pyfunction]
fn tracking_error(
    asset_returns: Vec<f64>,
    benchmark_returns: Vec<f64>,
    periods_per_year: f64,
) -> PyResult<f64> {
    if asset_returns.len() != benchmark_returns.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "asset_returns and benchmark_returns must have the same length",
        ));
    }

    if asset_returns.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "at least two returns are required",
        ));
    }

    if periods_per_year <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "periods_per_year must be positive",
        ));
    }

    let active_returns: Vec<f64> = asset_returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(a, b)| a - b)
        .collect();

    Ok(std_dev(active_returns, true)? * periods_per_year.sqrt())
}

#[pyfunction]
fn information_ratio(
    asset_returns: Vec<f64>,
    benchmark_returns: Vec<f64>,
    periods_per_year: f64,
) -> PyResult<f64> {
    if asset_returns.len() != benchmark_returns.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "asset_returns and benchmark_returns must have the same length",
        ));
    }

    if asset_returns.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("returns cannot be empty"));
    }

    if periods_per_year <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "periods_per_year must be positive",
        ));
    }

    let active_returns: Vec<f64> = asset_returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(a, b)| a - b)
        .collect();

    let avg_active_return = mean(active_returns.clone())? * periods_per_year;
    let te = tracking_error(asset_returns, benchmark_returns, periods_per_year)?;

    if te == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tracking error cannot be zero",
        ));
    }

    Ok(avg_active_return / te)
}

#[pyfunction]
fn expected_shortfall(returns: Vec<f64>, confidence_level: f64) -> PyResult<f64> {
    if returns.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("returns cannot be empty"));
    }

    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "confidence_level must be between 0 and 1",
        ));
    }

    let mut sorted_returns = returns;
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let tail_fraction = 1.0 - confidence_level;
    let tail_count = ((sorted_returns.len() as f64) * tail_fraction).ceil() as usize;
    let capped_tail_count = tail_count.max(1).min(sorted_returns.len());

    let tail_losses = &sorted_returns[..capped_tail_count];
    let avg_tail_return = tail_losses.iter().sum::<f64>() / capped_tail_count as f64;

    Ok(-avg_tail_return)
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

    m.add_function(wrap_pyfunction!(bond_price, m)?)?;
    m.add_function(wrap_pyfunction!(current_yield, m)?)?;
    m.add_function(wrap_pyfunction!(macaulay_duration, m)?)?;
    m.add_function(wrap_pyfunction!(modified_duration, m)?)?;
    m.add_function(wrap_pyfunction!(bond_convexity, m)?)?;

    m.add_function(wrap_pyfunction!(sma, m)?)?;
    m.add_function(wrap_pyfunction!(ema, m)?)?;
    m.add_function(wrap_pyfunction!(rsi, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_volatility, m)?)?;

    m.add_function(wrap_pyfunction!(macd, m)?)?;
    m.add_function(wrap_pyfunction!(bollinger_bands, m)?)?;
    m.add_function(wrap_pyfunction!(z_score, m)?)?;

    m.add_function(wrap_pyfunction!(total_return, m)?)?;
    m.add_function(wrap_pyfunction!(cagr, m)?)?;
    m.add_function(wrap_pyfunction!(drawdown_series, m)?)?;
    m.add_function(wrap_pyfunction!(calmar_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(win_rate, m)?)?;
    m.add_function(wrap_pyfunction!(profit_factor, m)?)?;
    m.add_function(wrap_pyfunction!(average_return, m)?)?;
    m.add_function(wrap_pyfunction!(annualized_return, m)?)?;
    m.add_function(wrap_pyfunction!(annualized_volatility, m)?)?;

    m.add_function(wrap_pyfunction!(downside_deviation, m)?)?;
    m.add_function(wrap_pyfunction!(sortino_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(beta, m)?)?;
    m.add_function(wrap_pyfunction!(alpha, m)?)?;
    m.add_function(wrap_pyfunction!(tracking_error, m)?)?;
    m.add_function(wrap_pyfunction!(information_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(expected_shortfall, m)?)?;

    Ok(())
}