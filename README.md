# LARP QuantMath

![CI](https://github.com/tru3-l1ght/LARP_finmath/actions/workflows/CI.yml/badge.svg)

LARP QuantMath is a Rust-powered Python library for quantitative finance, financial mathematics, portfolio analytics, risk metrics, technical indicators, symbolic calculus, regression, and linear algebra.

It is designed as the core computation engine for future LARP Financial Systems projects.

---

## Why This Project Exists

The goal of LARP QuantMath is to provide a fast, clean, and practical math engine for financial software.

It combines:

- Rust performance
- Python usability
- Quantitative finance functions
- Symbolic math through SymPy
- Test coverage through Pytest
- CI through GitHub Actions

---

## Installation

For local development:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install maturin pytest sympy numpy pandas scipy matplotlib openpyxl
maturin develop
```

---

## Running Tests

```bash
python3 -m pytest
```

---

# Function Reference

## Core Statistics

### `mean(values)`

Calculates the arithmetic average of a list of numbers.

Used for:

- Average returns
- Average prices
- Average portfolio performance
- Basic statistical summaries

Example:

```python
lq.mean([1, 2, 3, 4])
```

---

### `variance(values, sample)`

Calculates variance.

If `sample=True`, it uses sample variance.  
If `sample=False`, it uses population variance.

Used for:

- Measuring return dispersion
- Risk analysis
- Volatility calculations
- Statistical modeling

Example:

```python
lq.variance([1, 2, 3, 4], True)
```

---

### `std_dev(values, sample)`

Calculates standard deviation.

Used for:

- Volatility
- Risk measurement
- Dispersion analysis
- Comparing asset stability

Example:

```python
lq.std_dev([1, 2, 3, 4], True)
```

---

### `covariance(a, b, sample)`

Calculates covariance between two datasets.

Used for:

- Portfolio risk
- Asset relationship analysis
- Covariance matrices
- Factor models

Example:

```python
lq.covariance(asset_a_returns, asset_b_returns, True)
```

---

### `correlation(a, b)`

Calculates correlation between two datasets.

Used for:

- Measuring how two assets move together
- Diversification analysis
- Market relationship analysis
- Risk modeling

Example:

```python
lq.correlation(asset_a_returns, asset_b_returns)
```

---

## Returns

### `simple_returns(prices)`

Calculates simple percentage returns from a price series.

Formula:

```text
return = (price_t / price_t-1) - 1
```

Used for:

- Stock return calculations
- Portfolio analysis
- Backtesting
- Performance measurement

Example:

```python
lq.simple_returns([100, 105, 110])
```

---

### `log_returns(prices)`

Calculates logarithmic returns from a price series.

Formula:

```text
log_return = ln(price_t / price_t-1)
```

Used for:

- Quant finance models
- Time-series analysis
- Continuous compounding
- Risk modeling

Example:

```python
lq.log_returns([100, 105, 110])
```

---

# Options Pricing

## Black-Scholes

### `black_scholes_call(spot, strike, rate, volatility, time)`

Calculates the theoretical price of a European call option.

Used for:

- Option pricing
- Derivatives analytics
- Trading tools
- Risk systems

Example:

```python
lq.black_scholes_call(100, 100, 0.05, 0.20, 1)
```

---

### `black_scholes_put(spot, strike, rate, volatility, time)`

Calculates the theoretical price of a European put option.

Used for:

- Option pricing
- Hedging analysis
- Derivatives valuation
- Risk management

Example:

```python
lq.black_scholes_put(100, 100, 0.05, 0.20, 1)
```

---

### `black_scholes_d1(spot, strike, rate, volatility, time)`

Calculates the Black-Scholes `d1` term.

Used for:

- Option pricing internals
- Delta calculation
- Greek calculations
- Model debugging

Example:

```python
lq.black_scholes_d1(100, 100, 0.05, 0.20, 1)
```

---

### `black_scholes_d2(spot, strike, rate, volatility, time)`

Calculates the Black-Scholes `d2` term.

Used for:

- Option pricing internals
- Probability-related option calculations
- Greek calculations
- Model debugging

Example:

```python
lq.black_scholes_d2(100, 100, 0.05, 0.20, 1)
```

---

## Greeks

### `delta_call(spot, strike, rate, volatility, time)`

Calculates call option delta.

Used for:

- Measuring option sensitivity to stock price
- Hedge ratios
- Delta-neutral strategies
- Risk dashboards

Example:

```python
lq.delta_call(100, 100, 0.05, 0.20, 1)
```

---

### `delta_put(spot, strike, rate, volatility, time)`

Calculates put option delta.

Used for:

- Put option hedging
- Sensitivity analysis
- Risk management
- Options strategy evaluation

Example:

```python
lq.delta_put(100, 100, 0.05, 0.20, 1)
```

---

### `gamma(spot, strike, rate, volatility, time)`

Calculates gamma.

Gamma measures how fast delta changes as the stock price changes.

Used for:

- Convexity risk
- Options hedging
- Gamma scalping
- Risk management

Example:

```python
lq.gamma(100, 100, 0.05, 0.20, 1)
```

---

### `vega(spot, strike, rate, volatility, time)`

Calculates vega.

Vega measures option sensitivity to volatility.

Used for:

- Volatility risk
- Options trading
- Implied volatility analysis
- Risk dashboards

Example:

```python
lq.vega(100, 100, 0.05, 0.20, 1)
```

---

### `theta_call(spot, strike, rate, volatility, time)`

Calculates call option theta.

Theta measures time decay.

Used for:

- Options decay analysis
- Strategy evaluation
- Risk management
- Premium decay tracking

Example:

```python
lq.theta_call(100, 100, 0.05, 0.20, 1)
```

---

### `theta_put(spot, strike, rate, volatility, time)`

Calculates put option theta.

Used for:

- Put option time decay
- Options strategy analysis
- Risk modeling
- Trading dashboards

Example:

```python
lq.theta_put(100, 100, 0.05, 0.20, 1)
```

---

### `rho_call(spot, strike, rate, volatility, time)`

Calculates call option rho.

Rho measures sensitivity to interest rates.

Used for:

- Interest rate sensitivity
- Derivatives risk
- Option pricing analysis
- Macro-sensitive option strategies

Example:

```python
lq.rho_call(100, 100, 0.05, 0.20, 1)
```

---

### `rho_put(spot, strike, rate, volatility, time)`

Calculates put option rho.

Used for:

- Interest rate risk
- Put option pricing
- Derivatives analytics
- Risk dashboards

Example:

```python
lq.rho_put(100, 100, 0.05, 0.20, 1)
```

---

## Monte Carlo Options

### `monte_carlo_call(spot, strike, rate, volatility, time, simulations, seed)`

Prices a European call option using Monte Carlo simulation.

Used for:

- Simulation-based option pricing
- Comparing against Black-Scholes
- Teaching stochastic finance
- Future exotic option pricing

Example:

```python
lq.monte_carlo_call(100, 100, 0.05, 0.20, 1, 100000, 42)
```

---

### `monte_carlo_put(spot, strike, rate, volatility, time, simulations, seed)`

Prices a European put option using Monte Carlo simulation.

Used for:

- Simulation-based derivatives pricing
- Risk modeling
- Quant finance education
- Future exotic options work

Example:

```python
lq.monte_carlo_put(100, 100, 0.05, 0.20, 1, 100000, 42)
```

---

### `implied_volatility_call(market_price, spot, strike, rate, time, tolerance, max_iterations)`

Solves for implied volatility from a market call option price.

Used for:

- Finding market-implied volatility
- Options analytics
- Volatility surfaces
- Trading dashboards

Example:

```python
lq.implied_volatility_call(10.45, 100, 100, 0.05, 1, 1e-8, 100)
```

---

### `implied_volatility_put(market_price, spot, strike, rate, time, tolerance, max_iterations)`

Solves for implied volatility from a market put option price.

Used for:

- Put option implied volatility
- Volatility analysis
- Market expectation analysis
- Options research

Example:

```python
lq.implied_volatility_put(5.57, 100, 100, 0.05, 1, 1e-8, 100)
```

---

# Portfolio and Risk Analytics

### `portfolio_return(returns, weights)`

Calculates weighted portfolio return.

Used for:

- Portfolio performance
- Asset allocation
- Portfolio dashboards
- Investment analysis

Example:

```python
lq.portfolio_return([0.10, 0.08, 0.12], [0.4, 0.3, 0.3])
```

---

### `portfolio_volatility(covariance_matrix, weights)`

Calculates portfolio volatility using a covariance matrix.

Used for:

- Portfolio risk
- Modern Portfolio Theory
- Asset allocation
- Risk dashboards

Example:

```python
lq.portfolio_volatility(cov_matrix, weights)
```

---

### `portfolio_variance(covariance_matrix, weights)`

Calculates portfolio variance.

Used for:

- Portfolio optimization
- Risk modeling
- Minimum-variance portfolios
- Quant finance models

Example:

```python
lq.portfolio_variance(cov_matrix, weights)
```

---

### `sharpe_ratio(portfolio_return, risk_free_rate, portfolio_volatility)`

Calculates Sharpe ratio.

Used for:

- Risk-adjusted performance
- Comparing strategies
- Portfolio evaluation
- Fund analytics

Example:

```python
lq.sharpe_ratio(0.10, 0.02, 0.15)
```

---

### `historical_var(returns, confidence_level)`

Calculates historical Value at Risk.

Used for:

- Downside risk measurement
- Portfolio risk systems
- Risk reporting
- Stress testing

Example:

```python
lq.historical_var(returns, 0.95)
```

---

### `expected_shortfall(returns, confidence_level)`

Calculates expected shortfall, also called Conditional VaR.

Used for:

- Tail-risk measurement
- Risk management
- Portfolio stress analysis
- Downside loss estimation

Example:

```python
lq.expected_shortfall(returns, 0.95)
```

---

### `max_drawdown(values)`

Calculates maximum drawdown from a value series.

Used for:

- Backtesting
- Portfolio loss analysis
- Strategy evaluation
- Risk dashboards

Example:

```python
lq.max_drawdown([100, 120, 90, 130])
```

---

### `downside_deviation(returns, minimum_acceptable_return)`

Calculates downside deviation.

Used for:

- Downside risk
- Sortino ratio
- Strategy risk analysis
- Portfolio evaluation

Example:

```python
lq.downside_deviation(returns, 0.0)
```

---

### `sortino_ratio(returns, risk_free_rate, minimum_acceptable_return)`

Calculates Sortino ratio.

Used for:

- Downside-risk-adjusted performance
- Strategy comparison
- Portfolio analysis
- Fund evaluation

Example:

```python
lq.sortino_ratio(returns, 0.02, 0.0)
```

---

### `beta(asset_returns, market_returns)`

Calculates beta relative to the market.

Used for:

- Market sensitivity
- CAPM
- Risk attribution
- Equity analysis

Example:

```python
lq.beta(asset_returns, market_returns)
```

---

### `alpha(asset_returns, market_returns, risk_free_rate)`

Calculates alpha.

Used for:

- Measuring excess return
- Fund performance
- Strategy evaluation
- CAPM analysis

Example:

```python
lq.alpha(asset_returns, market_returns, 0.02)
```

---

### `tracking_error(portfolio_returns, benchmark_returns)`

Calculates tracking error.

Used for:

- Portfolio vs benchmark comparison
- Fund management
- Active risk measurement
- Index tracking analysis

Example:

```python
lq.tracking_error(portfolio_returns, benchmark_returns)
```

---

### `information_ratio(portfolio_returns, benchmark_returns)`

Calculates information ratio.

Used for:

- Active management performance
- Portfolio manager evaluation
- Benchmark-relative analysis
- Risk-adjusted excess return

Example:

```python
lq.information_ratio(portfolio_returns, benchmark_returns)
```

---

# Portfolio Optimization

### `normalize_weights(weights)`

Normalizes weights so they sum to 1.

Used for:

- Portfolio construction
- Cleaning user input
- Allocation tools
- Optimization preprocessing

Example:

```python
lq.normalize_weights([2, 3, 5])
```

---

### `equal_weight_portfolio(asset_count)`

Creates equal weights for a portfolio.

Used for:

- Equal-weight portfolio construction
- Baseline portfolios
- Benchmark allocation
- Beginner portfolio tools

Example:

```python
lq.equal_weight_portfolio(4)
```

---

### `minimum_variance_two_asset_weights(variance_a, variance_b, covariance_ab)`

Calculates minimum-variance weights for a two-asset portfolio.

Used for:

- Simple portfolio optimization
- Risk-minimizing allocation
- Teaching portfolio theory
- Asset allocation demos

Example:

```python
lq.minimum_variance_two_asset_weights(0.04, 0.09, 0.01)
```

---

### `risk_parity_two_asset_weights(volatility_a, volatility_b)`

Calculates two-asset inverse-volatility risk parity weights.

Used for:

- Risk parity allocation
- Balanced portfolio construction
- Volatility-based weighting
- Portfolio optimization demos

Example:

```python
lq.risk_parity_two_asset_weights(0.20, 0.40)
```

---

# Fixed Income

### `bond_price(face_value, coupon_rate, yield_rate, years, payments_per_year)`

Calculates the price of a coupon bond.

Used for:

- Bond valuation
- Fixed income analytics
- Yield analysis
- Investment tools

Example:

```python
lq.bond_price(1000, 0.05, 0.04, 5, 1)
```

---

### `current_yield(face_value, coupon_rate, market_price)`

Calculates current yield.

Used for:

- Bond income analysis
- Yield comparison
- Fixed income dashboards
- Investment screening

Example:

```python
lq.current_yield(1000, 0.05, 950)
```

---

### `macaulay_duration(face_value, coupon_rate, yield_rate, years, payments_per_year)`

Calculates Macaulay duration.

Used for:

- Interest rate sensitivity
- Bond risk analysis
- Duration matching
- Fixed income portfolio management

Example:

```python
lq.macaulay_duration(1000, 0.05, 0.04, 5, 1)
```

---

### `modified_duration(face_value, coupon_rate, yield_rate, years, payments_per_year)`

Calculates modified duration.

Used for:

- Estimating bond price sensitivity to yield changes
- Interest rate risk
- Fixed income risk dashboards
- Bond portfolio management

Example:

```python
lq.modified_duration(1000, 0.05, 0.04, 5, 1)
```

---

### `bond_convexity(face_value, coupon_rate, yield_rate, years, payments_per_year)`

Calculates bond convexity.

Used for:

- Second-order interest rate risk
- Bond risk analytics
- Duration-convexity analysis
- Fixed income modeling

Example:

```python
lq.bond_convexity(1000, 0.05, 0.04, 5, 1)
```

---

### `yield_to_maturity(face_value, coupon_rate, market_price, years, payments_per_year, tolerance, max_iterations)`

Solves for yield to maturity.

Used for:

- Bond yield analysis
- Fixed income valuation
- Comparing bonds
- Investment screening

Example:

```python
lq.yield_to_maturity(1000, 0.05, 950, 5, 1, 1e-8, 100)
```

---

### `zero_coupon_bond_price(face_value, yield_rate, years)`

Calculates zero-coupon bond price.

Used for:

- Discount bond valuation
- Present value analysis
- Yield curve tools
- Fixed income education

Example:

```python
lq.zero_coupon_bond_price(1000, 0.04, 5)
```

---

### `dv01(face_value, coupon_rate, yield_rate, years, payments_per_year)`

Calculates DV01.

DV01 estimates dollar price change for a 1 basis point yield move.

Used for:

- Bond risk
- Rate sensitivity
- Portfolio hedging
- Fixed income dashboards

Example:

```python
lq.dv01(1000, 0.05, 0.04, 5, 1)
```

---

### `forward_rate(spot_rate_short, spot_rate_long, short_years, long_years)`

Calculates implied forward rate between two maturities.

Used for:

- Yield curve analysis
- Interest rate modeling
- Forward rate estimation
- Fixed income research

Example:

```python
lq.forward_rate(0.03, 0.04, 1, 2)
```

---

# Technical Indicators

### `sma(values, period)`

Calculates simple moving average.

Used for:

- Trend analysis
- Trading signals
- Charting
- Technical analysis

Example:

```python
lq.sma(prices, 20)
```

---

### `ema(values, period)`

Calculates exponential moving average.

Used for:

- Trend-following systems
- Momentum analysis
- Trading dashboards
- Technical indicators

Example:

```python
lq.ema(prices, 20)
```

---

### `rsi(values, period)`

Calculates Relative Strength Index.

Used for:

- Momentum analysis
- Overbought/oversold signals
- Trading tools
- Market screening

Example:

```python
lq.rsi(prices, 14)
```

---

### `rolling_volatility(returns, period)`

Calculates rolling volatility.

Used for:

- Time-varying risk
- Strategy risk analysis
- Volatility dashboards
- Market regime detection

Example:

```python
lq.rolling_volatility(returns, 20)
```

---

### `macd(values, fast_period, slow_period, signal_period)`

Calculates MACD values.

Used for:

- Momentum trading
- Trend detection
- Technical analysis
- Signal generation

Example:

```python
lq.macd(prices, 12, 26, 9)
```

---

### `bollinger_bands(values, period, num_std_dev)`

Calculates Bollinger Bands.

Used for:

- Volatility bands
- Mean reversion signals
- Trading dashboards
- Technical analysis

Example:

```python
lq.bollinger_bands(prices, 20, 2.0)
```

---

### `z_score(values)`

Calculates z-scores.

Used for:

- Standardization
- Outlier detection
- Mean reversion analysis
- Statistical signals

Example:

```python
lq.z_score(values)
```

---

# Backtesting Metrics

### `total_return(values)`

Calculates total return from a value series.

Used for:

- Strategy performance
- Portfolio return analysis
- Backtesting
- Investment dashboards

Example:

```python
lq.total_return([100, 120])
```

---

### `cagr(start_value, end_value, years)`

Calculates compound annual growth rate.

Used for:

- Long-term performance measurement
- Investment comparison
- Portfolio reporting
- Financial analysis

Example:

```python
lq.cagr(100, 150, 3)
```

---

### `drawdown_series(values)`

Calculates drawdown at each point in a value series.

Used for:

- Backtesting
- Risk visualization
- Strategy analysis
- Portfolio diagnostics

Example:

```python
lq.drawdown_series([100, 120, 90, 130])
```

---

### `calmar_ratio(values, years)`

Calculates Calmar ratio.

Used for:

- Return vs drawdown analysis
- Strategy evaluation
- Fund comparison
- Risk-adjusted performance

Example:

```python
lq.calmar_ratio(values, 3)
```

---

### `win_rate(returns)`

Calculates percentage of positive returns.

Used for:

- Trading strategy evaluation
- Backtesting
- Performance diagnostics
- Trade analytics

Example:

```python
lq.win_rate(returns)
```

---

### `profit_factor(returns)`

Calculates gross profit divided by gross loss.

Used for:

- Trading strategy analysis
- Backtesting
- Profitability diagnostics
- Risk/reward evaluation

Example:

```python
lq.profit_factor(returns)
```

---

### `average_return(returns)`

Calculates average return.

Used for:

- Performance summaries
- Strategy comparison
- Backtesting
- Portfolio analysis

Example:

```python
lq.average_return(returns)
```

---

### `annualized_return(returns, periods_per_year)`

Calculates annualized return.

Used for:

- Strategy annualization
- Comparing different return frequencies
- Fund analytics
- Portfolio reporting

Example:

```python
lq.annualized_return(monthly_returns, 12)
```

---

### `annualized_volatility(returns, periods_per_year)`

Calculates annualized volatility.

Used for:

- Annualized risk
- Strategy comparison
- Portfolio analysis
- Risk dashboards

Example:

```python
lq.annualized_volatility(daily_returns, 252)
```

---

# Time Value of Money

### `present_value(future_value, rate, periods)`

Calculates present value.

Used for:

- Discounted cash flow
- Valuation
- Financial planning
- Investment analysis

Example:

```python
lq.present_value(110, 0.10, 1)
```

---

### `future_value(present_value, rate, periods)`

Calculates future value.

Used for:

- Compounding analysis
- Investment growth
- Savings models
- Financial planning

Example:

```python
lq.future_value(100, 0.10, 1)
```

---

### `net_present_value(rate, cash_flows)`

Calculates net present value.

Used for:

- Project valuation
- Capital budgeting
- Investment analysis
- DCF modeling

Example:

```python
lq.net_present_value(0.10, [-1000, 500, 700])
```

---

### `internal_rate_of_return(cash_flows, tolerance, max_iterations)`

Solves for internal rate of return.

Used for:

- Investment evaluation
- Project finance
- Private equity-style analysis
- Capital budgeting

Example:

```python
lq.internal_rate_of_return([-100, 110], 1e-9, 100)
```

---

### `loan_payment(principal, annual_rate, years, payments_per_year)`

Calculates loan payment.

Used for:

- Loan calculators
- Mortgage tools
- Debt analysis
- Personal finance apps

Example:

```python
lq.loan_payment(100000, 0.05, 30, 12)
```

---

### `annuity_payment(present_value, rate, periods)`

Calculates annuity payment.

Used for:

- Retirement calculators
- Loan amortization
- Financial planning
- Cash flow modeling

Example:

```python
lq.annuity_payment(100000, 0.05, 30)
```

---

# Linear Algebra

### `dot_product(a, b)`

Calculates dot product between two vectors.

Used for:

- Portfolio math
- Matrix operations
- Regression
- Quant models

Example:

```python
lq.dot_product([1, 2, 3], [4, 5, 6])
```

---

### `transpose(matrix)`

Transposes a matrix.

Used for:

- Matrix algebra
- Regression preparation
- Data transformation
- Linear algebra workflows

Example:

```python
lq.transpose([[1, 2, 3], [4, 5, 6]])
```

---

### `matrix_multiply(a, b)`

Multiplies two matrices.

Used for:

- Linear algebra
- Portfolio models
- Regression systems
- Quantitative finance calculations

Example:

```python
lq.matrix_multiply([[1, 2]], [[3], [4]])
```

---

### `matrix_vector_multiply(matrix, vector)`

Multiplies a matrix by a vector.

Used for:

- Portfolio risk
- Linear systems
- Regression
- Factor models

Example:

```python
lq.matrix_vector_multiply([[1, 2], [3, 4]], [5, 6])
```

---

### `identity_matrix(size)`

Creates an identity matrix.

Used for:

- Linear algebra
- Matrix operations
- Numerical methods
- Testing matrix functions

Example:

```python
lq.identity_matrix(3)
```

---

### `covariance_matrix(data, sample)`

Creates a covariance matrix from multiple data series.

Used for:

- Portfolio optimization
- Risk modeling
- Asset allocation
- Quant finance

Example:

```python
lq.covariance_matrix(asset_returns, True)
```

---

### `correlation_matrix(data)`

Creates a correlation matrix from multiple data series.

Used for:

- Diversification analysis
- Portfolio diagnostics
- Asset relationship analysis
- Risk modeling

Example:

```python
lq.correlation_matrix(asset_returns)
```

---

### `matrix_add(a, b)`

Adds two matrices.

Used for:

- Matrix algebra
- Numerical models
- Quant calculations
- Linear algebra workflows

Example:

```python
lq.matrix_add([[1, 2]], [[3, 4]])
```

---

### `matrix_subtract(a, b)`

Subtracts one matrix from another.

Used for:

- Matrix algebra
- Residual-style calculations
- Numerical models
- Quant workflows

Example:

```python
lq.matrix_subtract([[5, 6]], [[1, 2]])
```

---

### `scalar_multiply_matrix(matrix, scalar)`

Multiplies a matrix by a scalar.

Used for:

- Scaling matrices
- Linear algebra
- Risk model adjustments
- Quant calculations

Example:

```python
lq.scalar_multiply_matrix([[1, 2], [3, 4]], 2)
```

---

### `vector_norm(vector)`

Calculates Euclidean norm of a vector.

Used for:

- Vector magnitude
- Normalization
- Linear algebra
- Numerical methods

Example:

```python
lq.vector_norm([3, 4])
```

---

### `normalize_vector(vector)`

Normalizes a vector to unit length.

Used for:

- Machine learning preprocessing
- Linear algebra
- Factor models
- Numerical workflows

Example:

```python
lq.normalize_vector([3, 4])
```

---

### `determinant_2x2(matrix)`

Calculates determinant of a 2x2 matrix.

Used for:

- Matrix invertibility checks
- Linear algebra education
- Small matrix systems
- Numerical demos

Example:

```python
lq.determinant_2x2([[1, 2], [3, 4]])
```

---

### `inverse_2x2(matrix)`

Calculates inverse of a 2x2 matrix.

Used for:

- Solving small linear systems
- Linear algebra education
- Quant demos
- Matrix calculations

Example:

```python
lq.inverse_2x2([[4, 7], [2, 6]])
```

---

# Symbolic Calculus

Symbolic functions are powered by SymPy.

### `simplify_expr(expr)`

Simplifies a symbolic expression.

Used for:

- Algebra simplification
- Educational tools
- Formula cleanup
- Symbolic finance/math workflows

Example:

```python
lq.simplify_expr("(x**2 - 1) / (x - 1)")
```

---

### `derivative(expr, var, order=1)`

Calculates symbolic derivative.

Used for:

- Calculus
- Sensitivity analysis
- Financial formula derivatives
- Education tools

Example:

```python
lq.derivative("x**3 + 2*x", "x")
```

---

### `integral(expr, var)`

Calculates indefinite integral.

Used for:

- Calculus
- Financial math education
- Symbolic modeling
- Formula manipulation

Example:

```python
lq.integral("2*x", "x")
```

---

### `definite_integral(expr, var, lower, upper)`

Calculates definite integral over a range.

Used for:

- Area under curves
- Probability/math education
- Financial math
- Symbolic analysis

Example:

```python
lq.definite_integral("x", "x", 0, 1)
```

---

### `limit(expr, var, point, direction="+")`

Calculates symbolic limit.

Used for:

- Calculus
- Continuity analysis
- Financial math education
- Symbolic reasoning

Example:

```python
lq.limit("sin(x)/x", "x", 0)
```

---

### `solve_expr(expr, var)`

Solves an equation expression equal to zero.

Used for:

- Equation solving
- Break-even analysis
- Financial math
- Symbolic calculators

Example:

```python
lq.solve_expr("x**2 - 4", "x")
```

---

### `taylor_series(expr, var, point, order)`

Calculates Taylor series expansion.

Used for:

- Approximation
- Calculus education
- Numerical methods
- Financial formula expansion

Example:

```python
lq.taylor_series("sin(x)", "x", 0, 6)
```

---

# Regression and Factor Models

### `linear_regression(x, y)`

Calculates simple linear regression slope and intercept.

Used for:

- Trend modeling
- Factor analysis
- Forecasting basics
- Quant research

Example:

```python
slope, intercept = lq.linear_regression([1, 2, 3], [2, 4, 6])
```

---

### `predict_linear(x, slope, intercept)`

Generates predictions from a linear model.

Used for:

- Regression prediction
- Forecasting
- Model demos
- Quant analysis

Example:

```python
lq.predict_linear([1, 2, 3], 2, 0)
```

---

### `residuals(x, y, slope, intercept)`

Calculates residuals from a linear model.

Used for:

- Regression diagnostics
- Error analysis
- Model validation
- Forecasting evaluation

Example:

```python
lq.residuals([1, 2, 3], [2, 4, 6], 2, 0)
```

---

### `r_squared(x, y, slope, intercept)`

Calculates R-squared.

Used for:

- Regression fit quality
- Model evaluation
- Forecasting diagnostics
- Quant research

Example:

```python
lq.r_squared([1, 2, 3], [2, 4, 6], 2, 0)
```

---

### `capm_expected_return(risk_free_rate, beta_value, market_return)`

Calculates expected return using CAPM.

Used for:

- Equity valuation
- Cost of equity
- Portfolio theory
- Financial modeling

Example:

```python
lq.capm_expected_return(0.02, 1.2, 0.10)
```

---

### `factor_exposure(asset_returns, factor_returns)`

Calculates exposure of an asset to a factor.

Used for:

- Factor investing
- Risk attribution
- Quant equity analysis
- Portfolio diagnostics

Example:

```python
lq.factor_exposure(asset_returns, factor_returns)
```

---

# Usage Examples

## Black-Scholes Option Pricing

```python
import larp_quantmath as lq

call = lq.black_scholes_call(100, 100, 0.05, 0.20, 1)
put = lq.black_scholes_put(100, 100, 0.05, 0.20, 1)

print(call)
print(put)
```

---

## Portfolio Analytics

```python
import larp_quantmath as lq

returns = [0.10, 0.08, 0.12]
weights = [0.4, 0.3, 0.3]

cov = [
    [0.04, 0.01, 0.015],
    [0.01, 0.03, 0.012],
    [0.015, 0.012, 0.05],
]

portfolio_return = lq.portfolio_return(returns, weights)
portfolio_volatility = lq.portfolio_volatility(cov, weights)
sharpe = lq.sharpe_ratio(portfolio_return, 0.02, portfolio_volatility)

print(portfolio_return)
print(portfolio_volatility)
print(sharpe)
```

---

## Fixed Income

```python
import larp_quantmath as lq

price = lq.bond_price(1000, 0.05, 0.04, 5, 1)
duration = lq.macaulay_duration(1000, 0.05, 0.04, 5, 1)
dv01 = lq.dv01(1000, 0.05, 0.04, 5, 1)

print(price)
print(duration)
print(dv01)
```

---

## Symbolic Calculus

```python
import larp_quantmath as lq

print(lq.derivative("x**3 + 2*x", "x"))
print(lq.integral("2*x", "x"))
print(lq.limit("sin(x)/x", "x", 0))
print(lq.solve_expr("x**2 - 4", "x"))
```

---

## Linear Regression

```python
import larp_quantmath as lq

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 8, 10]

slope, intercept = lq.linear_regression(x, y)
predictions = lq.predict_linear(x, slope, intercept)
residual_values = lq.residuals(x, y, slope, intercept)
r2 = lq.r_squared(x, y, slope, intercept)

print(slope)
print(intercept)
print(predictions)
print(residual_values)
print(r2)
```

---

# Examples Folder

The `examples/` folder includes:

```text
options_demo.py
portfolio_demo.py
fixed_income_demo.py
symbolic_demo.py
regression_demo.py
```

Run them with:

```bash
python3 examples/options_demo.py
python3 examples/portfolio_demo.py
python3 examples/fixed_income_demo.py
python3 examples/symbolic_demo.py
python3 examples/regression_demo.py
```

---

# Project Structure

```text
LARP_finmath/
├── src/
│   └── lib.rs
├── larp_quantmath/
│   ├── __init__.py
│   └── symbolic.py
├── tests/
├── examples/
├── Cargo.toml
├── pyproject.toml
└── README.md
```

---

# Tech Stack

- Rust
- PyO3
- Maturin
- Python
- SymPy
- Pytest
- GitHub Actions

---

# Development Status

This project is currently in active development.

Current focus:

- Expanding quantitative finance tools
- Improving test coverage
- Adding examples and documentation
- Preparing for future LARP Financial Systems applications

---

## Benchmark

Monte Carlo call option pricing with 1,000,000 simulations:

| Implementation | Time |
|---|---:|
| Rust | 0.0787 seconds |
| Pure Python | 0.8387 seconds |

Rust speedup: approximately **10.66x**.

Run benchmark:

```bash
maturin develop --release
python3 benchmarks/monte_carlo_benchmark.py
```

---

# Roadmap

# Roadmap

Planned future improvements:

- General matrix inverse
- Cholesky decomposition
- Multiple linear regression
- Portfolio optimization beyond two assets
- Efficient frontier tools
- More forecasting tools
- More accounting and valuation ratios
- Benchmarks against pure Python implementations
- PyPI packaging
- Documentation site
