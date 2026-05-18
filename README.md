# LARP QuantMath

LARP QuantMath is a Rust-powered Python library for quantitative finance, financial mathematics, portfolio analytics, risk metrics, symbolic calculus, regression, and linear algebra.

It is designed as the core computation engine for future LARP Financial Systems projects.

---

## Features

### Core Statistics

- Mean
- Variance
- Standard deviation
- Covariance
- Correlation
- Simple returns
- Log returns

### Options Pricing

- Black-Scholes call pricing
- Black-Scholes put pricing
- Black-Scholes d1 and d2
- Greeks:
  - Delta
  - Gamma
  - Vega
  - Theta
  - Rho
- Monte Carlo call pricing
- Monte Carlo put pricing
- Implied volatility solver

### Portfolio and Risk Analytics

- Portfolio return
- Portfolio volatility
- Portfolio variance
- Sharpe ratio
- Historical VaR
- Expected shortfall
- Max drawdown
- Sortino ratio
- Downside deviation
- Beta
- Alpha
- Tracking error
- Information ratio

### Portfolio Optimization

- Normalize weights
- Equal-weight portfolio
- Minimum-variance two-asset allocation
- Risk-parity two-asset allocation

### Fixed Income

- Bond price
- Current yield
- Macaulay duration
- Modified duration
- Bond convexity
- Yield to maturity
- Zero-coupon bond price
- DV01
- Forward rate

### Technical Indicators

- Simple moving average
- Exponential moving average
- RSI
- Rolling volatility
- MACD
- Bollinger Bands
- Z-score

### Backtesting Metrics

- Total return
- CAGR
- Drawdown series
- Calmar ratio
- Win rate
- Profit factor
- Average return
- Annualized return
- Annualized volatility

### Time Value of Money

- Present value
- Future value
- Net present value
- Internal rate of return
- Loan payment
- Annuity payment

### Linear Algebra

- Dot product
- Matrix multiplication
- Matrix-vector multiplication
- Transpose
- Identity matrix
- Covariance matrix
- Correlation matrix
- Matrix addition
- Matrix subtraction
- Scalar matrix multiplication
- Vector norm
- Vector normalization
- 2x2 determinant
- 2x2 inverse

### Symbolic Calculus

Powered by SymPy:

- Expression simplification
- Derivatives
- Integrals
- Definite integrals
- Limits
- Equation solving
- Taylor series

### Regression and Factor Models

- Linear regression
- Linear prediction
- Residuals
- R-squared
- CAPM expected return
- Factor exposure

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

## Usage Examples

### Black-Scholes Option Pricing

```python
import larp_quantmath as lq

call = lq.black_scholes_call(100, 100, 0.05, 0.20, 1)
put = lq.black_scholes_put(100, 100, 0.05, 0.20, 1)

print(call)
print(put)
```

### Greeks

```python
import larp_quantmath as lq

delta = lq.delta_call(100, 100, 0.05, 0.20, 1)
gamma = lq.gamma(100, 100, 0.05, 0.20, 1)
vega = lq.vega(100, 100, 0.05, 0.20, 1)

print(delta)
print(gamma)
print(vega)
```

### Monte Carlo Option Pricing

```python
import larp_quantmath as lq

call_mc = lq.monte_carlo_call(100, 100, 0.05, 0.20, 1, 100_000, 42)
put_mc = lq.monte_carlo_put(100, 100, 0.05, 0.20, 1, 100_000, 42)

print(call_mc)
print(put_mc)
```

### Portfolio Analytics

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

### Portfolio Optimization

```python
import larp_quantmath as lq

weights = lq.equal_weight_portfolio(4)
normalized = lq.normalize_weights([2, 3, 5])

min_var = lq.minimum_variance_two_asset_weights(0.04, 0.09, 0.01)
risk_parity = lq.risk_parity_two_asset_weights(0.20, 0.40)

print(weights)
print(normalized)
print(min_var)
print(risk_parity)
```

### Fixed Income

```python
import larp_quantmath as lq

price = lq.bond_price(1000, 0.05, 0.04, 5, 1)
duration = lq.macaulay_duration(1000, 0.05, 0.04, 5, 1)
modified_duration = lq.modified_duration(1000, 0.05, 0.04, 5, 1)
convexity = lq.bond_convexity(1000, 0.05, 0.04, 5, 1)
dv01 = lq.dv01(1000, 0.05, 0.04, 5, 1)

print(price)
print(duration)
print(modified_duration)
print(convexity)
print(dv01)
```

### Time Value of Money

```python
import larp_quantmath as lq

pv = lq.present_value(110, 0.10, 1)
fv = lq.future_value(100, 0.10, 1)
npv = lq.net_present_value(0.10, [-1000, 500, 700])
irr = lq.internal_rate_of_return([-100, 110], 1e-9, 100)

print(pv)
print(fv)
print(npv)
print(irr)
```

### Technical Indicators

```python
import larp_quantmath as lq

prices = [100, 102, 101, 105, 107, 106, 110, 112]

sma = lq.sma(prices, 3)
ema = lq.ema(prices, 3)
rsi = lq.rsi(prices, 3)

print(sma)
print(ema)
print(rsi)
```

### Symbolic Calculus

```python
import larp_quantmath as lq

print(lq.simplify_expr("(x**2 - 1) / (x - 1)"))
print(lq.derivative("x**3 + 2*x", "x"))
print(lq.integral("2*x", "x"))
print(lq.definite_integral("x", "x", 0, 1))
print(lq.limit("sin(x)/x", "x", 0))
print(lq.solve_expr("x**2 - 4", "x"))
```

### Linear Algebra

```python
import larp_quantmath as lq

dot = lq.dot_product([1, 2, 3], [4, 5, 6])
product = lq.matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
inverse = lq.inverse_2x2([[4, 7], [2, 6]])

print(dot)
print(product)
print(inverse)
```

### Linear Regression

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

## Examples

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

## Project Structure

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

## Tech Stack

- Rust
- PyO3
- Maturin
- Python
- SymPy
- Pytest
- GitHub Actions

---

## Development Status

This project is currently in active development.

Current focus:

- Expanding quantitative finance tools
- Improving test coverage
- Adding examples and documentation
- Preparing for future LARP Financial Systems applications

---

## Roadmap

Planned future improvements:

- General matrix inverse
- Cholesky decomposition
- Multiple linear regression
- More portfolio optimization methods
- More forecasting tools
- More accounting and valuation ratios
- Benchmarks against pure Python implementations
- PyPI packaging

