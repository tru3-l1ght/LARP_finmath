import larp_quantmath as lq

returns = [0.10, 0.08, 0.12]
weights = [0.4, 0.3, 0.3]

covariance_matrix = [
    [0.04, 0.01, 0.015],
    [0.01, 0.03, 0.012],
    [0.015, 0.012, 0.05],
]

portfolio_return = lq.portfolio_return(returns, weights)
portfolio_volatility = lq.portfolio_volatility(covariance_matrix, weights)
sharpe = lq.sharpe_ratio(portfolio_return, 0.02, portfolio_volatility)

print("Portfolio return:", portfolio_return)
print("Portfolio volatility:", portfolio_volatility)
print("Sharpe ratio:", sharpe)

equal_weights = lq.equal_weight_portfolio(3)
print("Equal weights:", equal_weights)