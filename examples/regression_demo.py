import larp_quantmath as lq

x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 8, 10]

slope, intercept = lq.linear_regression(x, y)
predictions = lq.predict_linear(x, slope, intercept)
residual_values = lq.residuals(x, y, slope, intercept)
r2 = lq.r_squared(x, y, slope, intercept)

print("Slope:", slope)
print("Intercept:", intercept)
print("Predictions:", predictions)
print("Residuals:", residual_values)
print("R-squared:", r2)