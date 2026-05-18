import larp_quantmath as lq

print("Derivative:", lq.derivative("x**3 + 2*x", "x"))
print("Integral:", lq.integral("2*x", "x"))
print("Definite integral:", lq.definite_integral("x", "x", 0, 1))
print("Limit:", lq.limit("sin(x)/x", "x", 0))
print("Solve:", lq.solve_expr("x**2 - 4", "x"))