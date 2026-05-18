import larp_quantmath as lq

spot = 100
strike = 100
rate = 0.05
volatility = 0.20
time = 1

call_price = lq.black_scholes_call(spot, strike, rate, volatility, time)
put_price = lq.black_scholes_put(spot, strike, rate, volatility, time)

delta = lq.delta_call(spot, strike, rate, volatility, time)
gamma = lq.gamma(spot, strike, rate, volatility, time)
vega = lq.vega(spot, strike, rate, volatility, time)

print("Call price:", call_price)
print("Put price:", put_price)
print("Delta:", delta)
print("Gamma:", gamma)
print("Vega:", vega)