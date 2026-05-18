import math
import random
import time

import larp_quantmath as lq

print("NOTE: For a fair Rust benchmark, run: maturin develop --release")
print()

def pure_python_monte_carlo_call(
    spot,
    strike,
    rate,
    volatility,
    time_to_expiry,
    simulations,
    seed,
):
    random.seed(seed)

    total_payoff = 0.0

    for _ in range(simulations):
        z = random.gauss(0, 1)

        final_price = spot * math.exp(
            (rate - 0.5 * volatility**2) * time_to_expiry
            + volatility * math.sqrt(time_to_expiry) * z
        )

        payoff = max(final_price - strike, 0.0)
        total_payoff += payoff

    average_payoff = total_payoff / simulations
    return math.exp(-rate * time_to_expiry) * average_payoff


def benchmark():
    spot = 100
    strike = 100
    rate = 0.05
    volatility = 0.20
    time_to_expiry = 1
    simulations = 1_000_000
    seed = 42

    start = time.perf_counter()
    rust_result = lq.monte_carlo_call(
        spot,
        strike,
        rate,
        volatility,
        time_to_expiry,
        simulations,
        seed,
    )
    rust_time = time.perf_counter() - start

    start = time.perf_counter()
    python_result = pure_python_monte_carlo_call(
        spot,
        strike,
        rate,
        volatility,
        time_to_expiry,
        simulations,
        seed,
    )
    python_time = time.perf_counter() - start

    speedup = python_time / rust_time

    print("Monte Carlo Call Benchmark")
    print("--------------------------")
    print(f"Simulations: {simulations:,}")
    print(f"Rust result: {rust_result:.6f}")
    print(f"Python result: {python_result:.6f}")
    print(f"Rust time: {rust_time:.6f} seconds")
    print(f"Python time: {python_time:.6f} seconds")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    benchmark()