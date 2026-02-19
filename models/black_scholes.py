import math
from scipy.stats import norm

def _d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

def _d2(d1: float, sigma: float, T: float) -> float:
    return d1 - sigma * math.sqrt(T)

def bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)

    call = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call

def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)

if __name__ == "__main__":
    # quick test: ATM call, 1yr, 5% rate, 20% vol
    # expected: 10.45

    price = bs_call_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
    vega = bs_vega(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.20)
    print(f"bs call price: {price:.4f}")
    print(f"bs vega: {vega:.4f}")

    print(f"expired option (T=0): {bs_call_price(100, 90, 0, 0.05, 0, 0.2):.4f}")
    print(f"deep itm: {bs_call_price(150, 100, 1, 0.05, 0, 0.2):.4f}")
    print(f"deep otm: {bs_call_price(50, 100, 1, 0.05, 0, 0.2):.4f}")