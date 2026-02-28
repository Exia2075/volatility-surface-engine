import math
from scipy.stats import norm

def _d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0
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

def bs_put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0:
        return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = _d2(d1, sigma, T)

    put = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return put

def bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str="call") -> float:
    if option_type.lower() == "put":
        return bs_put_price(S, K, T, r, q, sigma)
    return bs_call_price(S, K, T, r, q, sigma)

def bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return 0.0

    d1 = _d1(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * norm.pdf(d1) * math.sqrt(T)

def bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str="call") -> float:
    if T <= 0:
        if option_type.lower() == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0 
    d1 = _d1(S, K, T, r, q, sigma)
    if option_type.lower() == "put":
        return math.exp(-q * T) * (norm.cdf(d1) - 1.0)
    return math.exp(-q * T) * norm.cdf(d1)

if __name__ == "__main__":
    # quick test: ATM call, 1yr, 5% rate, 20% vol
    # expected: 10.45, put = 5.57

    S, K, T, r, q, sigma = 100, 100, 1.0, 0.05, 0.0, 0.20

    call_price = bs_call_price(S, K, T, r, q, sigma)
    put_price = bs_call_price(S, K, T, r, q, sigma)
    vega = bs_vega(S, K, T, r, q, sigma)

    print(f"Call price: {call_price:.4f}")
    print(f"Put price: {put_price:.4f}")
    print(f"Vega: {vega:.4f}")

    parity_lhs = call_price - put_price
    parity_rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    print(f"C - P = {parity_lhs:.4f}")
    print(f"Sexp(-qT) - Kexp(-rT) = {parity_rhs:.4f}")
    print(f"Difference: {abs(parity_lhs - parity_rhs):.2e}")

    print(f"Expired option (T=0, ITM): {bs_call_price(110, 100, 0, 0.05, 0, 0.2):.4f}")
    print(f"Expired put (T=0, OTM): {bs_put_price(110, 100, 0, 0.05, 0, 0.2):.4f}")
    print(f"Deep ITM call: {bs_call_price(150, 100, 1, 0.05, 0, 0.2):.4f}")
    print(f"Deep OTM call: {bs_call_price(50, 100, 1, 0.05, 0, 0.2):.4f}")