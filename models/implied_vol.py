# we want to invert black-scholes to find sigma that reproduces a given market price

# game plan:
#  - try Newton-Raphson first (fast)
#  - if NR diverges -> bisection (slower, but guarantees convergence if a solution indeed exists in [SIGMA_LOW, SIGMA_HIGH])

from dataclasses import dataclass
from models.black_scholes import bs_call_price, bs_vega

import math

TOLERANCE = 1e-6
MAX_ITER_NR = 100
MAX_ITER_BIS = 200
SIGMA_LOW = 1e-6
SIGMA_HIGH = 10.0
MIN_VEGA = 1e-10
MIN_PRICE = 1e-6

@dataclass
class IVResult:
    implied_vol : float | None
    converged: bool
    iterations: int
    method: str
    error: str | None

def _newton_raphson(market_price, S, K, T, r, q) -> IVResult:
    sigma = .20 # choose sensible starting guess (20% vol)

    for i in range(1, MAX_ITER_NR +1):
        price = bs_call_price(S, K, T, r, q, sigma)
        diff = price - market_price
        vega = bs_vega(S, K, T, r, q, sigma)

        if abs(vega) < MIN_VEGA:
            # vega too small -> NR unstable
            return IVResult(None, False, i, 'newton-raphson', 'vega too small, try bisection')
        
        sigma_new = sigma - diff / vega

        sigma_new = max(SIGMA_LOW, min(sigma_new, SIGMA_HIGH))
        if abs(sigma_new - sigma) < TOLERANCE:
            return IVResult(sigma_new, True, i, 'newton-raphson', None)
        
        sigma = sigma_new

    return IVResult(None, False, MAX_ITER_NR, 'newton-raphson', 'max iterations reached, try bisection')

def _bisection(market_price, S, K, T, r, q) -> IVResult:
    lo, hi = SIGMA_LOW, SIGMA_HIGH

    price_lo = bs_call_price(S, K, T, r, q, lo)
    price_hi = bs_call_price(S, K, T, r, q, hi)

    if market_price < price_lo:
        # illiquid
        return IVResult(None, False, 0, 'bisection', f'market price {market_price:.4f} below BS min {price_lo:.4f}')
    
    if market_price > price_hi:
        # possible data error
        return IVResult(None, False, 0, 'bisection', f'market price {market_price:.4f} above BS max {price_hi:.4f}')
    
    for i in range(1, MAX_ITER_BIS + 1):
        mid = (lo + hi) / 2.0
        price = bs_call_price(S, K, T, r, q, mid)
        diff = price - market_price

        if abs(diff) < TOLERANCE:
            return IVResult(mid, True, i, 'bisection', None)
    
        if diff < 0:
            lo = mid
        else:
            hi = mid
    
    return IVResult(None, False, MAX_ITER_BIS, 'bisection', 'max iterations reached')

def compute_implied_vol(market_price: float,
                        S: float, 
                        K: float,
                        T: float,
                        r: float,
                        q: float = 0.0) -> IVResult:
    if T <= 0:
        return IVResult(None, False, 0, 'failed', 'option has expired (T <= 0)')
    
    # illiquid contract
    if market_price < MIN_PRICE:
        return IVResult(None, False, 0, 'failed', f'market price {market_price} is effectively zero')
    
    intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    if market_price < intrinsic - TOLERANCE * 100:
        return IVResult(None, False, 0, 'failed', f'price {market_price:.4f} is below intrinsic value {intrinsic:.4f}')
    
    res = _newton_raphson(market_price, S, K, T, r, q)
    if res.converged:
        return res
    
    return _bisection(market_price, S, K, T, r, q)

if __name__ == "__main__":
    from models.black_scholes import bs_call_price

    test_cases = [
        ("ATM  1yr  20%",  100, 100, 1.00, 0.05, 0.00, 0.20),
        ("OTM  6mo  30%",  100, 110, 0.50, 0.05, 0.00, 0.30),
        ("ITM  3mo  15%",  100,  90, 0.25, 0.04, 0.01, 0.15),
        ("ATM  1mo  50%",  100, 100, 1/12, 0.05, 0.00, 0.50),
        ("Deep OTM  40%",  100, 140, 1.00, 0.05, 0.00, 0.40),
    ]

    all_passed = True
    for desc, S, K, T, r, q, true_sigma in test_cases:
        market_price = bs_call_price(S, K, T, r, q, true_sigma)
        res = compute_implied_vol(market_price, S, K, T, r, q)

        if res.converged:
            err = abs(res.implied_vol - true_sigma)
            status = "PASS" if err < 1e-5 else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"  {status} | {desc:<18} | "
                  f"true={true_sigma:.4f}  solved={res.implied_vol:.6f}  "
                  f"err={err:.2e}  iters={res.iterations}  "
                  f"method={res.method}")
        else:
            all_passed = False
            print(f"  FAIL | {desc:<18} | did not converge: {res.error}")

    print(f"Result: {'OK' if all_passed else 'FAIL'}")

    # test failure cases
    print("\nFailure handling:")
    r1 = compute_implied_vol(0.0,   100, 100, 1.0, 0.05)
    r2 = compute_implied_vol(-1.0,  100, 100, 1.0, 0.05)
    r3 = compute_implied_vol(10.0,  100, 100, 0.0, 0.05)
    print(f"zero price: {r1.error}")
    print(f"negative px: {r2.error}")
    print(f"expired (T=0): {r3.error}")