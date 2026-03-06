# 1. retrieve option chain data from Yahoo Finance via yfinance
# 2. package into clean list of OptionContract dataclasses for IV solving
# 3. provide MockDataFetcher for offline testing (don't want to call API for every debug)

from __future__ import annotations

import math
import random
import time
from collections import Counter
from datetime import datetime, date
from dataclasses import dataclass 
from typing import Optional

from models.black_scholes import bs_call_price, bs_put_price

random.seed(42)


@dataclass(slots=True, frozen=True)
class OptionContract:
    ticker: str
    expiry: date
    strike: float
    T: float
    S: float
    market_price: float
    volume: int
    open_interest: int
    option_type: str
    moneyness: float
    r: float = 0.05
    q: float = 0.0

class DataFetcher:
    API_DELAY = 0.1

    def __init__(self, ticker: str, r: float=0.05, q: float=0.0):
        self.ticker = ticker.upper()
        self.r = r
        self.q = q
        self._yf_ticker = None
        self._last_fetch_time = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_fetch_time
        if elapsed < self.API_DELAY:
            time.sleep(self.API_DELAY - elapsed)
        self._last_fetch_time = time.time()

    def _load_ticker(self):
        if self._yf_ticker is not None:
            return
        
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
            "yfinance not installed. Run: pip install yfinance\n"
            "or use --mock for offline testing."
        )

        self._yf_ticker = yf.Ticker(self.ticker)

        try:
            info = self._yf_ticker.fast_info
            if not info:
                raise ValueError(f"Ticker '{self.ticker}' not found or no data available")
        except Exception as e:
            raise ValueError(f"Could not verify ticker '{self.ticker}': {e}")

    def _get_price(self) -> float:
        info = self._yf_ticker.fast_info

        price_fields = ['last_price', 'regularMarketPrice', 'previous_close', 'open', 'day_low']
        
        for field in price_fields:
            price = info.get(field)
            if price and price > 0:
                print(f"[DataFetcher] Got price from '{field}': ${float(price):.2f}")
                return float(price)
        
        try:
            hist = self._yf_ticker.history(period="1d")
            if not hist.empty and 'Close' in hist.columns:
                price = hist['Close'].iloc[-1]
                if price > 0:
                    print(f"[DataFetcher] Got price from history: ${float(price):.2f}")
                    return float(price)
        except Exception:
            pass

        raise ValueError(f"Could not retrive price for {self.ticker}")
    
    def _get_risk(self) -> float:
        try:
            import yfinance as yf
            tbill = yf.Ticker("^IRX")
            rate = tbill.fast_info.get("last_price") or tbill.fast_info.get("previous_close")
            if rate and rate > 0:
                rate_decimal = float(rate) / 100.0
                print(f"[DataFetcher] Risk-free rate from ^IRX: {rate_decimal:.2%}")
                return rate_decimal
        except Exception as e:
            print(f"[DataFetcher] Could not fetch ^IRX rate: {e}")

        print(f"[DataFetcher] Using default risk-free rate: {self.r:.2%}")
        return self.r

    def _compute_T(self, expiry_str: str) -> float:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = date.today()
        days = (expiry_date - today).days
        return max(days/365.25, 0.0)
    
    def _mid_price(self, bid: float, ask: float, last: float) -> Optional[float]:
        if bid > 0 and ask > 0 and ask >= bid:
            return (bid + ask) / 2.0
        if last > 0:
            return last
        return None
    
    def _get_dividend_yield(self) -> float:
        try:
            info = self._yf_ticker.info
            if info:
                div_yield = info.get('dividendYield', 0)
                if div_yield and div_yield > 0:
                    print(f"[DataFetcher] Dividend yield: {div_yield:.2%}")
                    return float(div_yield)
        except Exception:
            pass
        return self.q

    def fetch(self, option_type: str="call", remove_illiquid: bool=True, min_T: float=7/365, max_T: float=2.0, otm_only: bool=False, itm_only: bool=False) -> list[OptionContract]:
        self._load_ticker()
        print(f"[DataFetcher] Fetching data for {self.ticker}...")

        S = self._get_price()
        r = self._get_risk()
        q = self._get_dividend_yield()
        print(f"[DataFetcher] Underlying price: ${S:.2f} | Risk-free rate: {r:.2f} | Div yield: {q:.2%}")

        try:
            expiries = self._yf_ticker.options
        except Exception as e:
            raise ValueError(f"Could not fetch expiration dates for {self.ticker}: {e}")
        
        if not expiries:
            raise ValueError(f"No options available for {self.ticker}")
        
        print(f"[DataFetcher] Found {len(expiries)} expiry dates.")

        contracts: list[OptionContract] = []
        failed = 0

        for i, expiry_str in enumerate(expiries):
            T = self._compute_T(expiry_str)

            if T < min_T or T > max_T:
                continue

            self._rate_limit()

            try: 
                chain = self._yf_ticker.option_chain(expiry_str)
            except Exception as e:
                print(f"[DataFetcher] Could not fetch chain for {expiry_str}: {e}")
                failed = 0
                continue

            df = chain.calls if option_type == "call" else chain.puts

            for _, row in df.iterrows():
                strike = float(row["strike"])
                bid = float(row.get("bid", 0) or 0)
                ask = float(row.get("ask", 0) or 0)
                last = float(row.get("lastPrice", 0) or 0)
                volume = int(row.get("volume", 0) or 0)
                oi = int(row.get("openInterest", 0) or 0)

                market_price = self._mid_price(bid, ask, last)
                if market_price is None or market_price <= 0:
                    continue

                moneyness = strike/S
                
                contract = OptionContract(
                    ticker = self.ticker, 
                    expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date(),
                    strike = strike,
                    T = T,
                    S = S,
                    market_price = market_price,
                    volume = volume,
                    open_interest = oi,
                    option_type = option_type,
                    moneyness = moneyness,
                    r = r, 
                    q = 0.0,
                )
                contracts.append(contract)
            
        if failed > 0:
            print(f"[DataFetcher] Warning: {failed} chains failed to fetch")

        before = len(contracts)

        if remove_illiquid:
            contracts = [c for c in contracts if c.volume > 0]
            
        if option_type == "call":
            if otm_only:
                contracts = [c for c in contracts if c.moneyness > 1.0]
            elif itm_only:
                contracts = [c for c in contracts if c.moneyness < 1.0]
        else:
            if otm_only:
                contracts = [c for c in contracts if c.moneyness < 1.0]
            elif itm_only:
                contracts = [c for c in contracts if c.moneyness > 1.0]
            
        after = len(contracts)
        print(f"[DataFetcher] {before} contracts fetched -> {after} after filtering.")

        if not contracts:
            raise ValueError(
                "No contracts remaining after filtering. "
                "Try relaxing filters: --filter all, adjust --min-maturity/--max-maturity, " 
                "or use a more liquid ticker."
            )

        contracts.sort(key=lambda c: (c.T, c.strike))
        return contracts

class MockDataFetcher:
    def __init__(self,
                 ticker: str="MOCK",
                 S: float=100.0,
                 r: float=0.5,
                 q: float=0.0,
                 base_vol: float=0.20,
                 skew: float=-0.10,
                 curvature: float=0.15):
        
        self.ticker = ticker
        self.S = S
        self.r = r
        self.q = q
        self.base_vol = base_vol
        self.skew = skew
        self.curvature = curvature

    def _smile_vol(self, K: float, T: float) -> float:
        m = K / self.S - 1.0
        term = math.sqrt(T) if T > 0 else 1.0
        vol = self.base_vol + self.skew * m + self.curvature * m ** 2
        return max(vol, 0.01)
    
    def fetch(self,
              option_type: str="call",
              remove_illiquid: bool=True,
              min_T: float=7/365,
              max_T: float=2.0,
              otm_only: bool=False,
              itm_only: bool=False,
              ) -> list[OptionContract]:
        
        expiry_days = [7, 14, 30, 45, 60, 90, 120, 180, 270, 365, 548, 730]

        moneyness_grid = [
            0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 0.925, 0.95, 0.975,
            1.00, 1.025, 1.05, 1.075, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40
        ]

        contracts = []
        print(f"[MockDataFetcher] Generating synthetic chain for {self.ticker} "
              f"(S={self.S}, base_vol={self.base_vol:.0%}, skew={self.skew:+.2f})")
        
        for T in expiry_days:
            if T < min_T or T > max_T:
                continue

            expiry_date = date.fromordinal(date.today().toordinal() + int(T*365.25))
            
            for m in moneyness_grid:
                K = round(self.S*m, 1)
                sigma = self._smile_vol(K, T)

                if option_type == "call":
                    price = bs_call_price(self.S, K, T, self.r, self.q, sigma)
                else:
                    price = bs_put_price(self.S, K, T, self.r, self.q, sigma)

                noise = random.uniform(-0.003, 0.003) * price
                market_price = max(price + noise, 0.01)

                atm_proximity = max(0, 1 - abs(m - 1.0) * 3)
                base_volume = int(random.gauss(atm_proximity * 800, 150))
                volume = max(0, base_volume)

                contracts.append(OptionContract(
                    ticker = self.ticker,
                    expiry = expiry_date,
                    strike = K,
                    T = T,
                    S = self.S,
                    market_price = market_price,
                    volume = volume,
                    open_interest = volume * random.randint(2, 10),
                    option_type = option_type,
                    moneyness = m,
                    r = self.r,
                    q = self.q,
                ))
        
        before = len(contracts)

        if remove_illiquid:
            contracts = [c for c in contracts if c.volume > 0]
        
        if option_type == "call":
            if otm_only:
                contracts = [c for c in contracts if c.moneyness > 1.0]
            elif itm_only:
                contracts = [c for c in contracts if c.moneyness < 1.0]
        else:
            if otm_only:
                contracts = [c for c in contracts if c.moneyness < 1.0]
            elif itm_only:
                contracts = [c for c in contracts if c.moneyness > 1.0]

        after = len(contracts)
        print(f"[MockDataFetcher] {before} contracts generated -> {after} after filtering.")

        contracts.sort(key=lambda c: (c.T, c.strike))
        return contracts
    
if __name__ == "__main__":
    fetcher = MockDataFetcher(ticker="MOCK", S=100.0)
    contracts = fetcher.fetch(option_type="call", remove_illiquid=True)

    print("\nSample contracts (first 8):")
    print(f"{'Expiry':<12} {'Strike':>8} {'T':>6} {'Mono':>6} {'Price':>8} {'Vol':>6} {'OI':>6}")
    for c in contracts[:8]:
        print(f"{c.expiry!s:<12} {c.strike:>8.1f} {c.T:>6.3f} "
              f"{c.moneyness:>6.3f} {c.market_price:>8.4f} "
              f"{c.volume:>6} {c.open_interest:>6}")

    expiry_counts = Counter(c.expiry for c in contracts)
    print("\nContracts per expiry:")
    for expiry, count in sorted(expiry_counts.items()):
        print(f"{expiry} -> {count} contracts")

    print(f"\nTotal contracts: {len(contracts)}")