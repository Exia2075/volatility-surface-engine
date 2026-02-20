# 1. retrieve option chain data from Yahoo Finance via yfinance
# 2. package into clean list of OptionContract dataclasses for IV solving
# 3. provide MockDataFetcher for offline testing (don't want to call API for every debug)

from __future__ import annotations

import math, random
from collections import Counter
from datetime import datetime, date
from dataclasses import dataclass 
from models.black_scholes import bs_call_price

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
    def __init__(self, ticker: str, r: float = 0.05):
        self.ticker = ticker.upper()
        self.r = r
        self._yf_ticker = None

    def _load_ticker(self):
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError(
            "yfinance not installed. Run: pip install yfinance\n"
            "or use MockDataFetcher for offline testing."
        )

        self._yf_ticker = yf.Ticker(self.ticker)

    def _get_price(self) -> float:
        info = self._yf_ticker.fast_info
        price = info.get("last_price") or info.get("regularMarketPrice")
        if not price:
            raise ValueError(f"Could not retrieve price for {self.ticker}.")
        return float(price)
    
    def _get_risk(self) -> float:
        try:
            import yfinance as yf
            tbill = yf.Ticker("^IRX")
            rate = tbill.fast_info.get("last_price")
            if rate:
                return float(rate) / 100.0
        except Exception:
            pass
        return self.r

    def _compute_T(self, expiry_str: str) -> float:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        today = date.today()
        days = (expiry_date - today).days
        return max(days/365.25, 0.0)
    
    def _mid_price(self, bid: float, ask: float, last: float) -> float | None:
        if bid > 0 and ask > 0 and ask >= bid:
            return (bid + ask) / 2.0
        if last > 0:
            return last
        return None
    
    def fetch(self,
              option_type: str="call",
              remove_illiquid: bool=True,
              min_T: float=7/365,
              max_T: float=2.0,
              otm_only: bool=False,
              itm_only: bool=False,
              ) -> list[OptionContract]:
        if self._yf_ticker is None:
            self._load_ticker()

        print(f"[DataFetcher] Fetching data for {self.ticker}...")

        S = self._get_price()
        r = self._get_risk()
        print(f"[DataFetcher] Underlying price: ${S:.2f} | Risk-free rate: {r:.2f}")

        expiries = self._yf_ticker.options
        if not expiries:
            raise ValueError(f"No options available for {self.ticker}")
        print(f"[DataFetcher] Found {len(expiries)} expiry dates.")

        contracts: list[OptionContract] = []

        for expiry_str in expiries:
            T = self._compute_T(expiry_str)

            if T < min_T or T > max_T:
                continue

            try: 
                chain = self._yf_ticker.option_chain(expiry_str)
            except Exception as e:
                print(f"[DataFetcher] Could not fetch chain for {expiry_str}: {e}")
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
                if market_price is None:
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
            
        before = len(contracts)

        if remove_illiquid:
            contracts = [c for c in contracts if c.volume > 0]
            
        if otm_only:
            contracts = [c for c in contracts if c.moneyness > 1.0]
        elif itm_only:
            contracts = [c for c in contracts if c.moneyness < 1.0]
            
        after = len(contracts)
        print(f"[DataFetcher] {before} contracts fetched -> {after} after filtering.")

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
        expiry_Ts = [7/365, 14/365, 30/365, 60/365, 90/365, 180/365, 270/365, 365/365, 548/365, 730/365]
        moneyness_grid = [0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.00, 1.025, 1.05, 1.075, 1.10, 1.15, 1.20, 1.25, 1.30]

        contracts = []
        print(f"[MockDataFetcher] Generating synthetic chain for {self.ticker} "
              f"(S={self.S}, base_vol={self.base_vol:.0%}, skew={self.skew:+.2f})")
        
        for T in expiry_Ts:
            if T < min_T or T > max_T:
                continue

            expiry_date = date.fromordinal(date.today().toordinal() + int(T*365.25))
            
            for m in moneyness_grid:
                K = round(self.S*m, 1)
                sigma = self._smile_vol(K, T)
                price = bs_call_price(self.S, K, T, self.r, self.q, sigma)

                noise = random.uniform(-0.005, 0.005) * price
                market_price = max(price + noise, 0.001)

                atm_proximity = 1 - abs(m - 1.0) * 5
                volume = max(0, int(random.gauss(atm_proximity * 500, 100)))

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
        
        if otm_only:
            contracts = [c for c in contracts if c.moneyness > 1.0]
        elif itm_only:
            contracts = [c for c in contracts if c.moneyness < 1.0]

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