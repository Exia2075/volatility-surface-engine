# volatility-surface-engine

A Black-Scholes implied volatility surface engine. It supports live data and 3D visualization. It builds a volatility surface from option chains.

---

## Motivation

Imagine you want to understand how the market prices options. You could just assume a flat volatility:

```
vol = 0.20
price = bs_call_price(S=100, K=100, T=1, r=0.05, sigma=vol)
```

In this scenario, every option is priced using the same volatility. However, real markets don't behave that way. Out-of-the-money puts typically have higher implied volatility. Short-dated options behave differently from long-dated ones. Volatility changes across both strike and time. 

The volatility surface engine solves this by computing implied volatility for each individual contract, then constructing the full 3D surface. Instead of a flat assumption, you could do the following:

```
contracts = fetch_option_chain("AAPL")

ivs = []
for contract in contracts:
	iv = implied_volatility(
		price=contract.market_price,
		S=contract.spot
		K=contract.strike,
		T=contract.time_to_expiry,
		r=contract.rate
		)
		ivs.append(iv)

surface = build_surface(ivs)
plot_surface(surface)
```

This ensures that you never assume volatility is flat, but also that you can capture skew and term structure, as well as visualize how the market prices risk.

---

## ⚙️ Installation

Inside your terminal:

```
git clone https://github.com/Exia2075/volatility-surface-engine
cd volatility-surface-engine
pip install -r requirements.txt
```

---

## 👏 Contributing

I love help! Contribute by forking the repo and opening pull requests.

All pull requests should be submitted to the `main` branch.

