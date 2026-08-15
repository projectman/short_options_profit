# Short Options Profit & Diagonal Spread Analyzer

A Python project for evaluating and ranking candidate short put options to pair with a basis long option position (e.g. **UPS 80P Exp 06/17/2027 @ $3.37**) for diagonal spreads.

---

## ⚠️ Important Option Valuation Rule

> **RULE:** To estimate the price of options, **ALWAYS USE THE MEDIUM PRICE FOR BID/ASK**:
> 
> $$\text{Mid Price} = \frac{\text{Bid} + \text{Ask}}{2}$$

---

## Strategy & Analysis Workflow

1. **Option Selection Criteria**:
   - Filter put options from files in `source/`.
   - Select only **OTM puts** ($\text{Strike} < \text{Spot Price}$).
   - Filter for Delta in range: $|\Delta| \in [0.15, 0.55]$.

2. **80% Extrinsic Profit Target & Days to Target**:
   - Calculate extrinsic value: $\text{Extrinsic} = \text{Mid Price} - \text{Intrinsic Value} = \text{Mid Price}$.
   - Target profit: $80\%$ of extrinsic value ($\text{Target Profit} = 0.80 \times \text{Mid}$).
   - Target exit price: Option theoretical price decays to $20\%$ of current Mid price ($\text{Target Price} = 0.20 \times \text{Mid}$).
   - Black-Scholes solver computes `days_to_target` holding underlying spot price and volatility constant.

3. **Diagonal Spread Risk (At Expiration)**:
   - For a short put strike $K_{\text{short}}$ sold at $\text{Mid}_{\text{short}}$ against the basis long put ($K_{\text{long}}=80$, $\text{Cost}_{\text{long}}=\$3.37$):
     $$\text{Max Risk} = (K_{\text{short}} - 80) + (3.37 - \text{Mid}_{\text{short}})$$
     $$\text{Spread Risk (USD)} = \text{Max Risk} \times 100$$

4. **Daily Profit & Relative Profit Metrics**:
   - $\text{daily\_profit} = \frac{\text{Target Profit (USD)}}{\text{days\_to\_target}}$
   - $\text{daily\_relative\_profit} = \frac{\text{daily\_profit}}{\text{Spread Risk (USD)}} \times 100\%$

5. **Output Table Format**:
   Results are sorted by **Delta** and placed in the `output/` folder:
   - `delta`
   - `index` / `short_put_index` (e.g. `UPS 2026-09-18 100.00P`)
   - `daily_relative_profit` (%)
   - `days_to_target`
   - `profit_usd` ($)

---

## Getting Started

### 1. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 2. Run the Analyzer
```bash
python main.py
```

### 3. Check Outputs
The generated analysis is saved to:
- `output/diagonal_spread_analysis.csv`
- `output/diagonal_spread_analysis.md`

### 4. Run Tests
```bash
pytest
```
