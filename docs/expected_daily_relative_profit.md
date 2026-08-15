# Theoretical Framework: Probability-Adjusted Daily Relative Profit & Delta Efficiency

## 1. The Core Dilemma: Nominal Yield vs. Probability of Success

When selling options to capture extrinsic value:
- **High Delta ($|\Delta| \in [0.40, 0.55]$)**: Collects high upfront premium and appears to offer high nominal daily relative profit. However, it carries high directional risk, large drawdowns, and a near $50\%$ chance of expiring ITM or suffering early assignment.
- **Low-to-Mid Delta ($|\Delta| \in [0.15, 0.30]$)**: Collects lower nominal premium, but possesses a high statistical win rate ($70\%\text{–}85\%$) and faster relative theta decay.

Evaluating candidates strictly by nominal daily return creates a selection bias toward high-delta options that take excessive tail risk. Introducing probability adjustments yields a mathematically sound, **risk-adjusted ranking metric**.

---

## 2. Option Probability Metrics

From Black-Scholes and options pricing theory:

### A. Probability of Expiring Out-of-the-Money ($P_{\text{OTM}}$)
The risk-neutral probability that a short put expires worthless at maturity is:
$$P_{\text{OTM}} = 1 - N(-d_2) \approx 1 - |\Delta|$$

Where:
- $N(\cdot)$ is the cumulative standard normal distribution.
- $|\Delta|$ is the absolute value of the put's delta.

### B. Probability of Touch ($P_{\text{touch}}$)
During the trade lifespan, the probability that the underlying asset price touches the strike price at any point prior to expiration is approximately:
$$P_{\text{touch}} \approx 2 \times |\Delta|$$
A $0.35\Delta$ put has an estimated $\approx 70\%$ chance of touching the strike price before maturity.

### C. Probability of Reaching the 80% Extrinsic Profit Target ($P_{\text{target}}$)
Because our strategy exits early as soon as **80% of extrinsic value** has decayed ($t = \text{days\_to\_target}$), $P_{\text{target}}$ is higher than the full-term $P_{\text{OTM}}$ because capital is removed from the market before tail-risk exposure near expiration.

---

## 3. Expected Daily Relative Profit Formulation

To incorporate probability of success into the daily return metric:

$$\text{Expected Daily Rel Profit (\%)} = P_{\text{win}} \times \text{Nominal Daily Rel Profit}$$

$$\text{Expected Daily Rel Profit (\%)} = (1 - |\Delta|) \times \left( \frac{\text{Target Profit USD}}{\text{days\_to\_target} \times \text{Spread Risk USD}} \right) \times 100\%$$

### Why this metric works:
1. **Penalizes High-Risk Bets**: High-delta options ($|\Delta| > 0.45$) have their nominal yield discounted by $\approx 50\%$.
2. **Rewards the "Sweet Spot"**: Strikes in the $0.20\text{–}0.35\Delta$ range typically achieve the highest expected compound return.
3. **Realistic Capital Growth**: Aligns ranking directly with long-term portfolio growth rather than optimistic single-trade payoffs.

---

## 4. Delta Efficiency Metric

$$\text{Delta Efficiency} = \frac{\text{Nominal Daily Rel Profit}}{|\Delta|}$$

- Measures the basis points of daily return earned per unit of directional exposure taken.
- Shorter DTE options (e.g., 30–45 DTE) and OTM strikes ($0.15\text{–}0.25\Delta$) consistently demonstrate the highest Delta Efficiency.
