# Algorithm 5: Low-Profit Item Selection (`small_items`)

## Problem Context

The Stochastic Knapsack Problem seeks to maximize profit subject to a probabilistic capacity constraint. For Bernoulli items, each item *i* has a two-valued size distribution:

$$X_i = \begin{cases} w_i & \text{with probability } p_i \\ 0 & \text{with probability } 1-p_i \end{cases}$$

where $w_i$ is the realization size, $p_i$ is the realization probability, and $P_i$ is the profit.

Algorithm 5 is a subalgorithm of Algorithm 3, called after high-profit items are selected. It handles low-profit items (where $P_i < \varepsilon^{15} p_i P$) with small realization sizes ($w_i \leq \varepsilon^{10}\tau$).


## Algorithm Description

### Input
- Set $\mathcal{X}_s$ of Bernoulli items where $w_i \leq \varepsilon^{10}\tau$ for all $i \in \mathcal{X}_s$ (small items)
- Threshold $\tau$ from constraint generation
- Approximation parameter $\varepsilon$ (passed from Algorithm 3)

### Procedure

**Step 1: Deterministic Transformation**

For each item $i$, create a deterministic item with:
- Size: $\mu_i = p_i \cdot w_i$ (expected size)
- Profit: $P_i$ (unchanged)

**Step 2: Target Mean Enumeration**

Generate target means $m \in [D/2, |\mathcal{X}_s|]$ as integer powers of $(1+\varepsilon^2)$, where $D = \min_{i \in \mathcal{X}_s} \mu_i$.

The sequence is: $m_1 = D/2$, $m_{k+1} = m_k \cdot (1+\varepsilon^2)$

**Step 3: Deterministic Knapsack**

For each target mean $m$, find a $(1-\varepsilon^2)$-approximate solution to:

$$\max \sum_{i \in S} P_i \quad \text{subject to} \quad \sum_{i \in S} \mu_i \leq m$$

This is the standard 0-1 knapsack problem and can be solved in polynomial time.

**Step 4: Collection Assembly**

Return collection $\mathcal{A}$ containing all computed subsets.

### Output

Collection $\mathcal{A}$ of $O(\log n / \varepsilon^2)$ candidate subsets.

**Guarantees (Lemma VII.6):** At least one subset satisfies:
- **Profit:** $\text{profit}(S) \geq (1-\varepsilon)\text{profit}(\text{OPT}) - \varepsilon^2 P$
- **Overflow:** $\Pr[X(S) \geq \beta] \leq \Pr[X(\text{OPT}) \geq \beta] + 5\varepsilon$ for all $\beta \geq \tau$

---

## Complexity Analysis

- **Iterations:** $O(\log n / \varepsilon^2)$ target means
- **Per iteration:** $O(n^2/\varepsilon^2)$ for deterministic knapsack FPTAS
- **Total runtime:** $O(n^2 \log n / \varepsilon^4)$ = polynomial

(Note: FPTAS is the standard polynomial-time approximation scheme for the knapsack problem.)

---

## Implementation

A Python implementation is provided in `small_items.py`, including:
- `BernoulliItem` class for item representation
- `small_items()` function implementing Algorithm 5
- `deterministic_knapsack_dp()` FPTAS solver
- Example usage and validation

**Usage:**
```python
from small_items import BernoulliItem, small_items

items = [BernoulliItem(w=0.05, p=0.4, P=10), ...]
solutions = small_items(items, tau=0.5, epsilon=0.1)
```

---

## References

De, A., Khanna, S., & White, N. (2025). "Stochastic Knapsack without Relaxing the Capacity". *IEEE FOCS 2025*, pp. 2410-2445.