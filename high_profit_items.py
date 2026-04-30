import numpy as np
from experimetnal_space import generate_instance, best_overflow

def knapsack_01(profits, sizes, capacity, precision=10000):
    """
    Standard 0/1 knapsack DP.
    
    Parameters:
    -----------
    profits   : list of floats
    sizes     : list of floats (continuous, will discretize)
    capacity  : float
    precision : bins for discretization
                Decision: 10000 bins gives accuracy 0.0001
                sufficient since capacity = eps^3 is small
    
    Returns:
    --------
    best_profit : float
    selected    : list of indices of selected items
    """
    n            = len(profits)
    cap_bins     = int(capacity * precision)
    size_bins    = [max(1, int(s * precision)) 
                    for s in sizes]
    
    # dp[c] = best profit achievable with capacity c bins
    dp      = np.zeros(cap_bins + 1)
    # track selected items
    selected = [[] for _ in range(cap_bins + 1)]
    
    for i in range(n):
        # go backwards to avoid using same item twice
        for c in range(cap_bins, size_bins[i] - 1, -1):
            new_val = dp[c - size_bins[i]] + profits[i]
            if new_val > dp[c]:
                dp[c]       = new_val
                selected[c] = selected[c - size_bins[i]] \
                              + [i]
    
    return dp[cap_bins], selected[cap_bins]


def group_items(w, p, Pi, indices, eps, n_total):
    """
    Group items by similar pi and Pi values.
    Simplified version of Lemma VII.2.
    
    Simplification decision: use factor (1+eps) for
    both pi and Pi grouping instead of theoretical
    (1+eps^20) and (1+eps). Reason: eps^20 produces
    trivially fine grouping where each item gets its
    own group, making the algorithm meaningless in
    practice. Using (1+eps) for both produces
    meaningful groups while preserving the core idea.
    
    Parameters:
    -----------
    w, p, Pi  : instance arrays
    indices   : list of item indices to group
    eps       : approximation parameter
    n_total   : total number of items
    
    Returns:
    --------
    groups : list of lists of item indices
    """
    if len(indices) == 0:
        return []
    
    factor_p = 1 + eps      # simplified from 1 + eps^20
    factor_P = 1 + eps        # same as paper
    p_min    = eps / (n_total ** 2)
    
    # --- Phase 1: group by pi ---
    phase1_groups = {}
    
    for i in indices:
        if p[i] < p_min:
            continue
        
        # j = bucket index for p[i]
        # p^(j) = factor_p^(-j)
        # j = -log(p[i]) / log(factor_p)
        j = int(np.floor(
            -np.log(p[i]) / np.log(factor_p)
        ))
        
        if j not in phase1_groups:
            phase1_groups[j] = []
        phase1_groups[j].append(i)
    
    #print(f"  Phase 1: {len(phase1_groups)} pi-groups")

    # --- Faza 2: znotraj vsake pi skupine,
    #             razdeli še po Pi ---
    final_groups = []
    
    for j, items in phase1_groups.items():
        
        phase2 = {}
        
        for i in items:
            if Pi[i] <= 0:
                continue
            
            k = int(np.floor(
                np.log(Pi[i]) / np.log(factor_P)
            ))
            
            if k not in phase2:
                phase2[k] = []
            phase2[k].append(i)
        
        for k, group in phase2.items():
            final_groups.append(group)
    
    return final_groups

def multiple_choice_knapsack(group_options, 
                              capacity, 
                              eps,
                              precision=1000):
    """
    Multiple choice knapsack:
    from each group choose exactly one option
    (including option of choosing nothing).
    
    Parameters:
    -----------
    group_options : list of lists of 
                    (profit, exp_size, items) tuples
    capacity      : float, knapsack capacity
    precision     : bins for discretization
    
    Returns:
    --------
    selected_items : list of item indices
    """
    cap_bins = int(capacity * precision)
    
    # dp[c] = (best_profit, items_selected)
    dp       = [0.0] * (cap_bins + 1)
    selected = [[] for _ in range(cap_bins + 1)]
    
    for options in group_options:
        new_dp       = dp.copy()
        new_selected = [s.copy() for s in selected]
        
        for profit, exp_size, items in options:
            size_bins = int(exp_size * precision)
            
            # go backwards to avoid reuse
            for c in range(cap_bins, 
                          size_bins - 1, -1):
                val = dp[c - size_bins] + profit
                if val > new_dp[c]:
                    new_dp[c]       = val
                    new_selected[c] = (
                        selected[c - size_bins] 
                        + items
                    )
        
        dp       = new_dp
        selected = new_selected
    
    return selected[cap_bins]

def select_from_groups(w, p, Pi, groups, eps):
    """
    Select items from groups to maximize profit.
    
    Approach:
    1. For each group G, generate candidate 
       (k, profit, expected_size) triples
       where k items with lowest w_i are taken
    2. Treat each (group, k) pair as a deterministic
       item in a knapsack:
       - profit = sum of Pi for k items
       - size   = sum of w_i * p_i for k items
                  (expected size contribution)
    3. Solve deterministic knapsack over these pairs
    
    This avoids exponential enumeration entirely.
    """
    
    # for each group generate all possible
    # (k items with lowest wi) options
    group_options = []
    
    for group in groups:
        # sort by wi ascending
        sorted_g = sorted(group, key=lambda i: w[i])
        
        options = [(0.0, 0.0, [])]
        
        cumulative_profit   = 0.0
        cumulative_exp_size = 0.0
        
        for k in range(1, len(sorted_g) + 1):
            i = sorted_g[k-1]
            cumulative_profit   += Pi[i]
            cumulative_exp_size += w[i] * p[i]
            
            options.append((
                cumulative_profit,
                cumulative_exp_size,
                sorted_g[:k]
            ))
        
        group_options.append(options)
    
    return group_options


def high_profit_items(w, p, Pi, P_guess, eps, alpha=0.1):
    """
    Algorithm 4: select high-profit items.
    
    Simplified implementation with following decisions:
    
    1. Xh/Xm split: faithful to paper
       Xh: P_i > p_i * P_guess / eps^3
       Xm: everything else
       Note: Xh is always empty for random instances
             due to extreme threshold eps^3
    
    2. Grouping: simplified from Lemma VII.2
       Use factor (1+eps) instead of (1+eps^20)
       Reason: eps^20 produces trivially fine grouping
               where each item gets its own group
    
    3. Selection: multiple choice knapsack over groups
       - For each group: options are k=0,1,...,|G| items
         with lowest wi
       - Knapsack capacity = alpha (by Markov inequality:
         if E[X] <= alpha then Pr[X>=1] <= alpha)
       - After selection: verify actual overflow and
         remove items if needed
    
    Parameters:
    -----------
    w, p, Pi  : instance arrays
    P_guess   : current estimate of OPT profit
    eps       : approximation parameter
    alpha     : overflow probability bound
    
    Returns:
    --------
    final_set    : list of item indices
    final_profit : float
    """
    n       = len(w)
    indices = list(range(n))
    
    # =================================================
    # STEP 1: Split into Xh and Xm
    # =================================================
    # Xh: very high profit relative to p_i
    # P_i > p_i * P_guess / eps^3
    # In practice always empty for random instances
    
    X_h = [i for i in indices
           if Pi[i] > p[i] * P_guess / (eps**3)]
    X_m = [i for i in indices
           if Pi[i] <= p[i] * P_guess / (eps**3)]
    
    # =================================================
    # STEP 2: Handle Xh
    # =================================================
    # Items in Xh almost never realize (sum p_i < eps^3)
    # So solve deterministic knapsack with:
    #   item sizes    = p_i
    #   item profits  = P_i
    #   capacity      = eps^3
    
    if len(X_h) == 0:
        L_h = []
    else:
        profits_h = [Pi[i] for i in X_h]
        sizes_h   = [p[i]  for i in X_h]
        _, sel    = knapsack_01(
            profits_h, sizes_h, eps**3
        )
        L_h = [X_h[i] for i in sel]
    
    # =================================================
    # STEP 3: Group Xm
    # =================================================
    # Group by similar p_i and P_i values
    # Simplified: factor (1+eps) instead of (1+eps^20)
    
    groups = group_items(w, p, Pi, X_m, eps, n)
    
    # =================================================
    # STEP 4: Generate options per group
    # =================================================
    # For each group G, options are:
    # k=0: take nothing
    # k=1: take 1 item with lowest wi
    # k=2: take 2 items with lowest wi
    # ...
    # k=|G|: take all items
    
    group_options = []
    
    for group in groups:
        sorted_g = sorted(group, key=lambda i: w[i])
        
        # always include k=0 option
        options = [(0.0, 0.0, [])]
        
        cumulative_profit   = 0.0
        cumulative_exp_size = 0.0
        
        for k in range(1, len(sorted_g) + 1):
            i = sorted_g[k-1]
            cumulative_profit   += Pi[i]
            cumulative_exp_size += w[i] * p[i]
            
            options.append((
                cumulative_profit,
                cumulative_exp_size,
                sorted_g[:k]
            ))
        
        group_options.append(options)
    
    # =================================================
    # STEP 5: Multiple choice knapsack
    # =================================================
    # From each group choose exactly one option.
    # Capacity = alpha because by Markov inequality:
    #   Pr[X >= 1] <= E[X]
    # So if E[X] <= alpha then Pr[X >= 1] <= alpha
    # This is conservative but guarantees feasibility
    
    knapsack_capacity = min(0.9, np.sqrt(alpha) * 2)
    cap_bins = int(knapsack_capacity * 1000)    
    
    dp       = [0.0] * (cap_bins + 1)
    selected = [[] for _ in range(cap_bins + 1)]
    
    for options in group_options:
        new_dp       = dp.copy()
        new_selected = [s.copy() for s in selected]
        
        for profit, exp_size, items in options:
            if profit == 0:
                continue  # skip k=0
            
            size_bins = max(1, int(exp_size * 1000))
            
            for c in range(cap_bins, 
                           size_bins - 1, -1):
                val = dp[c - size_bins] + profit
                if val > new_dp[c]:
                    new_dp[c]       = val
                    new_selected[c] = (
                        selected[c - size_bins]
                        + items
                    )
        
        dp       = new_dp
        selected = new_selected
    
    selected_m = selected[cap_bins]
    
    # =================================================
    # STEP 6: Combine Xh and Xm selections
    # =================================================
    
    final_set = L_h + selected_m
    
    # =================================================
    # STEP 7: Verify and fix overflow
    # =================================================
    # Multiple choice knapsack uses expected size
    # which is only an approximation of overflow prob.
    # Verify actual overflow and remove items if needed.
    # Remove items with highest w_i * p_i first
    # (largest expected size contribution)
    
    # STEP 7: Verify and fix overflow
# Must ensure actual overflow <= alpha
# (not alpha + eps — that is theoretical guarantee
#  for the full algorithm, not our simplification)

    if len(final_set) > 0:
        actual_overflow = best_overflow(w, p, final_set)
    
        while (actual_overflow > alpha
            and len(final_set) > 0):
            
            worst = max(final_set,
                        key=lambda i: w[i] * p[i])
            final_set.remove(worst)
            
            if len(final_set) > 0:
                actual_overflow = best_overflow(
                    w, p, final_set
                )
            else:
                actual_overflow = 0.0
    
    final_profit = sum(Pi[i] for i in final_set)
    
    return final_set, final_profit


if __name__ == "__main__":
    from experimetnal_space import (
        generate_instance, best_overflow,
        brute_force_opt
    )
    import time
    
    eps = 0.1
    
    # =================================================
    # EXPERIMENT 1: Profit ratio vs OPT (small n)
    # =================================================
    print("=== Eksperiment 1: Razmerje dobičkov ===")
    print(f"{'alpha':>6} {'avg_ratio':>10} "
          f"{'min_ratio':>10} {'avg_ov':>10} "
          f"{'violations':>12}")
    print("-" * 55)
    
    n_trials = 20
    
    for alpha in [0.01, 0.05, 0.1, 0.2, 0.3]:
        ratios     = []
        overflows  = []
        violations = 0
        
        for trial in range(n_trials):
            w, p, Pi = generate_instance(
                12, 'uniform', seed=trial
            )
            P_guess = Pi.sum() / 2
            
            alg_set, alg_profit = high_profit_items(
                w, p, Pi, P_guess, eps, alpha
            )
            opt_profit, _, _ = brute_force_opt(
                w, p, Pi, alpha
            )
            actual_ov = best_overflow(w, p, alg_set)
            ratio     = (alg_profit / opt_profit
                        if opt_profit > 0 else 1.0)
            
            ratios.append(ratio)
            overflows.append(actual_ov)
            if actual_ov > alpha:
                violations += 1
        
        print(f"  {alpha:4.2f} "
              f"{np.mean(ratios):10.4f} "
              f"{min(ratios):10.4f} "
              f"{np.mean(overflows):10.4f} "
              f"{violations:12d}")
    
    # =================================================
    # EXPERIMENT 2: Instance types (small n)
    # =================================================
    print("\n=== Eksperiment 2: Tipi instanc ===")
    print(f"{'type':>12} {'avg_ratio':>10} "
          f"{'min_ratio':>10} {'avg_ov':>10} "
          f"{'violations':>12}")
    print("-" * 55)
    
    alpha = 0.1
    
    for itype in ['uniform', 'high_risk', 
                  'low_risk', 'mixed']:
        ratios     = []
        overflows  = []
        violations = 0
        
        for trial in range(n_trials):
            w, p, Pi = generate_instance(
                12, itype, seed=trial
            )
            P_guess = Pi.sum() / 2
            
            alg_set, alg_profit = high_profit_items(
                w, p, Pi, P_guess, eps, alpha
            )
            opt_profit, _, _ = brute_force_opt(
                w, p, Pi, alpha
            )
            actual_ov = best_overflow(w, p, alg_set)
            ratio     = (alg_profit / opt_profit
                        if opt_profit > 0 else 1.0)
            
            ratios.append(ratio)
            overflows.append(actual_ov)
            if actual_ov > alpha:
                violations += 1
        
        print(f"  {itype:>12} "
              f"{np.mean(ratios):10.4f} "
              f"{min(ratios):10.4f} "
              f"{np.mean(overflows):10.4f} "
              f"{violations:12d}")
    
    # =================================================
    # EXPERIMENT 3: Runtime scaling (large n)
    # =================================================

    print("\n=== Eksperiment 3: Čas izvajanja ===")
    print(f"{'n':>6} {'avg_time':>10} "
          f"{'avg_ov':>10} {'ov_ok':>8}")
    print("-" * 40)
    
    alpha = 0.1
    
    for n in [10, 50, 100, 200, 500, 1000, 10000, 50000, 100000]:
        times     = []
        overflows = []
        ok        = True
        
        for trial in range(5):
            w, p, Pi = generate_instance(
                n, 'uniform', seed=trial
            )
            P_guess = Pi.sum() / 2
            
            start = time.time()
            alg_set, _ = high_profit_items(
                w, p, Pi, P_guess, eps, alpha
            )
            runtime = time.time() - start
            
            actual_ov = best_overflow(w, p, alg_set)
            
            times.append(runtime)
            overflows.append(actual_ov)
            if actual_ov > alpha:
                ok = False
        
        print(f"  {n:4d} "
              f"{np.mean(times):10.4f}s "
              f"{np.mean(overflows):10.4f} "
              f"{'✓' if ok else '✗':>8}")
    
                


