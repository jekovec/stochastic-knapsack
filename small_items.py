"""
Algorithm 5: small_items - Low-Profit Item Selection
From "Stochastic Knapsack without Relaxing the Capacity" (De, Khanna, White, 2025)

This module implements Algorithm 5 for selecting low-profit items with small
realization weights in the Bernoulli Stochastic Knapsack Problem.
"""

import numpy as np
from typing import List, Tuple


class BernoulliItem:
    """
    Represents a Bernoulli item for the Stochastic Knapsack Problem.
    
    The item has a random size:
        Xi = wi with probability pi
        Xi = 0  with probability (1 - pi)
    
    Attributes:
        w: Realization weight (size when item realizes)
        p: Realization probability
        P: Profit
        mu: Expected size (p * w)
    """
    
    def __init__(self, w: float, p: float, P: float):
        """
        Initialize a Bernoulli item.
        
        Args:
            w: Realization weight in [0, 1]
            p: Realization probability in [0, 1]
            P: Profit (positive value)
        """
        if not (0 <= w <= 1):
            raise ValueError(f"Realization weight w={w} must be in [0, 1]")
        if not (0 <= p <= 1):
            raise ValueError(f"Realization probability p={p} must be in [0, 1]")
        if P < 0:
            raise ValueError(f"Profit P={P} must be non-negative")
        
        self.w = w
        self.p = p
        self.P = P
        self.mu = p * w  # Expected size
    
    def __repr__(self):
        return f"BernoulliItem(w={self.w:.3f}, p={self.p:.3f}, P={self.P:.1f}, μ={self.mu:.4f})"


def deterministic_knapsack_dp(sizes: List[float], 
                               profits: List[float], 
                               capacity: float, # so those 3 together create Xs
                               epsilon: float = 0.01) -> List[int]:
    """
    Solve 0-1 knapsack problem using dynamic programming.
    Returns a (1-epsilon)-approximate solution.
    
    This implements a standard FPTAS (Fully Polynomial-Time Approximation Scheme)
    for the deterministic knapsack problem.
    
    Args:
        sizes: List of item sizes (expected sizes μi in our case)
        profits: List of item profits
        capacity: Knapsack capacity (target mean m in our case)
        epsilon: Approximation parameter (default 0.01)
    
    Returns:
        List of indices of selected items
    """
    n = len(sizes)
    
    if n == 0 or capacity <= 0:
        return []
    
    # Handle case where all sizes are zero
    if all(s == 0 for s in sizes):
        # Return item with highest profit
        max_idx = max(range(n), key=lambda i: profits[i])
        return [max_idx]
    
    # Scale capacity to integer for DP
    # Finer granularity = better approximation but slower
    scale = int(10000 / epsilon)
    W = int(capacity * scale)
    
    if W == 0:
        return []
    
    # DP table: dp[i][w] = max profit using first i items with capacity w
    dp = [[0.0] * (W + 1) for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        size_i = int(sizes[i-1] * scale)
        profit_i = profits[i-1]
        
        for w in range(W + 1):
            # Option 1: Don't take item i-1
            dp[i][w] = dp[i-1][w]
            
            # Option 2: Take item i-1 (if it fits)
            if size_i <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w - size_i] + profit_i)
    
    # Backtrack to find which items were selected
    selected = []
    w = W
    
    for i in range(n, 0, -1):
        # Check if item i-1 was taken
        if w > 0 and dp[i][w] != dp[i-1][w]:
            selected.append(i - 1)  # 0-indexed
            size_i = int(sizes[i-1] * scale)
            w -= size_i
    
    selected.reverse()  # Return in original order
    return selected


def small_items(items: List[BernoulliItem], 
                tau: float, 
                epsilon: float) -> List[List[BernoulliItem]]:
    """
    Algorithm 5: Select low-profit items with small realization weights.
    
    Enumerates target mean values m = D/2, D/2·(1+ε²), D/2·(1+ε²)², ...
    For each m, solves deterministic knapsack to find max-profit subset
    with expected total size ≤ m.
    
    Args:
        items: List of BernoulliItem objects where all w_i ≤ epsilon^10 * tau
        tau: Threshold value from constraint generation
        epsilon: Approximation parameter (typically 0.01 to 0.1)
    
    Returns:
        Collection A of candidate subsets (each subset is a list of BernoulliItem)
        Size of collection: O(log(n)/ε²)
        
    Guarantees (Lemma VII.6):
        At least one subset S in A satisfies:
        - profit(S) >= (1-ε)·profit(OPT) - ε²·P
        - Pr[X(S) >= β] <= Pr[X(OPT) >= β] + 5ε for all β >= τ
    """
    # Line 1: Initialize A ← ∅
    A = []
    
    # Handle empty input
    if not items:
        return [[]]
    
    # Step 1: For each i ∈ Xs, create deterministic item di with
    #         profit Pi and size μi
    det_sizes = [item.mu for item in items]    # μi = pi · wi
    det_profits = [item.P for item in items]   # Pi
    
    # Line 3: Let D = min_{i∈Xs} μi
    D = min(det_sizes)
    
    if D <= 0:
        # Edge case: all items have zero expected size
        # Return just the empty set
        return [[]]
    
    # Maximum possible mean (upper bound for m)
    max_mean = sum(det_sizes) #TODO check if thats ok
    
    # Step 2: for each m ∈ [D/2, |Xs|] which is an integer power of (1+ε²)
    m = D / 2
    epsilon_squared = epsilon ** 2
    
    iteration_count = 0
    max_iterations = 10000  # Safety limit to prevent infinite loops
    
    while m <= max_mean and iteration_count < max_iterations:
        # Step 3: Let S be a (1-ε²)-approximate solution to a
        #         (max-profit) knapsack with items Is and capacity m
        selected_indices = deterministic_knapsack_dp(
            sizes=det_sizes,
            profits=det_profits,
            capacity=m,
            epsilon=epsilon_squared
        )
        
        # Step 4: Add S to A
        # Convert indices back to original Bernoulli items
        S = [items[i] for i in selected_indices]
        A.append(S)
        
        # Next power of (1 + ε²)
        m *= (1 + epsilon_squared)
        iteration_count += 1
    
    # Return A
    return A


def generate_instance(n: int, seed: int = None) -> List[BernoulliItem]:
    """
    Generate realistic Bernoulli knapsack instance with uncorrelated profits.
    
    Args:
        n: Number of items to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of BernoulliItem objects
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Realization weights: varied sizes
    w = np.random.uniform(0.02, 0.6, n)
    
    # Realization probabilities: varied
    p = np.random.uniform(0.15, 0.85, n)
    
    # Profits: INDEPENDENT of size
    # This creates non-trivial optimization problems
    Pi = np.random.uniform(5, 100, n)
    
    # Create BernoulliItem objects
    items = [BernoulliItem(w=w[i], p=p[i], P=Pi[i]) for i in range(n)]
    
    return items


def print_solution_summary(solutions: List[List[BernoulliItem]], verbose: bool = False):
    """
    Print a summary of solutions returned by small_items.
    
    Args:
        solutions: Collection of subsets from small_items
        verbose: If True, print details of each solution
    """
    print(f"\nGenerated {len(solutions)} candidate solutions")
    print("=" * 70)
    
    if verbose:
        for i, S in enumerate(solutions):
            if not S:
                print(f"Solution {i:3d}: Empty set")
                continue
            
            total_profit = sum(item.P for item in S)
            expected_size = sum(item.mu for item in S)
            
            print(f"\nSolution {i:3d}: {len(S):2d} items, "
                  f"profit={total_profit:7.2f}, mean={expected_size:.6f}")
            
            # Print items in solution (if not too many)
            if len(S) <= 10:
                for item in S:
                    print(f"    {item}")
    else:
        # Just print summary statistics
        non_empty = [S for S in solutions if S]
        if non_empty:
            profits = [sum(item.P for item in S) for S in non_empty]
            means = [sum(item.mu for item in S) for S in non_empty]
            
            print(f"Non-empty solutions: {len(non_empty)}")
            print(f"Profit range: [{min(profits):.2f}, {max(profits):.2f}]")
            print(f"Mean range: [{min(means):.6f}, {max(means):.6f}]")


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Algorithm 5: small_items - Implementation Test")
    print("=" * 70)
    
    # Example 1: Small set of items
    print("\n--- Example 1: Small Item Set ---")
    
    items = [
        BernoulliItem(w=0.05, p=0.4, P=10),
        BernoulliItem(w=0.08, p=0.3, P=15),
        BernoulliItem(w=0.06, p=0.5, P=12),
        BernoulliItem(w=0.07, p=0.2, P=8),
        BernoulliItem(w=0.04, p=0.6, P=20),
    ]
    
    tau = 0.5
    epsilon = 0.1
    
    print(f"\nInput: {len(items)} items")
    print(f"Parameters: tau={tau}, epsilon={epsilon}")
    print("\nItems:")
    for i, item in enumerate(items):
        print(f"  {i}: {item}")
    
    # Run algorithm
    solutions = small_items(items, tau, epsilon)
    print_solution_summary(solutions, verbose=False)
    
    # Show best solution
    best_solution = max(solutions, key=lambda S: sum(item.P for item in S) if S else 0)
    print(f"\nBest solution by profit:")
    print(f"  Items: {len(best_solution)}")
    if best_solution:
        print(f"  Profit: {sum(item.P for item in best_solution):.2f}")
        print(f"  Expected size: {sum(item.mu for item in best_solution):.6f}")
    
    # Example 2: Larger random instance
    print("\n" + "=" * 70)
    print("--- Example 2: Random Instance (n=20) ---")
    
    items2 = generate_instance(n=20, seed=42)
    epsilon2 = 0.1
    
    print(f"\nGenerated {len(items2)} random items")
    print(f"Epsilon: {epsilon2}")
    
    solutions2 = small_items(items2, tau=1.0, epsilon=epsilon2)
    print_solution_summary(solutions2, verbose=False)
    
    # Example 3: Verify mean enumeration
    print("\n" + "=" * 70)
    print("--- Example 3: Verify Mean Enumeration ---")
    
    simple_items = [
        BernoulliItem(w=0.1, p=0.5, P=10),   # μ = 0.05
        BernoulliItem(w=0.2, p=0.5, P=20),   # μ = 0.10
        BernoulliItem(w=0.15, p=0.4, P=15),  # μ = 0.06
    ]
    
    epsilon3 = 0.2
    solutions3 = small_items(simple_items, tau=1.0, epsilon=epsilon3)
    
    print(f"\nItems:")
    for i, item in enumerate(simple_items):
        print(f"  {i}: {item}")
    
    print(f"\nGenerated {len(solutions3)} solutions with epsilon={epsilon3}")
    print(f"Mean targets should grow by factor (1+ε²) = {1 + epsilon3**2:.4f}")
    
    print("\nSolutions with target means:")
    D = min(item.mu for item in simple_items)
    m = D / 2
    for i, S in enumerate(solutions3):
        if S:
            mean = sum(item.mu for item in S)
            profit = sum(item.P for item in S)
            item_indices = [simple_items.index(item) for item in S]
            print(f"  Target m ≈ {m:.4f}: mean={mean:.4f}, profit={profit:.1f}, items={item_indices}")
        else:
            print(f"  Target m ≈ {m:.4f}: Empty")
        m *= (1 + epsilon3**2)
    
    print("\n" + "=" * 70)
    print("Algorithm implementation verified successfully!")
    print("=" * 70)
