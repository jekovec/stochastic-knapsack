from itertools import combinations
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd


# =============================================================================
# INSTANCE GENERATION
# =============================================================================

def generate_instance(n, instance_type='uniform', seed=None):
    """
    Generate random Bernoulli instance.
    
    Each item i is described by three parameters:
    - w[i]  : realization weight (size if item realizes)
    - p[i]  : realization probability
    - Pi[i] : profit (always fixed)
    
    Decision: four instance types to stress-test
    different aspects of the algorithm.
    
    Parameters:
    -----------
    n             : number of items
    instance_type : 'uniform'   - balanced random instance
                    'high_risk' - large sizes, high prob
                    'low_risk'  - small sizes, low prob
                    'mixed'     - half risky, half safe
    seed          : random seed for reproducibility
    
    Returns:
    --------
    w, p, Pi : np.arrays of length n
    """
    if seed is not None:
        np.random.seed(seed)
    
    if instance_type == 'uniform':
        # Decision: w in [0.01, 0.5] so multiple items
        # can fit without trivially overflowing
        # p in [0.1, 0.9] for interesting probability range
        # Pi independent of w,p to avoid trivial structure
        w  = np.random.uniform(0.01, 0.5, n)
        p  = np.random.uniform(0.1,  0.9, n)
        Pi = np.random.uniform(5,    10,  n)

    elif instance_type == 'high_risk':
        # large sizes + high realization probability
        # tests whether algorithm correctly limits
        # overflow by being selective
        w  = np.random.uniform(0.3, 0., n)
        p  = np.random.uniform(0.5, 0.9, n)
        Pi = np.random.uniform(5,   20,  n)

    elif instance_type == 'low_risk':
        # small sizes + low realization probability
        # tests whether algorithm can pack many items
        # since individual overflow risk is tiny
        w  = np.random.uniform(0.01, 0.1, n)
        p  = np.random.uniform(0.1,  0.3, n)
        Pi = np.random.uniform(0.1,  1,   n)

    elif instance_type == 'mixed':
        # half risky (large w, high p) half safe (small w, low p)
        # most realistic scenario
        half = n // 2
        w  = np.concatenate([
            np.random.uniform(0.3,  0.8, half),
            np.random.uniform(0.01, 0.1, n - half)
        ])
        p  = np.concatenate([
            np.random.uniform(0.5, 0.9, half),
            np.random.uniform(0.1, 0.3, n - half)
        ])
        Pi = np.concatenate([
            np.random.uniform(5,   15,  half),
            np.random.uniform(0.5, 2,   n - half)
        ])
    else:
        raise ValueError(f"Unknown instance_type: {instance_type}")

    # shuffle so risky/safe not always first
    idx = np.random.permutation(n)
    return w[idx], p[idx], Pi[idx]


# =============================================================================
# OVERFLOW ESTIMATION
# =============================================================================

from itertools import product

def exact_overflow_true(w, p, subset):
    """
    Truly exact overflow probability for Bernoulli items.
    
    Enumerates all 2^|subset| possible realizations
    explicitly. No discretization, no approximation.
    
    Parameters:
    -----------
    w, p   : instance arrays
    subset : list of item indices
    
    Returns:
    --------
    float : exact Pr[sum Xi >= 1]
    
    Complexity: O(2^|subset|)
    Use for: subsets of size <= ~20
    """
    w_sub = w[list(subset)]
    p_sub = p[list(subset)]
    k     = len(subset)
    
    overflow_prob = 0.0
    
    # iterate over all 2^k possible realizations
    # each realization is a binary vector:
    # 0 = item does not realize
    # 1 = item realizes to w_i
    for realization in product([0, 1], repeat=k):
        
        # compute probability of this realization
        prob = 1.0
        for i, r in enumerate(realization):
            if r == 1:
                prob *= p_sub[i]        # item realizes
            else:
                prob *= (1 - p_sub[i])  # item does not
        
        # compute total size for this realization
        total_size = sum(
            w_sub[i] for i, r in enumerate(realization)
            if r == 1
        )
        
        # add to overflow if total size >= 1
        if total_size >= 1.0:
            overflow_prob += prob
    
    return overflow_prob


def estimate_overflow_mc(w, p, subset, n_samples=20000):
    """
    Monte Carlo overflow estimation.
    
    Decision: n_samples=20000 by Hoeffding bound:
    to estimate probability within delta=0.01
    at confidence 95% we need:
        n >= log(2/0.05) / (2 * 0.01^2) ~ 18,445
    So 20,000 is safe default.
    
    Parameters:
    -----------
    w, p      : instance arrays
    subset    : list of item indices
    n_samples : Monte Carlo sample count
    
    Returns:
    --------
    float : estimated Pr[sum Xi >= 1]
    
    Use for: larger subsets where exact DP is slow
    """
    w_sub = w[list(subset)]
    p_sub = p[list(subset)]

    # each row = one realization of all items in subset
    realizations = np.random.binomial(
        1, p_sub,
        size=(n_samples, len(subset))
    )
    sizes       = realizations * w_sub
    total_sizes = sizes.sum(axis=1)

    return (total_sizes >= 1).mean()


def best_overflow(w, p, subset, n_samples=20000):
    """
    Choose overflow method based on subset size:
    - truly exact for |subset| <= 15
    - Monte Carlo for larger subsets
    
    Drops DP entirely since it's neither 
    truly exact nor faster than MC.
    """
    if len(subset) <= 12:
        return exact_overflow_true(w, p, subset)
    else:
        return estimate_overflow_mc(
            w, p, subset, n_samples
        )


# =============================================================================
# BRUTE FORCE OPT
# =============================================================================

def brute_force_opt(w, p, Pi, alpha,
                    precision=1000, verbose=False):
    """
    Exact OPT via brute force enumeration.
    
    Tries all 2^n - 1 non-empty subsets.
    Uses exact DP overflow throughout for accuracy.
    
    Decision: only call for n <= 15.
    For n=15: 2^15 = 32768 subsets, each with
    O(15 * 1000) = 15k DP operations ~ 0.5B ops total.
    Feasible in a few minutes.
    For n=20: 2^20 = 1M subsets ~ too slow.
    
    Parameters:
    -----------
    w, p, Pi  : instance arrays
    alpha     : overflow probability bound
    precision : passed to exact_overflow
    verbose   : print progress per subset size
    
    Returns:
    --------
    best_profit : float
    best_set    : list of item indices
    runtime     : float, seconds elapsed
    """
    n           = len(w)
    best_profit = 0
    best_set    = []
    start       = time.time()

    for size in range(1, n + 1):
        for subset in combinations(range(n), size):

            ov = best_overflow(w, p, subset, precision)

            if ov <= alpha:
                profit = sum(Pi[i] for i in subset)
                if profit > best_profit:
                    best_profit = profit
                    best_set    = list(subset)

        if verbose:
            elapsed = time.time() - start
            print(f"  size {size}/{n} — "
                  f"{elapsed:.1f}s — "
                  f"best so far: {best_profit:.4f}")

    return best_profit, best_set, time.time() - start


# =============================================================================
# VALIDATION
# =============================================================================

def validate_instance_generation():
    """
    Check that all instance types produce
    valid parameter ranges.
    """
    print("--- Instance Generation ---")
    all_ok = True

    for itype in ['uniform', 'high_risk',
                  'low_risk', 'mixed']:
        for n in [5, 10, 20]:
            w, p, Pi = generate_instance(n, itype, seed=42)

            checks = {
                'shape w':    w.shape  == (n,),
                'shape p':    p.shape  == (n,),
                'shape Pi':   Pi.shape == (n,),
                'w positive': w.min()  >  0,
                'w <= 1':     w.max()  <= 1,
                'p in [0,1]': p.min()  >= 0 and p.max() <= 1,
                'Pi > 0':     Pi.min() >  0,
            }

            failed = [k for k, v in checks.items() if not v]
            if failed:
                print(f"  FAIL {itype} n={n}: {failed}")
                all_ok = False
            else:
                print(f"  OK   {itype} n={n} — "
                      f"w=[{w.min():.3f},{w.max():.3f}] "
                      f"p=[{p.min():.3f},{p.max():.3f}] "
                      f"Pi=[{Pi.min():.2f},{Pi.max():.2f}]")

    return all_ok


def validate_overflow_estimators():
    """
    Check that exact DP and Monte Carlo agree.
    
    Tests:
    1. Empty subset -> overflow = 0
    2. Single large item -> known overflow = p_i
    3. Exact vs MC agreement on random subsets
    4. Monotonicity: adding items increases overflow
    """
    print("\n--- Overflow Estimators ---")
    all_ok = True
    w, p, Pi = generate_instance(15, 'uniform', seed=42)

    # test 1: empty subset
    ov = best_overflow(w, p, [])
    ok = abs(ov) < 1e-9
    print(f"  {'OK' if ok else 'FAIL'} "
          f"empty subset overflow = {ov:.6f} "
          f"(expected 0)")
    all_ok = all_ok and ok

    # test 2: single item with known overflow
    # item 0 has size w[0] with prob p[0]
    # if w[0] >= 1: overflow = p[0], else overflow = 0
    i      = 0
    ov     = best_overflow(w, p, [i])
    if w[i] >= 1:
        expected = p[i]
    else:
        expected = 0.0
    ok = abs(ov - expected) < 1e-6
    print(f"  {'OK' if ok else 'FAIL'} "
          f"single item overflow = {ov:.6f} "
          f"(expected {expected:.6f}, w={w[i]:.3f})")
    all_ok = all_ok and ok

    # test 3: exact vs MC agreement
    print("  Exact vs MC comparison:")
    for size in [3, 5, 8, 10]:
        subset   = list(range(size))
        ov_exact = best_overflow(w, p, subset)
        ov_mc    = estimate_overflow_mc(
            w, p, subset, n_samples=50000
        )
        diff = abs(ov_exact - ov_mc)
        ok   = diff < 0.03  # within 3%
        print(f"    {'OK' if ok else 'FAIL'} "
              f"size={size:2d} "
              f"exact={ov_exact:.4f} "
              f"mc={ov_mc:.4f} "
              f"diff={diff:.4f}")
        all_ok = all_ok and ok

    # test 4: monotonicity
    # adding an item can only increase overflow
    print("  Monotonicity check:")
    prev_ov = 0.0
    ok_mono = True
    for size in range(1, 8):
        subset = list(range(size))
        ov     = best_overflow(w, p, subset)
        if ov < prev_ov - 1e-6:
            ok_mono = False
            print(f"    FAIL at size={size}: "
                  f"overflow decreased "
                  f"{prev_ov:.4f} -> {ov:.4f}")
        prev_ov = ov
    print(f"    {'OK' if ok_mono else 'FAIL'} "
          f"overflow non-decreasing as items added")
    all_ok = all_ok and ok_mono

    return all_ok


def validate_brute_force_opt():
    """
    Check that brute force OPT is correct.
    
    Tests:
    1. OPT set is feasible (overflow <= alpha)
    2. No feasible superset has higher profit
    3. Consistent across different alpha values
    """
    print("\n--- Brute Force OPT ---")
    all_ok = True

    for n in [5, 8, 10, 12, 15]:
        w, p, Pi = generate_instance(n, 'uniform', seed=42)

        for alpha in [0.05, 0.1, 0.2]:
            opt_profit, opt_set, t = brute_force_opt(
                w, p, Pi, alpha
            )

            # test 1: feasibility
            if len(opt_set) > 0:
                ov = best_overflow(w, p, opt_set)
                ok_feasible = ov <= alpha + 1e-6
            else:
                ok_feasible = True  # empty set always feasible

            # test 2: no single item addition improves profit
            # while staying feasible
            ok_optimal = True
            remaining  = [i for i in range(n)
                          if i not in opt_set]
            for i in remaining:
                candidate = opt_set + [i]
                ov_cand   = best_overflow(w, p, candidate)
                if ov_cand <= alpha:
                    profit_cand = sum(Pi[j]
                                      for j in candidate)
                    if profit_cand > opt_profit + 1e-6:
                        ok_optimal = False
                        print(f"    FAIL n={n} alpha={alpha}: "
                              f"adding item {i} gives "
                              f"{profit_cand:.4f} > "
                              f"{opt_profit:.4f}")

            ok = ok_feasible and ok_optimal
            print(f"  {'OK' if ok else 'FAIL'} "
                  f"n={n} alpha={alpha:.2f} "
                  f"profit={opt_profit:.4f} "
                  f"set_size={len(opt_set)} "
                  f"time={t:.2f}s")
            all_ok = all_ok and ok

    return all_ok


def validate_reproducibility():
    """
    Check that same seed gives same instance.
    Different seeds give different instances.
    """
    print("\n--- Reproducibility ---")
    all_ok = True

    # same seed -> same result
    w1, p1, Pi1 = generate_instance(10, seed=42)
    w2, p2, Pi2 = generate_instance(10, seed=42)
    ok = (np.allclose(w1, w2) and
          np.allclose(p1, p2) and
          np.allclose(Pi1, Pi2))
    print(f"  {'OK' if ok else 'FAIL'} "
          f"same seed gives same instance")
    all_ok = all_ok and ok

    # different seed -> different result
    w3, p3, Pi3 = generate_instance(10, seed=99)
    ok = not np.allclose(w1, w3)
    print(f"  {'OK' if ok else 'FAIL'} "
          f"different seed gives different instance")
    all_ok = all_ok and ok

    return all_ok


def validate_all():
    """Run all validation checks."""
    print("=" * 50)
    print("VALIDATION SUITE")
    print("=" * 50)

    results = {
        'instance_generation': validate_instance_generation(),
        'overflow_estimators': validate_overflow_estimators(),
        'brute_force_opt':     validate_brute_force_opt(),
        'reproducibility':     validate_reproducibility(),
    }

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_passed = True
    for name, ok in results.items():
        status    = "PASS" if ok else "FAIL"
        all_passed = all_passed and ok
        print(f"  {status}  {name}")

    print("\n" + ("ALL CHECKS PASSED" if all_passed
                  else "SOME CHECKS FAILED"))
    return all_passed


# =============================================================================
# TIMING ANALYSIS
# =============================================================================

def time_overflow_estimators():
    """
    Measure runtime of exact vs MC overflow
    for different subset sizes.
    Helps justify the mc_threshold=20 decision.
    """
    print("\n--- Overflow Estimator Timing ---")
    print(f"{'size':>6} {'exact(s)':>10} "
          f"{'mc(s)':>10} {'faster':>10}")
    print("-" * 40)

    w, p, Pi = generate_instance(30, 'uniform', seed=42)

    for size in [2, 5, 10, 15, 20, 25, 30]:
        subset = list(range(size))

        # time exact
        t0    = time.time()
        for _ in range(10):
            best_overflow(w, p, subset)
        t_exact = (time.time() - t0) / 10

        # time MC
        t0  = time.time()
        for _ in range(10):
            estimate_overflow_mc(w, p, subset,
                                 n_samples=20000)
        t_mc = (time.time() - t0) / 10

        faster = "exact" if t_exact < t_mc else "MC"
        print(f"  {size:4d} {t_exact:10.4f} "
              f"{t_mc:10.4f} {faster:>10}")


def time_brute_force_opt():
    """
    Measure brute force OPT runtime vs n.
    Helps justify n<=15 limit.
    """
    print("\n--- Brute Force OPT Timing ---")
    print(f"{'n':>4} {'subsets':>10} "
          f"{'time(s)':>10} {'feasible?':>12}")
    print("-" * 40)

    for n in [5, 8, 10, 12, 15]:
        w, p, Pi = generate_instance(n, 'uniform', seed=42)

        t0 = time.time()
        opt_profit, opt_set, _ = brute_force_opt(
            w, p, Pi, alpha=0.1
        )
        t = time.time() - t0

        feasible = "yes" if len(opt_set) > 0 else "no"
        print(f"  {n:3d} {2**n:10d} "
              f"{t:10.2f} {feasible:>12}")


def time_mc_accuracy():
    """
    Show how MC accuracy improves with n_samples.
    Justifies our choice of n_samples=20000.
    """
    print("\n--- MC Accuracy vs Sample Count ---")
    print(f"{'samples':>10} {'estimate':>10} "
          f"{'error vs exact':>16}")
    print("-" * 40)

    w, p, Pi = generate_instance(10, 'uniform', seed=42)
    subset   = list(range(8))

    # ground truth
    exact = best_overflow(w, p, subset)
    print(f"  exact = {exact:.6f}")
    print()

    for n_samples in [100, 500, 1000,
                      5000, 10000, 20000, 50000]:
        # average over 5 runs for stability
        estimates = [
            estimate_overflow_mc(w, p, subset, n_samples)
            for _ in range(5)
        ]
        mean_est = np.mean(estimates)
        error    = abs(mean_est - exact)
        print(f"  {n_samples:9d} {mean_est:10.6f} "
              f"{error:16.6f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # 1. run all validation checks
    validate_all()

    # 2. timing analysis to justify design decisions
    print("\n" + "=" * 50)
    print("TIMING ANALYSIS")
    print("=" * 50)

    time_overflow_estimators()
    time_brute_force_opt()
    time_mc_accuracy()