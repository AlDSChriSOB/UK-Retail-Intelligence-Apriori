"""Independent level-wise Apriori mining; existing pair-rule benchmark is unchanged."""
from collections import Counter
from itertools import combinations
from math import ceil, inf

def generate_candidates(previous, k):
    """Self-join L(k-1), then prune candidates with infrequent subsets."""
    previous = {tuple(sorted(x)) for x in previous}
    candidates = set()
    ordered = sorted(previous)
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            if left[:k - 2] != right[:k - 2]:
                continue
            candidate = tuple(sorted(set(left) | set(right)))
            if len(candidate) == k and all(tuple(s) in previous for s in combinations(candidate, k - 1)):
                candidates.add(candidate)
    return sorted(candidates)

def apriori(transactions, min_support=0.05, max_len=None):
    """Return frozenset -> support fraction; transaction items must be strings."""
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1]")
    if max_len is not None and (not isinstance(max_len, int) or max_len < 1):
        raise ValueError("max_len must be positive integer or None")
    baskets = [frozenset(t) for t in transactions]
    n = len(baskets)
    if not n:
        raise ValueError("at least one transaction is required")
    if any(not all(isinstance(x, str) for x in b) for b in baskets):
        raise TypeError("product identifiers must be strings")
    threshold = ceil(min_support * n - 1e-12)
    singles = Counter(item for basket in baskets for item in basket)
    current = sorted((item,) for item, count in singles.items() if count >= threshold)
    supports = {frozenset(x): singles[x[0]] / n for x in current}
    k = 2
    while current and (max_len is None or k <= max_len):
        candidates = generate_candidates(current, k)
        if not candidates:
            break
        counts = Counter()
        for basket in baskets:
            for candidate in candidates:
                if set(candidate).issubset(basket):
                    counts[candidate] += 1
        current = [c for c in candidates if counts[c] >= threshold]
        supports.update({frozenset(c): counts[c] / n for c in current})
        k += 1
    return supports

def association_rules(supports, min_confidence=0.0):
    """All nonempty proper partitions, including multi-item antecedents."""
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    result = []
    for itemset, joint in supports.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset)
        for size in range(1, len(items)):
            for antecedent in combinations(items, size):
                a = frozenset(antecedent)
                b = itemset - a
                sa, sb = supports[a], supports[b]
                confidence = joint / sa
                if confidence + 1e-12 < min_confidence:
                    continue
                result.append({
                    "antecedents": a, "consequents": b, "support": joint,
                    "antecedent_support": sa, "consequent_support": sb,
                    "confidence": confidence, "lift": confidence / sb,
                    "leverage": joint - sa * sb,
                    "conviction": inf if abs(confidence - 1) < 1e-12 else (1 - sb) / (1 - confidence),
                })
    return sorted(result, key=lambda r: (tuple(sorted(r["antecedents"])), tuple(sorted(r["consequents"]))))
