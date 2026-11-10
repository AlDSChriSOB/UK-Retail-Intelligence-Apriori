"""Transparent Apriori implementation for binary transaction baskets."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable
import pandas as pd


def _normalise_transactions(transactions: Iterable[Iterable[str]]) -> list[frozenset[str]]:
    return [frozenset(map(str, row)) for row in transactions]


def _support(itemset: frozenset[str], transactions: list[frozenset[str]]) -> float:
    if not transactions:
        return 0.0
    return sum(itemset.issubset(t) for t in transactions) / len(transactions)


def _join_step(previous: set[frozenset[str]], k: int) -> set[frozenset[str]]:
    """Generate size-k candidates, pruning candidates with infrequent subsets."""
    candidates: set[frozenset[str]] = set()
    previous_list = sorted(previous, key=lambda s: tuple(sorted(s)))
    for i in range(len(previous_list)):
        for j in range(i + 1, len(previous_list)):
            union = previous_list[i] | previous_list[j]
            if len(union) != k:
                continue
            if all(frozenset(s) in previous for s in combinations(union, k - 1)):
                candidates.add(union)
    return candidates


def apriori(
    transactions: Iterable[Iterable[str]],
    min_support: float = 0.01,
    max_len: int | None = None,
) -> pd.DataFrame:
    """Return frequent itemsets with columns itemsets and support."""
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0, 1].")
    tx = _normalise_transactions(transactions)
    if not tx:
        return pd.DataFrame(columns=["itemsets", "support"])

    item_counts: dict[str, int] = {}
    for basket in tx:
        for item in basket:
            item_counts[item] = item_counts.get(item, 0) + 1

    frequent: dict[frozenset[str], float] = {}
    current = {
        frozenset([item]): count / len(tx)
        for item, count in item_counts.items()
        if count / len(tx) >= min_support
    }
    frequent.update(current)

    k = 2
    while current and (max_len is None or k <= max_len):
        candidates = _join_step(set(current), k)
        next_level: dict[frozenset[str], float] = {}
        for candidate in candidates:
            sup = _support(candidate, tx)
            if sup >= min_support:
                next_level[candidate] = sup
        frequent.update(next_level)
        current = next_level
        k += 1

    rows = [{"itemsets": itemset, "support": support}
            for itemset, support in frequent.items()]
    return pd.DataFrame(rows).sort_values(
        ["support"], ascending=False, ignore_index=True
    )


def generate_rules(
    frequent_itemsets: pd.DataFrame,
    min_confidence: float = 0.20,
) -> pd.DataFrame:
    """Generate rules with support, confidence, lift, leverage and conviction."""
    columns = [
        "antecedents", "consequents", "support", "confidence", "lift",
        "leverage", "conviction"
    ]
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1].")
    if frequent_itemsets.empty:
        return pd.DataFrame(columns=columns)

    support_map = {
        frozenset(row.itemsets): float(row.support)
        for row in frequent_itemsets.itertuples(index=False)
    }
    rules = []
    for itemset, joint_support in support_map.items():
        if len(itemset) < 2:
            continue
        items = sorted(itemset)
        for r in range(1, len(items)):
            for antecedent_tuple in combinations(items, r):
                antecedent = frozenset(antecedent_tuple)
                consequent = itemset - antecedent
                ant_support = support_map.get(antecedent)
                con_support = support_map.get(consequent)
                if ant_support is None or con_support is None or ant_support == 0:
                    continue
                confidence = joint_support / ant_support
                if confidence < min_confidence:
                    continue
                lift = confidence / con_support if con_support else float("inf")
                leverage = joint_support - ant_support * con_support
                conviction = ((1 - con_support) / (1 - confidence)
                              if confidence < 1 else float("inf"))
                rules.append({
                    "antecedents": antecedent,
                    "consequents": consequent,
                    "support": joint_support,
                    "confidence": confidence,
                    "lift": lift,
                    "leverage": leverage,
                    "conviction": conviction,
                })
    if not rules:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rules).sort_values(
        ["lift", "confidence", "support"],
        ascending=False,
        ignore_index=True,
    )
