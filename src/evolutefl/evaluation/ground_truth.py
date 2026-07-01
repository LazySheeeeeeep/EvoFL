from __future__ import annotations


def normalize_function_id(value: str) -> str:
    return (value or "").replace("\\", "/").strip().lower()


def hit_at_k(ranked_functions: list[str], ground_truth_functions: list[str], k: int = 5) -> bool:
    predicted = {normalize_function_id(value) for value in ranked_functions[:k]}
    truth = {normalize_function_id(value) for value in ground_truth_functions}
    return bool(predicted.intersection(truth))

