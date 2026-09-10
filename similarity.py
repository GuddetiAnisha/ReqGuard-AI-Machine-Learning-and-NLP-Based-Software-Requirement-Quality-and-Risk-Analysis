import re
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

NEGATION_TERMS = {"not", "never", "no", "disable", "disabled", "deny", "reject", "prevent", "forbid", "prohibit"}
ANTONYM_PAIRS = [
    ("enable", "disable"), ("allow", "deny"), ("accept", "reject"),
    ("show", "hide"), ("store", "delete"), ("retain", "delete"),
    ("automatic", "manual"), ("increase", "decrease")
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9% ]+", " ", text.lower())).strip()


def pairwise_similarity(texts: List[str]) -> np.ndarray:
    texts = [_normalize(t) for t in texts]
    if len(texts) < 2:
        return np.eye(len(texts))
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(matrix)


def find_duplicates(texts: List[str], threshold: float = 0.70) -> List[Tuple[int, int, float]]:
    if len(texts) < 2:
        return []
    sims = pairwise_similarity(texts)
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if sims[i, j] >= threshold:
                pairs.append((i, j, round(float(sims[i, j]) * 100, 1)))
    return sorted(pairs, key=lambda x: x[2], reverse=True)


def _has_negation(text: str) -> bool:
    tokens = set(_normalize(text).split())
    return bool(tokens & NEGATION_TERMS)


def _antonym_signal(a: str, b: str) -> bool:
    a, b = _normalize(a), _normalize(b)
    for left, right in ANTONYM_PAIRS:
        if (left in a and right in b) or (right in a and left in b):
            return True
    return False


def find_potential_conflicts(texts: List[str], similarity_floor: float = 0.25) -> List[Tuple[int, int, float, str]]:
    if len(texts) < 2:
        return []
    sims = pairwise_similarity(texts)
    conflicts = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = float(sims[i, j])
            negation_mismatch = _has_negation(texts[i]) != _has_negation(texts[j])
            antonym = _antonym_signal(texts[i], texts[j])
            if sim >= similarity_floor and (negation_mismatch or antonym):
                reason = "negation mismatch" if negation_mismatch else "opposing action terms"
                conflicts.append((i, j, round(sim * 100, 1), reason))
    return sorted(conflicts, key=lambda x: x[2], reverse=True)
