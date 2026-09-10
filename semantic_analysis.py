from __future__ import annotations
import re
from functools import lru_cache
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_SENTENCE_MODEL = "all-MiniLM-L6-v2"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9% ]+", " ", (text or "").lower())).strip()


@lru_cache(maxsize=1)
def _load_sentence_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(DEFAULT_SENTENCE_MODEL)
    except Exception:
        return None


def embedding_backend() -> str:
    return "Sentence-BERT (all-MiniLM-L6-v2)" if _load_sentence_model() is not None else "TF-IDF fallback"


def semantic_similarity_matrix(texts: List[str], prefer_transformer: bool = True) -> Tuple[np.ndarray, str]:
    texts = [str(t) for t in texts]
    if len(texts) == 0:
        return np.empty((0, 0)), "none"
    if len(texts) == 1:
        return np.eye(1), embedding_backend() if prefer_transformer else "TF-IDF"

    if prefer_transformer:
        model = _load_sentence_model()
        if model is not None:
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(embeddings) @ np.asarray(embeddings).T, "Sentence-BERT"

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform([_normalize(t) for t in texts])
    return cosine_similarity(matrix), "TF-IDF"


def find_semantic_duplicates(texts: List[str], threshold: float = 0.78, prefer_transformer: bool = True):
    sims, backend = semantic_similarity_matrix(texts, prefer_transformer=prefer_transformer)
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if float(sims[i, j]) >= threshold:
                pairs.append((i, j, round(float(sims[i, j]) * 100, 1)))
    return sorted(pairs, key=lambda x: x[2], reverse=True), backend


def compare_tfidf_vs_transformer(text_a: str, text_b: str):
    tfidf, _ = semantic_similarity_matrix([text_a, text_b], prefer_transformer=False)
    transformer_model = _load_sentence_model()
    if transformer_model is None:
        return {"TF-IDF": float(tfidf[0, 1]), "Sentence-BERT": None}
    emb = transformer_model.encode([text_a, text_b], normalize_embeddings=True, show_progress_bar=False)
    return {"TF-IDF": float(tfidf[0, 1]), "Sentence-BERT": float(np.dot(emb[0], emb[1]))}


NUMERIC = re.compile(r"\b(\d+(?:\.\d+)?)\s*(ms|milliseconds?|seconds?|minutes?|hours?|days?|%|percent|mb|gb|users?|attempts?)\b", re.I)
NEGATION = re.compile(r"\b(not|never|no|must not|shall not|should not|disable|deny|reject|prevent)\b", re.I)


def find_semantic_conflicts(texts: List[str], similarity_floor: float = 0.58, prefer_transformer: bool = True):
    sims, backend = semantic_similarity_matrix(texts, prefer_transformer=prefer_transformer)
    output = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = float(sims[i, j])
            if sim < similarity_floor:
                continue
            a, b = texts[i], texts[j]
            negation_mismatch = bool(NEGATION.search(a)) != bool(NEGATION.search(b))
            nums_a = {(m.group(1), m.group(2).lower()) for m in NUMERIC.finditer(a)}
            nums_b = {(m.group(1), m.group(2).lower()) for m in NUMERIC.finditer(b)}
            units_a = {unit for _, unit in nums_a}
            units_b = {unit for _, unit in nums_b}
            numeric_mismatch = bool(nums_a and nums_b and (units_a & units_b) and nums_a != nums_b)
            if negation_mismatch or numeric_mismatch:
                reasons = []
                if negation_mismatch:
                    reasons.append("opposite/negated behavior")
                if numeric_mismatch:
                    reasons.append("different numeric constraints")
                output.append((i, j, round(sim * 100, 1), "; ".join(reasons)))
    return sorted(output, key=lambda x: x[2], reverse=True), backend
