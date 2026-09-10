from __future__ import annotations
from typing import List
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from semantic_analysis import _load_sentence_model


def cluster_requirements(texts: List[str], n_clusters: int = 3, prefer_transformer: bool = True):
    texts = [str(x) for x in texts]
    if len(texts) < 2:
        return [0] * len(texts), "not enough data"
    n_clusters = max(2, min(int(n_clusters), len(texts)))

    if prefer_transformer:
        model = _load_sentence_model()
        if model is not None:
            X = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            backend = "Sentence-BERT embeddings"
        else:
            X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
            backend = "TF-IDF fallback"
    else:
        X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts)
        backend = "TF-IDF"

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    return labels.tolist(), backend
