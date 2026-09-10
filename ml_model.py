from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "best_risk_classifier.joblib"
REGRESSOR_PATH = MODEL_DIR / "quality_regressor.joblib"

LABELS = ["LOW", "MEDIUM", "HIGH"]


def _classifier_candidates() -> Dict[str, Pipeline]:
    tfidf = dict(ngram_range=(1, 2), stop_words="english", min_df=1, sublinear_tf=True)
    return {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", LogisticRegression(max_iter=1600, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", LinearSVC(class_weight="balanced", random_state=42)),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf)),
            ("clf", MultinomialNB()),
        ]),
    }


def _evaluate_classifier(model, X_test, y_test):
    preds = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=LABELS).tolist(),
        "classification_report": classification_report(y_test, preds, labels=LABELS, zero_division=0),
        "predictions": preds.tolist(),
    }


def train_and_compare_models(csv_path: str | Path):
    df = pd.read_csv(csv_path)
    if not {"requirement", "risk"}.issubset(df.columns):
        raise ValueError("Training CSV must contain 'requirement' and 'risk' columns.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["requirement"].astype(str), df["risk"].astype(str),
        test_size=0.25, random_state=42, stratify=df["risk"]
    )

    results = {}
    trained = {}
    for name, pipeline in _classifier_candidates().items():
        model = clone(pipeline)
        model.fit(X_train, y_train)
        results[name] = _evaluate_classifier(model, X_test, y_test)
        trained[name] = model

    best_name = max(results, key=lambda name: (results[name]["f1"], results[name]["accuracy"]))
    best_model = trained[best_name]
    joblib.dump(best_model, BEST_MODEL_PATH)

    return best_model, best_name, results, len(X_train), len(X_test)


def train_model(csv_path: str | Path, model_path: str | Path = BEST_MODEL_PATH):
    model, best_name, results, train_size, test_size = train_and_compare_models(csv_path)
    if Path(model_path) != BEST_MODEL_PATH:
        joblib.dump(model, model_path)
    metrics = dict(results[best_name])
    metrics.update({"best_model": best_name, "train_size": train_size, "test_size": test_size, "all_models": results})
    return model, metrics


def load_model(model_path: str | Path = BEST_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        # backward compatibility with earlier project version
        old = MODEL_DIR / "risk_classifier.joblib"
        if old.exists():
            return joblib.load(old)
        return None
    return joblib.load(model_path)


def _decision_confidence(model, text: str):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        return {c: float(p) for c, p in zip(model.classes_, probs)}
    # LinearSVC: turn decision values into softmax-like normalized scores for display only
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function([text])).reshape(-1)
        scores = scores - scores.max()
        probs = np.exp(scores) / np.exp(scores).sum()
        return {c: float(p) for c, p in zip(model.classes_, probs)}
    return None


def predict_risk(text: str, model=None):
    model = model or load_model()
    if model is None:
        return None, None
    label = str(model.predict([text])[0])
    return label, _decision_confidence(model, text)


def train_quality_regressor(csv_path: str | Path):
    df = pd.read_csv(csv_path)
    if not {"requirement", "quality_score"}.issubset(df.columns):
        raise ValueError("Training CSV must contain 'requirement' and 'quality_score' columns.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["requirement"].astype(str), df["quality_score"].astype(float),
        test_size=0.25, random_state=42
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
        ("reg", RandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=2)),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(mean_squared_error(y_test, pred) ** 0.5),
        "r2": float(r2_score(y_test, pred)),
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    joblib.dump(model, REGRESSOR_PATH)
    return model, metrics


def load_quality_regressor():
    return joblib.load(REGRESSOR_PATH) if REGRESSOR_PATH.exists() else None


def predict_quality_score(text: str, model=None):
    model = model or load_quality_regressor()
    if model is None:
        return None
    return float(np.clip(model.predict([text])[0], 0, 100))
