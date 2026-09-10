from pathlib import Path
from ml_model import train_and_compare_models, train_quality_regressor, predict_risk, predict_quality_score
from hybrid import hybrid_assessment
from semantic_analysis import semantic_similarity_matrix
from clustering import cluster_requirements

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data' / 'training_requirements.csv'


def test_multiple_classifiers_train():
    model, best_name, results, train_n, test_n = train_and_compare_models(DATA)
    assert best_name in results
    assert len(results) == 4
    assert train_n > test_n > 0
    label, _ = predict_risk('The system should work quickly.', model)
    assert label in {'LOW','MEDIUM','HIGH'}


def test_quality_regression_and_hybrid():
    reg, metrics = train_quality_regressor(DATA)
    score = predict_quality_score('The API shall respond within 2 seconds.', reg)
    assert 0 <= score <= 100
    h = hybrid_assessment('The application should respond quickly.', classifier=None, regressor=reg)
    assert 0 <= h['final_quality_score'] <= 100
    assert h['final_risk'] in {'LOW','MEDIUM','HIGH'}


def test_semantic_fallback_matrix():
    sims, backend = semantic_similarity_matrix([
        'User shall reset password using email.',
        'Registered users can recover passwords by email.',
        'The payment service shall reject invalid cards.'
    ], prefer_transformer=False)
    assert sims.shape == (3,3)
    assert backend == 'TF-IDF'


def test_clustering_tfidf():
    labels, backend = cluster_requirements([
        'User shall login with email.',
        'User shall reset password by email.',
        'System shall process card payment.',
        'System shall refund a payment.'
    ], n_clusters=2, prefer_transformer=False)
    assert len(labels) == 4
    assert backend == 'TF-IDF'
