from __future__ import annotations
from quality_engine import score_requirement
from ml_model import predict_risk, predict_quality_score

RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def hybrid_assessment(text: str, classifier=None, regressor=None):
    rule = score_requirement(text)
    ml_risk, probabilities = predict_risk(text, classifier)
    ml_quality = predict_quality_score(text, regressor)

    if ml_quality is None:
        final_quality = rule.overall
    else:
        # Rules remain slightly more influential because they are explainable.
        final_quality = round(0.6 * rule.overall + 0.4 * ml_quality, 1)

    quality_risk = "LOW" if final_quality >= 80 else "MEDIUM" if final_quality >= 60 else "HIGH"
    candidates = [rule.risk, quality_risk]
    if ml_risk:
        candidates.append(ml_risk)
    final_risk = max(candidates, key=lambda r: RISK_ORDER.get(r, 1))

    return {
        "rule_result": rule,
        "ml_risk": ml_risk,
        "ml_risk_probabilities": probabilities,
        "ml_quality_score": None if ml_quality is None else round(ml_quality, 1),
        "final_quality_score": final_quality,
        "final_risk": final_risk,
    }
