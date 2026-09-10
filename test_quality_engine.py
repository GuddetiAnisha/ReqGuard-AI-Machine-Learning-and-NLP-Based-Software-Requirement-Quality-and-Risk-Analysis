from quality_engine import score_requirement


def test_clear_measurable_requirement_scores_higher():
    strong = score_requirement("The API shall return a response within 2 seconds after a valid request.")
    weak = score_requirement("The API should respond quickly.")
    assert strong.overall > weak.overall
    assert strong.testability > weak.testability


def test_empty_requirement_is_high_risk():
    result = score_requirement("")
    assert result.overall == 0
    assert result.risk == "HIGH"
