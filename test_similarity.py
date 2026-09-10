from similarity import find_duplicates, find_potential_conflicts


def test_duplicate_detection():
    texts = [
        "Users shall reset passwords using a verified email address.",
        "A user shall recover a password using the verified email address.",
        "The system shall export reports as PDF files."
    ]
    duplicates = find_duplicates(texts, threshold=0.25)
    assert any(i == 0 and j == 1 for i, j, _ in duplicates)


def test_conflict_detection_with_negation():
    texts = [
        "The session shall expire after 5 minutes of inactivity.",
        "The session shall not expire after 5 minutes of inactivity."
    ]
    conflicts = find_potential_conflicts(texts, similarity_floor=0.2)
    assert len(conflicts) >= 1
