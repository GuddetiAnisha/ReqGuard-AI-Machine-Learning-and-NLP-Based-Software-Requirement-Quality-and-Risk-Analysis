import re
from dataclasses import dataclass, asdict
from ambiguity import detect_ambiguity

MEASURABLE_PATTERN = re.compile(
    r"(\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|s|seconds?|minutes?|hours?|%|percent|mb|gb|kb|users?|requests?|attempts?)\b)",
    re.I,
)
CONDITION_WORDS = {"when", "if", "after", "before", "during", "while", "unless", "upon"}
ACTOR_WORDS = {"user", "admin", "administrator", "system", "application", "service", "operator", "customer"}
OUTCOME_WORDS = {"display", "save", "send", "reject", "accept", "create", "update", "delete", "redirect", "notify", "return", "record", "process", "prevent", "allow"}

@dataclass
class QualityResult:
    clarity: float
    completeness: float
    testability: float
    specificity: float
    ambiguity: float
    overall: float
    risk: str
    issues: list

    def to_dict(self):
        return asdict(self)


def _bounded(v):
    return round(max(0.0, min(100.0, v)), 1)


def score_requirement(text: str) -> QualityResult:
    text = (text or "").strip()
    if not text:
        return QualityResult(0, 0, 0, 0, 0, 0, "HIGH", ["Requirement is empty."])

    lower = text.lower()
    words = re.findall(r"\b\w+[\w-]*\b", lower)
    word_count = len(words)
    ambiguity_result = detect_ambiguity(text)
    issues = list(ambiguity_result.issues)

    # Clarity
    clarity = 100.0
    if word_count < 6:
        clarity -= 30
        issues.append("Requirement is very short and may lack context.")
    if word_count > 45:
        clarity -= 15
        issues.append("Requirement is long; consider splitting it into smaller atomic requirements.")
    if text.count(" and ") >= 2:
        clarity -= 12
        issues.append("Requirement may contain multiple behaviors joined together; consider making it atomic.")
    clarity -= (100 - ambiguity_result.score) * 0.55

    # Completeness: actor + behavior + condition/context + outcome/object
    actor = any(w in lower for w in ACTOR_WORDS)
    behavior = any(w in lower for w in OUTCOME_WORDS) or bool(re.search(r"\b(shall|must|should)\b", lower))
    condition = any(re.search(rf"\b{re.escape(w)}\b", lower) for w in CONDITION_WORDS)
    has_object = word_count >= 10
    completeness = 30 + 25 * actor + 25 * behavior + 10 * condition + 10 * has_object
    if not actor:
        issues.append("Actor or responsible component is not explicit.")
    if not behavior:
        issues.append("Expected system behavior is not explicit.")
    if not condition:
        issues.append("Trigger, condition, or operating context is not stated.")

    # Testability
    testability = 45.0
    if re.search(r"\b(shall|must|should)\b", lower):
        testability += 15
    if MEASURABLE_PATTERN.search(text):
        testability += 25
    if any(w in lower for w in OUTCOME_WORDS):
        testability += 15
    testability -= (100 - ambiguity_result.score) * 0.35
    if not MEASURABLE_PATTERN.search(text):
        issues.append("No measurable threshold/value was detected; add one where relevant.")

    # Specificity
    specificity = 40.0
    if actor:
        specificity += 15
    if condition:
        specificity += 15
    if MEASURABLE_PATTERN.search(text):
        specificity += 20
    if word_count >= 10:
        specificity += 10
    specificity -= len(ambiguity_result.matches) * 8

    clarity = _bounded(clarity)
    completeness = _bounded(completeness)
    testability = _bounded(testability)
    specificity = _bounded(specificity)
    ambiguity = ambiguity_result.score

    overall = _bounded(
        0.30 * clarity
        + 0.25 * completeness
        + 0.25 * testability
        + 0.20 * specificity
    )

    if overall >= 80:
        risk = "LOW"
    elif overall >= 60:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    # preserve order, remove duplicate issue strings
    issues = list(dict.fromkeys(issues))
    return QualityResult(clarity, completeness, testability, specificity, ambiguity, overall, risk, issues)
