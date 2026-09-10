import re
from dataclasses import dataclass

AMBIGUOUS_TERMS = {
    "quickly": "Replace with a measurable response-time target.",
    "fast": "Define an exact performance threshold.",
    "efficient": "Specify the metric that defines efficiency.",
    "efficiently": "Specify the metric that defines efficiency.",
    "user-friendly": "Define concrete usability criteria.",
    "easy": "State measurable usability expectations.",
    "simple": "Explain what makes the behavior simple.",
    "appropriate": "Specify the exact acceptable behavior/value.",
    "sufficient": "Specify the minimum acceptable amount or threshold.",
    "normally": "Define the operating conditions precisely.",
    "usually": "Replace with an explicit condition or percentage.",
    "often": "Use a measurable frequency.",
    "sometimes": "Specify the exact condition or frequency.",
    "etc": "List the required items explicitly.",
    "and/or": "Choose the intended logical relationship explicitly.",
    "as soon as possible": "Define a maximum acceptable delay.",
    "minimal": "Specify an exact upper bound.",
    "adequate": "Define measurable acceptance criteria.",
    "reasonable": "Replace with a measurable threshold or condition.",
}

WEAK_VERBS = {"support", "handle", "manage", "provide", "allow"}

@dataclass
class AmbiguityResult:
    score: float
    matches: list
    issues: list


def detect_ambiguity(text: str) -> AmbiguityResult:
    lower = text.lower().strip()
    matches, issues = [], []
    for term, advice in AMBIGUOUS_TERMS.items():
        if term in lower:
            matches.append(term)
            issues.append(f'"{term}" is ambiguous. {advice}')

    # Weak modal/optional language
    if re.search(r"\b(may|might|could)\b", lower):
        matches.append("weak modality")
        issues.append("Optional wording such as may/might/could can make mandatory behavior unclear.")

    # Weak generic verbs without enough detail
    words = set(re.findall(r"[a-zA-Z-]+", lower))
    generic = sorted(words.intersection(WEAK_VERBS))
    if generic and len(lower.split()) < 12:
        issues.append("The requirement uses broad action wording; add the exact system behavior and outcome.")

    penalty = min(80, len(matches) * 18 + (10 if generic else 0))
    score = max(0.0, 100.0 - penalty)
    return AmbiguityResult(round(score, 1), matches, issues)
