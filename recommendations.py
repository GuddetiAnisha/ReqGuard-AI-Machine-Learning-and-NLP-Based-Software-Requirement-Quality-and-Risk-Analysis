import re
from ambiguity import AMBIGUOUS_TERMS


def generate_recommendations(text: str, result) -> list[str]:
    recs = []
    lower = text.lower()

    found_ambiguous = [t for t in AMBIGUOUS_TERMS if t in lower]
    if found_ambiguous:
        recs.append("Replace ambiguous wording with measurable, verifiable acceptance criteria.")
    if result.completeness < 75:
        recs.append("State the actor/component, trigger or condition, exact behavior, and expected outcome.")
    if result.testability < 75:
        recs.append("Add observable results and quantitative thresholds where applicable.")
    if result.clarity < 75:
        recs.append("Use one main behavior per requirement and avoid combining several responsibilities.")
    if not re.search(r"\b(shall|must|should)\b", lower):
        recs.append("Use consistent requirement language such as 'shall' for mandatory behavior.")
    if not recs:
        recs.append("Requirement is strong. Review domain constraints and edge conditions before approval.")
    return recs


def suggest_rewrite(text: str) -> str:
    lower = text.lower().strip()
    replacements = {
        "quickly": "within 2 seconds",
        "fast": "within 2 seconds",
        "as soon as possible": "within 2 seconds",
        "user-friendly": "with no more than 3 user actions for the primary workflow",
        "efficiently": "within the defined performance threshold",
        "appropriate": "valid",
        "reasonable": "defined",
    }
    rewritten = text.strip()
    for bad, good in replacements.items():
        rewritten = re.sub(re.escape(bad), good, rewritten, flags=re.I)

    if not re.search(r"\b(shall|must|should)\b", rewritten.lower()):
        # Conservative rewrite; do not invent domain behavior beyond converting form.
        if rewritten:
            rewritten = "The system shall " + rewritten[0].lower() + rewritten[1:]
    if rewritten and not rewritten.endswith("."):
        rewritten += "."
    return rewritten
