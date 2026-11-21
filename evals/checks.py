"""Pure checking logic for the eval suite (no database or API needed)."""
import re

DEFAULT_TOLERANCE = 0.05
STEP_LIMIT_MESSAGE = "I couldn't get a working answer"

_NUMBER_IN_TEXT = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*")
_COMMON_VALUES = {0.001, 0.01, 0.05, 100.0}


def _walk_numbers(value):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_numbers(item)


def result_numbers(outcome):
    """Every number in the executed query rows and statistics results."""
    numbers = []
    for query in outcome["queries"]:
        numbers.extend(_walk_numbers(query["rows"]))
    for analysis in outcome["analyses"]:
        numbers.extend(_walk_numbers(analysis["result"]))
    return numbers


def _parse_text_numbers(text):
    """(value, decimals) for each number written in `text`."""
    found = []
    for raw in _NUMBER_IN_TEXT.findall(text):
        cleaned = raw.replace(",", "").rstrip(".")
        if not cleaned or cleaned == "-":
            continue
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        found.append((float(cleaned), decimals))
    return found


def ungrounded_numbers(answer, outcome, question=""):
    """
    Numbers the answer states that match nothing in the results (allowing for
    rounding and for percentages). These are warnings, not failures: derived
    figures like 1/0.28 will show up here, and so will invented ones.
    """
    known = result_numbers(outcome)
    in_question = {value for value, _ in _parse_text_numbers(question)}
    flagged = []

    for value, decimals in _parse_text_numbers(answer):
        if value in _COMMON_VALUES or value in in_question:
            continue
        if value == int(value) and abs(value) <= 10:
            continue
        slack = 0.5 * 10 ** -decimals + 1e-9
        if any(abs(r - value) <= slack or abs(r * 100 - value) <= slack for r in known):
            continue
        flagged.append(value)

    return flagged


def evaluate_case(case, outcome):
    """Returns (failures, warnings) as lists of strings."""
    failures, warnings = [], []
    answer = outcome["answer"] or ""
    numbers = result_numbers(outcome)
    tools_used = [analysis["tool"] for analysis in outcome["analyses"]]

    if not answer.strip():
        failures.append("empty answer")
    if answer.startswith(STEP_LIMIT_MESSAGE):
        failures.append("ran out of steps without an answer")

    for expected in case.get("expect_numbers", []):
        value = expected["value"]
        tolerance = expected.get("tolerance", DEFAULT_TOLERANCE)
        if not any(abs(n - value) <= tolerance for n in numbers):
            failures.append(f"expected {value} (+/-{tolerance}) not found in any query or analysis result")

    wanted_tool = case.get("expect_tool")
    if wanted_tool:
        if wanted_tool not in tools_used:
            failures.append(f"expected tool {wanted_tool}, used {tools_used or 'none'}")
        else:
            for key, wanted in case.get("expect_tool_args", {}).items():
                matches = [
                    a for a in outcome["analyses"]
                    if a["tool"] == wanted_tool
                       and str(a["arguments"].get(key, "")).lower() == str(wanted).lower()
                ]
                if not matches:
                    failures.append(f"{wanted_tool} was not called with {key}={wanted!r}")

    include_any = case.get("answer_must_include_any")
    if include_any and not any(re.search(p, answer, re.IGNORECASE) for p in include_any):
        failures.append(f"answer matched none of {include_any}")

    for pattern in case.get("answer_must_not_match", []):
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            failures.append(f"answer contains forbidden text: {match.group(0)!r}")

    limit = case.get("max_api_calls")
    if limit is not None and outcome.get("api_calls", 0) > limit:
        failures.append(f"used {outcome['api_calls']} API calls (limit {limit})")

    if case.get("grounded"):
        extra = ungrounded_numbers(answer, outcome, case["question"])
        if extra:
            warnings.append(f"numbers in the answer not found in the results: {extra}")

    return failures, warnings