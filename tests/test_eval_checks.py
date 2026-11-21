from evals.checks import evaluate_case, ungrounded_numbers


def outcome(rows=None, analyses=None, answer="ok", calls=2):
    queries = [{"sql": "q", "columns": [], "rows": rows}] if rows is not None else []
    return {"answer": answer, "queries": queries, "analyses": analyses or [], "api_calls": calls}


def tool_call(name, arguments, result=None):
    return {"tool": name, "arguments": arguments, "result": result or {}}


def test_expected_number_found_within_tolerance():
    case = {"question": "q", "expect_numbers": [{"value": 96.81, "tolerance": 0.06}]}
    assert evaluate_case(case, outcome(rows=[[1, "female", 96.8]]))[0] == []


def test_truncated_percentage_is_caught():
    case = {"question": "q", "expect_numbers": [{"value": 96.81, "tolerance": 0.06}]}
    failures, _ = evaluate_case(case, outcome(rows=[[1, "female", 96]]))
    assert failures and "96.81" in failures[0]


def test_expected_tool_and_arguments():
    case = {"question": "q", "expect_tool": "compare_groups",
            "expect_tool_args": {"group_column": "sex"}}
    good = outcome(analyses=[tool_call("compare_groups", {"group_column": "Sex"})])
    assert evaluate_case(case, good)[0] == []
    assert evaluate_case(case, outcome())[0]
    wrong = outcome(analyses=[tool_call("compare_groups", {"group_column": "race"})])
    assert "sex" in evaluate_case(case, wrong)[0][0]


def test_answer_text_rules():
    case = {"question": "q", "answer_must_include_any": ["dropped"],
            "answer_must_not_match": [r"\b0\.\d+\s+lower\b"]}
    assert evaluate_case(case, outcome(answer="177 rows dropped"))[0] == []
    failures, _ = evaluate_case(case, outcome(answer="a 0.96 lower odds"))
    assert len(failures) == 2


def test_step_limit_and_api_call_budget():
    failures, _ = evaluate_case({"question": "q", "max_api_calls": 2},
                                outcome(answer="I couldn't get a working answer within the step limit.", calls=6))
    assert len(failures) == 2


def test_ungrounded_numbers_flag_invented_values_only():
    results = outcome(analyses=[tool_call("run_logistic_regression", {},
                                          {"coefficients": [{"Odds Ratio": 12.4062}], "accuracy": 0.7927})])
    answer = "Women had 12.4 times the odds; second class had 2.1 times; accuracy was 79%; p < 0.001."
    assert ungrounded_numbers(answer, results) == [2.1]


def test_grounding_is_a_warning_not_a_failure():
    case = {"question": "q", "grounded": True}
    failures, warnings = evaluate_case(case, outcome(rows=[[5000]], answer="There were 4,321 people."))
    assert failures == [] and warnings