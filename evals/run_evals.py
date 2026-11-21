"""Run the eval questions against the real assistant.

    python -m evals.run_evals                      # all cases, user "demo"
    python -m evals.run_evals --only titanic_logistic adult_male_count
    python -m evals.run_evals --user alice --pause 4

Each case costs 2-4 Cohere calls. Results (answers, SQL, tool calls) are
written to evals/last_run.json so you can read what the model actually did.
"""
import argparse
import json
import sys
import time
from pathlib import Path

from app.ai.assistant import answer_question
from app.database.connection import get_ingest_connection, get_query_connection
from app.database.ingest import get_user_datasets
from evals.checks import evaluate_case

CASES_PATH = Path(__file__).with_name("cases.json")
LOG_PATH = Path(__file__).with_name("last_run.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="demo")
    parser.add_argument("--only", nargs="*", help="case ids to run")
    parser.add_argument("--pause", type=float, default=3.0,
                        help="seconds between cases (trial keys allow ~20 calls/minute)")
    args = parser.parse_args()

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]

    with get_ingest_connection() as conn:
        datasets = get_user_datasets(conn, args.user)
    available = {d["table_name"] for d in datasets}

    log, passed, failed, skipped, total_calls = [], 0, 0, 0, 0

    with get_query_connection() as query_conn:
        for case in cases:
            if case["table"] not in available:
                print(f"SKIP  {case['id']}: table '{case['table']}' not found for user '{args.user}'")
                skipped += 1
                continue

            try:
                outcome = answer_question(case["question"], args.user, datasets, query_conn)
            except Exception as error:  # keep going so one failure doesn't hide the rest
                print(f"ERROR {case['id']}: {type(error).__name__}: {error}")
                failed += 1
                log.append({"id": case["id"], "error": str(error)})
                continue

            failures, warnings = evaluate_case(case, outcome)
            total_calls += outcome["api_calls"]
            status = "FAIL" if failures else "PASS"
            passed += not failures
            failed += bool(failures)

            print(f"{status}  {case['id']}  ({outcome['api_calls']} calls)")
            for reason in failures:
                print(f"        - {reason}")
            for note in warnings:
                print(f"        ? {note}")

            log.append({
                "id": case["id"], "status": status, "failures": failures, "warnings": warnings,
                "question": case["question"], "answer": outcome["answer"],
                "sql": [q["sql"] for q in outcome["queries"]],
                "tools": [{"tool": a["tool"], "arguments": a["arguments"]} for a in outcome["analyses"]],
                "api_calls": outcome["api_calls"],
            })
            time.sleep(args.pause)

    LOG_PATH.write_text(json.dumps(log, indent=2, default=str), encoding="utf-8")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped. {total_calls} API calls used.")
    print(f"Details: {LOG_PATH}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()