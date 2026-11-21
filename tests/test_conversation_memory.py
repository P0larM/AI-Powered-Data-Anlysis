import json
from types import SimpleNamespace as NS

from app.ai import assistant


def text_reply(text):
    return NS(message=NS(tool_calls=None, tool_plan=None, content=[NS(type="text", text=text)]))


def sql_reply(query):
    call = NS(id="c1", type="function", function=NS(name="run_sql", arguments=json.dumps({"query": query})))
    return NS(message=NS(tool_calls=[call], tool_plan="plan", content=None))


class FakeClient:
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def chat(self, model, messages, tools, **kwargs):
        self.sent.append(list(messages))
        return self.replies.pop(0)


DATASETS = [{"table_name": "titanic", "row_count": 1, "columns": [], "profile": {"columns": {}}}]
TURN = {"question": "survival by class?", "answer": "1st class 62%.",
        "sql": ["SELECT pclass FROM titanic"],
        "analyses": [{"tool": "run_logistic_regression", "arguments": {"target": "survived"}}]}


def test_history_becomes_alternating_messages_with_notes():
    messages = assistant.build_history_messages([TURN])
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "survival by class?"
    assert "1st class 62%." in messages[1]["content"]
    assert "SELECT pclass FROM titanic" in messages[1]["content"]
    assert "run_logistic_regression" in messages[1]["content"]


def test_history_is_capped_and_bad_turns_skipped():
    turns = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(20)]
    turns.append({"question": "", "answer": "ignored"})
    messages = assistant.build_history_messages(turns)
    # the cap applies before blank turns are dropped, so the blank one uses up a slot
    assert len(messages) == 2 * (assistant.MAX_HISTORY_TURNS - 1)
    assert messages[-2]["content"] == "q19"


def test_notes_are_length_capped():
    huge = {"question": "q", "answer": "a", "sql": ["SELECT " + "x" * 10_000]}
    notes = assistant._turn_notes(huge)
    assert len(notes) <= assistant.MAX_NOTES_CHARS


def test_follow_up_sees_earlier_turns_and_returns_its_own_turn(monkeypatch):
    monkeypatch.setattr(assistant, "run_user_query", lambda *a, **k: {
        "sql": "SELECT 1", "columns": ["n"], "rows": [[1]], "row_count": 1, "truncated": False})
    client = FakeClient([sql_reply("SELECT 1"), text_reply("Now by sex: 74% of women.")])

    outcome = assistant.answer_question("and by sex?", "demo", DATASETS, None, client=client, history=[TURN])

    first_call = client.sent[0]
    assert [m["role"] for m in first_call] == ["system", "user", "assistant", "user"]
    assert first_call[1]["content"] == "survival by class?"
    assert first_call[-1]["content"] == "and by sex?"
    assert outcome["turn"] == {"question": "and by sex?", "answer": "Now by sex: 74% of women.",
                               "sql": ["SELECT 1"], "analyses": []}


def test_first_question_works_without_history():
    client = FakeClient([text_reply("hi")])
    outcome = assistant.answer_question("hello", "demo", DATASETS, None, client=client)
    assert [m["role"] for m in client.sent[0]] == ["system", "user"]
    assert outcome["turn"]["question"] == "hello"