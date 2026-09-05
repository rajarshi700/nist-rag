import pytest

from nist_rag.config import AppError
from nist_rag.graph import ChatSession, build_graph

from .conftest import FakeRetriever, ScriptedLLM


def assessment(verdict="sufficient", ids=None, query=""):
    return {
        "verdict": verdict,
        "relevant_ids": ids if ids is not None else ["S1"],
        "reason": "Test evidence assessment.",
        "suggested_query": query,
    }


def answer(text="The functions are Govern, Map, Measure, and Manage.", ids=None):
    return {"answerable": True, "claims": [{"text": text, "source_ids": ids or ["S1"]}]}


def test_three_turn_conversation_uses_history_and_fresh_retrieval(hit):
    llm = ScriptedLLM(
        assessment(),
        answer(),
        {"query": "Which NIST AI RMF core function is cross-cutting?"},
        assessment(),
        answer("Govern is cross-cutting."),
        {"query": "How does the GOVERN function relate to the other NIST AI RMF functions?"},
        assessment(),
        answer("Govern informs the other three functions."),
    )
    retriever = FakeRetriever([hit])
    graph = build_graph(retriever, llm)
    session = ChatSession(graph, "conversation-a")
    first = session.ask("What are the four core functions?")
    second = session.ask("Which one is cross-cutting?")
    third = session.ask("How does it relate to the others?")
    assert [first.status, second.status, third.status] == ["answered"] * 3
    assert "Govern" in second.text
    assert len(retriever.queries) == 3
    assert "GOVERN" in retriever.queries[-1]
    assert len(llm.calls[2][1]["history"]) == 2
    assert len(llm.calls[5][1]["history"]) == 4
    assert third.trace == ["prepare", "retrieve", "assess", "answer"]
    state = graph.get_state({"configurable": {"thread_id": "conversation-a"}}).values
    assert len(state["messages"]) == 6
    assert state["attempts"] == 1
    assert first.sources[0]["url"].endswith("#page=25")
    assert first.sources[0]["printed_page"] == "20"


def test_insufficient_context_rewrites_once_then_answers(hit):
    llm = ScriptedLLM(
        assessment("insufficient", [], "NIST GOVERN cross-cutting function"),
        assessment(),
        answer("Govern is cross-cutting."),
    )
    retriever = FakeRetriever([hit])
    result = ChatSession(build_graph(retriever, llm)).ask("Which function cuts across the others?")
    assert result.status == "answered"
    assert retriever.queries[-1] == "NIST GOVERN cross-cutting function"
    assert result.trace.count("rewrite") == 1
    assert result.trace.count("retrieve") == 2


def test_retry_is_bounded_and_does_not_generate_without_evidence(hit):
    llm = ScriptedLLM(assessment("insufficient", []), assessment("insufficient", []))
    result = ChatSession(build_graph(FakeRetriever([hit]), llm)).ask(
        "What is the exact required fine?"
    )
    assert result.status == "insufficient_context"
    assert result.trace.count("retrieve") == 2
    assert "answer" not in result.trace
    assert result.sources == []


def test_out_of_scope_stops_without_retry(hit):
    llm = ScriptedLLM(assessment("out_of_scope", []))
    result = ChatSession(build_graph(FakeRetriever([hit]), llm)).ask("Who won the football match?")
    assert result.status == "out_of_scope"
    assert result.sources == []
    assert "rewrite" not in result.trace


def test_empty_retrieval_does_not_call_llm():
    retriever = FakeRetriever([])
    result = ChatSession(build_graph(retriever, ScriptedLLM())).ask("What is GOVERN?")
    assert result.status == "insufficient_context"
    assert len(retriever.queries) == 2


@pytest.mark.parametrize(
    "responses",
    [
        [assessment(ids=["S999"])],
        [assessment(), answer(ids=["S999"])],
        [assessment(), answer(text="A claim [S99].")],
        [assessment(), AppError("The LLM is unavailable.")],
    ],
)
def test_invalid_citations_and_api_errors_are_not_presented_as_answers(hit, responses):
    result = ChatSession(build_graph(FakeRetriever([hit]), ScriptedLLM(*responses))).ask(
        "Functions?"
    )
    assert result.status == "error"
    assert result.sources == []
    assert result.trace[-1] == "abstain"


def test_generator_can_abstain_even_after_positive_grade(hit):
    llm = ScriptedLLM(assessment(), {"answerable": False, "claims": []})
    result = ChatSession(build_graph(FakeRetriever([hit]), llm)).ask("Functions?")
    assert result.status == "insufficient_context"
    assert result.sources == []


def test_threads_and_reset_do_not_share_history(hit):
    llm = ScriptedLLM(assessment(), answer(), assessment(), answer(), assessment(), answer())
    graph = build_graph(FakeRetriever([hit]), llm)
    a, b = ChatSession(graph), ChatSession(graph)
    a.ask("Functions?")
    b.ask("What is cross-cutting?")
    a.reset()
    a.ask("What is Govern?")
    assert len(llm.calls) == 6  # No contextualization call on a fresh thread.


def test_bad_questions_are_rejected_without_advancing_the_graph():
    session = ChatSession(build_graph(FakeRetriever([]), ScriptedLLM()))
    for text in [" ", "x" * 2001]:
        with pytest.raises(AppError):
            session.ask(text)


def test_contextualization_failure_does_not_retrieve_an_ambiguous_question(hit):
    llm = ScriptedLLM(assessment(), answer(), AppError("Rate limited."))
    retriever = FakeRetriever([hit])
    session = ChatSession(build_graph(retriever, llm))
    session.ask("Functions?")
    result = session.ask("Which one?")
    assert result.status == "error"
    assert len(retriever.queries) == 1


def test_retrieval_failure_is_an_operational_error():
    class BrokenRetriever:
        def search(self, query):
            raise OSError("bad database")

    result = ChatSession(build_graph(BrokenRetriever(), ScriptedLLM())).ask("Functions?")
    assert result.status == "error"
    assert result.trace == ["prepare", "retrieve", "abstain"]
