import re
from typing import Annotated, TypedDict
from uuid import uuid4

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import prompts
from .config import AppError
from .models import Answer, Assessment, GroundedAnswer, QueryPlan


class RAGState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    standalone_question: str
    query: str
    evidence: list[dict]
    relevant_ids: list[str]
    verdict: str
    suggested_query: str
    reason: str
    attempts: int
    answer: str
    sources: list[dict]
    error: str
    status: str
    trace: list[str]


def render_answer(result: GroundedAnswer, evidence: list[dict]):
    """Resolve references ourselves so the model cannot invent a source URL or page."""
    if not result.answerable or not result.claims:
        raise AppError("The selected passages did not support an answer.")
    by_id = {item["id"]: item for item in evidence}
    source_numbers, sources, paragraphs = {}, [], []
    for claim in result.claims:
        if re.search(r"\[(?:S\d+|\d+)\]|https?://", claim.text):
            raise AppError("The answer contained references outside the citation fields.")
        references = []
        for source_id in dict.fromkeys(claim.source_ids):
            if source_id not in by_id:
                raise AppError("The answer contained an unverified source reference.")
            item = by_id[source_id]
            key = (item["document_id"], item["pdf_page"], item["section"])
            if key not in source_numbers:
                number = len(sources) + 1
                source_numbers[key] = number
                sources.append(
                    {
                        "number": number,
                        "document": item["document"],
                        "pdf_page": item["pdf_page"],
                        "printed_page": item["printed_page"],
                        "section": item["section"],
                        "url": f"{item['url']}#page={item['pdf_page']}",
                    }
                )
            references.append(source_numbers[key])
        citations = " ".join(f"[{n}]" for n in dict.fromkeys(references))
        paragraphs.append(f"{claim.text.strip()} {citations}")
    return "\n\n".join(paragraphs), sources


def build_graph(retriever, llm, checkpointer=None):
    def prepare(state):
        question = state["messages"][-1].content.strip()
        reset = {
            "question": question,
            "standalone_question": question,
            "query": question,
            "evidence": [],
            "relevant_ids": [],
            "verdict": "",
            "suggested_query": "",
            "reason": "",
            "attempts": 0,
            "answer": "",
            "sources": [],
            "error": "",
            "status": "pending",
            "trace": ["prepare"],
        }
        history = state["messages"][:-1][-6:]
        if history:
            try:
                plan = llm.complete(
                    QueryPlan,
                    prompts.CONTEXTUALIZE,
                    {
                        "history": [{"role": m.type, "content": m.content[:3000]} for m in history],
                        "question": question,
                    },
                    max_tokens=300,
                )
                reset.update(standalone_question=plan.query, query=plan.query)
            except AppError as exc:
                reset["error"] = str(exc)
        return reset

    def retrieve(state):
        update = {"attempts": state["attempts"] + 1, "trace": state["trace"] + ["retrieve"]}
        try:
            hits = retriever.search(state["query"])
            update["evidence"] = [
                {
                    "id": f"S{i}",
                    "text": hit.chunk.text,
                    "document_id": hit.chunk.document_id,
                    "document": hit.chunk.document,
                    "pdf_page": hit.chunk.pdf_page,
                    "printed_page": hit.chunk.printed_page,
                    "section": hit.chunk.section,
                    "url": hit.chunk.url,
                }
                for i, hit in enumerate(hits, start=1)
            ]
        except Exception as exc:
            update.update(
                evidence=[],
                error=f"Retrieval failed ({type(exc).__name__}). Rebuild the index or retry.",
            )
        return update

    def assess(state):
        update = {"trace": state["trace"] + ["assess"], "relevant_ids": []}
        if not state["evidence"]:
            return {
                **update,
                "verdict": "insufficient",
                "suggested_query": "",
                "reason": "No passages found.",
            }
        try:
            result = llm.complete(
                Assessment,
                prompts.ASSESS,
                {
                    "question": state["standalone_question"],
                    "search_query": state["query"],
                    "excerpts": state["evidence"],
                },
            )
            known = {item["id"] for item in state["evidence"]}
            if any(item not in known for item in result.relevant_ids):
                raise AppError("The relevance check returned an unknown source reference.")
            verdict = result.verdict
            if verdict == "sufficient" and not result.relevant_ids:
                verdict = "insufficient"
            update.update(
                verdict=verdict,
                relevant_ids=result.relevant_ids,
                reason=result.reason,
                suggested_query=result.suggested_query.strip(),
            )
        except AppError as exc:
            update["error"] = str(exc)
        return update

    def rewrite(state):
        query = state["suggested_query"]
        if not query or query.casefold() == state["query"].casefold():
            query = (
                "NIST AI risk management framework generative AI profile "
                f"{state['standalone_question']}"
            )
        return {"query": query, "trace": state["trace"] + ["rewrite"]}

    def answer(state):
        update = {"trace": state["trace"] + ["answer"]}
        evidence = [item for item in state["evidence"] if item["id"] in state["relevant_ids"]]
        try:
            result = llm.complete(
                GroundedAnswer,
                prompts.ANSWER,
                {
                    "question": state["standalone_question"],
                    "evidence": evidence,
                },
                max_tokens=1400,
            )
            if not result.answerable or not result.claims:
                return {**update, "verdict": "insufficient"}
            text, sources = render_answer(result, evidence)
            update.update(
                answer=text, sources=sources, status="answered", messages=[AIMessage(content=text)]
            )
        except AppError as exc:
            update["error"] = str(exc)
        return update

    def abstain(state):
        if state.get("error"):
            text = f"I could not complete this request. {state['error']}"
            status = "error"
        elif state.get("verdict") == "out_of_scope":
            text = (
                "I can answer questions about the NIST AI RMF and its Generative AI Profile. "
                "This question is outside that scope."
            )
            status = "out_of_scope"
        else:
            text = (
                "I could not find enough evidence in the two NIST documents "
                "to answer that reliably. "
                "Try naming the function, risk, or section you mean."
            )
            status = "insufficient_context"
        return {
            "answer": text,
            "sources": [],
            "status": status,
            "messages": [AIMessage(content=text)],
            "trace": state["trace"] + ["abstain"],
        }

    def after_assess(state):
        if state.get("error") or state["verdict"] == "out_of_scope":
            return "abstain"
        if state["verdict"] == "sufficient":
            return "answer"
        return "rewrite" if state["attempts"] < 2 else "abstain"

    graph = StateGraph(RAGState)
    for name, node in [
        ("prepare", prepare),
        ("retrieve", retrieve),
        ("assess", assess),
        ("rewrite", rewrite),
        ("answer", answer),
        ("abstain", abstain),
    ]:
        graph.add_node(name, node)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges(
        "prepare", lambda s: "abstain" if s["error"] else "retrieve", ["abstain", "retrieve"]
    )
    graph.add_conditional_edges(
        "retrieve", lambda s: "abstain" if s["error"] else "assess", ["abstain", "assess"]
    )
    graph.add_conditional_edges("assess", after_assess, ["answer", "rewrite", "abstain"])
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges(
        "answer", lambda s: END if s.get("status") == "answered" else "abstain", [END, "abstain"]
    )
    graph.add_edge("abstain", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


class ChatSession:
    def __init__(self, graph, thread_id=None):
        self.graph = graph
        self.thread_id = thread_id or str(uuid4())

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise AppError("Please enter a question.")
        if len(question) > 2000:
            raise AppError("Please keep the question under 2,000 characters.")
        state = self.graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            {
                "configurable": {"thread_id": self.thread_id},
                "recursion_limit": 16,
            },
        )
        return Answer(
            state["answer"], state["sources"], state["query"], state["trace"], state["status"]
        )

    def reset(self):
        self.thread_id = str(uuid4())
