"""Small, inspectable retrieval check; --live also makes real Groq API calls."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from nist_rag.config import AppError, Settings
from nist_rag.graph import ChatSession, build_graph
from nist_rag.llm import GroqLLM
from nist_rag.retrieval import Retriever

ROOT = Path(__file__).resolve().parents[1]


def evaluate(live=False):
    settings = Settings.from_env()
    if live and not settings.api_key:
        raise AppError("Set GROQ_API_KEY before running --live.")
    retriever = Retriever(settings)
    report = {"retrieval": [], "live": None}
    try:
        cases = json.loads((ROOT / "examples" / "retrieval_cases.json").read_text("utf-8"))
        for case in cases:
            hits = retriever.search(case["question"])
            rank = next(
                (
                    i
                    for i, hit in enumerate(hits, start=1)
                    if hit.chunk.document_id == case["document_id"]
                    and hit.chunk.pdf_page in case["pdf_pages"]
                    and all(term in hit.chunk.text.lower() for term in case["terms"])
                ),
                None,
            )
            report["retrieval"].append(
                {
                    "question": case["question"],
                    "passed": rank is not None,
                    "first_relevant_rank": rank,
                    "retrieved_pages": [f"{h.chunk.document_id}:{h.chunk.pdf_page}" for h in hits],
                }
            )
        passed = sum(item["passed"] for item in report["retrieval"])
        report["hit_rate_at_k"] = passed / len(cases)
        report["k"] = settings.top_k
        if live:
            llm = GroqLLM(settings)
            try:
                session = ChatSession(build_graph(retriever, llm))
                questions = [
                    ("What are the four core functions of the AI RMF?", "answered"),
                    ("Which one is cross-cutting?", "answered"),
                    ("How does it relate to the other functions?", "answered"),
                    ("What is confabulation in the Generative AI Profile?", "answered"),
                    ("Who won yesterday's football match?", "out_of_scope"),
                    (
                        "What exact dollar fine does NIST require for every AI RMF violation?",
                        "insufficient_context",
                    ),
                ]
                report["live"] = []
                for question, expected in questions:
                    result = session.ask(question)
                    accepted = {expected}
                    if expected == "insufficient_context":
                        accepted.add("out_of_scope")
                    report["live"].append(
                        {
                            "question": question,
                            "expected_status": expected,
                            "status_passed": result.status in accepted,
                            **asdict(result),
                        }
                    )
                    if result.status == "error":
                        break  # Do not burn quota after an API failure.
            finally:
                llm.close()
        report["note"] = (
            "This is a small development set, not an unbiased benchmark. "
            "Live checks compare statuses only; read the answers and cited pages to judge "
            "correctness, relevance, follow-up resolution, and unsupported claims."
        )
        return report
    finally:
        retriever.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use the real LLM; consumes API quota.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()
    try:
        report = evaluate(args.live)
    except AppError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    serialized = json.dumps(report, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    passed = all(row["passed"] for row in report["retrieval"])
    if report["live"] is not None:
        passed = (
            passed
            and len(report["live"]) == 6
            and all(row["status_passed"] for row in report["live"])
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
