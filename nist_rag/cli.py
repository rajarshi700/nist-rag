import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict

from .config import AppError, Settings
from .graph import ChatSession, build_graph
from .ingest import build_index
from .llm import GroqLLM
from .retrieval import Retriever


def print_answer(answer, *, trace=False):
    print(f"\n{answer.text}")
    if answer.sources:
        print("\nSources:")
        for source in answer.sources:
            section = f" — Section {source['section']}" if source["section"] else ""
            printed = f" (printed page {source['printed_page']})" if source["printed_page"] else ""
            print(
                f"[{source['number']}] {source['document']}{section} "
                f"— PDF page {source['pdf_page']}{printed}\n    {source['url']}"
            )
    if trace:
        print(f"\nSearch query: {answer.query}\nGraph: {' -> '.join(answer.trace)}")
        print(f"Status: {answer.status}")
    print()


def parser():
    root = argparse.ArgumentParser(description="Ask questions about the NIST AI RMF documents.")
    commands = root.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Download the PDFs and build the local index.")
    ingest.add_argument("--force", action="store_true", help="Rebuild the index from the PDFs.")
    chat = commands.add_parser("chat", help="Start a conversation. Use /new or /quit.")
    chat.add_argument("--trace", action="store_true", help="Show the search query and graph path.")
    ask = commands.add_parser("ask", help="Ask one standalone question.")
    ask.add_argument("question")
    ask.add_argument("--trace", action="store_true")
    ask.add_argument("--json", action="store_true", help="Print a machine-readable result.")
    search = commands.add_parser("search", help="Inspect retrieval without calling an LLM.")
    search.add_argument("question")
    search.add_argument("--json", action="store_true")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        settings = Settings.from_env()
        if args.command == "ingest":
            print("Loading both PDFs and preparing the index. The first run downloads the model.")
            report = build_index(settings, force=args.force)
            verb = "Reused" if report["reused"] else "Indexed"
            print(
                f"{verb} {report['chunks']} chunks from {sum(report['pages'].values())} PDF pages."
            )
            return 0
        with ExitStack() as cleanup:
            llm = None
            if args.command != "search":
                llm = GroqLLM(settings)
                cleanup.callback(llm.close)
            retriever = Retriever(settings)
            cleanup.callback(retriever.close)
            if args.command == "search":
                hits = retriever.search(args.question)
                if args.json:
                    print(json.dumps([asdict(hit) for hit in hits], ensure_ascii=False, indent=2))
                else:
                    for i, hit in enumerate(hits, start=1):
                        chunk = hit.chunk
                        print(f"\n{i}. {chunk.document} — PDF page {chunk.pdf_page}")
                        print(f"Section: {chunk.section or 'not detected'}")
                        print(
                            f"RRF: {hit.score:.4f} | cosine: {hit.dense_score} "
                            f"| BM25: {hit.lexical_score}"
                        )
                        print(chunk.text)
                return 0
            session = ChatSession(build_graph(retriever, llm))
            if args.command == "ask":
                answer = session.ask(args.question)
                if args.json:
                    print(json.dumps(asdict(answer), ensure_ascii=False, indent=2))
                else:
                    print_answer(answer, trace=args.trace)
                return 1 if answer.status == "error" else 0
            print("NIST RMF chat. Ask a question, /new starts fresh, /quit exits.")
            while True:
                try:
                    question = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    return 0
                if question.lower() in {"/quit", "/exit"}:
                    return 0
                if question.lower() == "/new":
                    session.reset()
                    print("Started a new conversation.\n")
                    continue
                if not question:
                    continue
                try:
                    print_answer(session.ask(question), trace=args.trace)
                except AppError as exc:
                    print(f"{exc}\n")
    except (AppError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
