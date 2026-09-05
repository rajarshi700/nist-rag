from dataclasses import replace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace

from nist_rag.models import Chunk, Hit


@pytest.fixture
def hit():
    return Hit(
        Chunk(
            id="9eac1e88-c848-48b0-8d3c-a7bb6218a957",
            text="The Core has four functions: GOVERN, MAP, MEASURE, and MANAGE. "
            "GOVERN is cross-cutting and informs the other three functions.",
            document_id="rmf",
            document="NIST AI RMF 1.0",
            pdf_page=25,
            printed_page="20",
            section="5. AI RMF Core",
            url="https://nvlpubs.nist.gov/nistpubs/ai/nist.ai.100-1.pdf",
        ),
        score=0.03,
        dense_score=0.8,
    )


@pytest.fixture
def tokenizer():
    tokenizer = Tokenizer(WordPiece({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.hits


class ScriptedLLM:
    """Deterministic graph test double; it does not test a real model's accuracy."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, schema, system, payload, **kwargs):
        self.calls.append((schema, payload))
        if not self.responses:
            raise AssertionError("Unexpected LLM call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return schema.model_validate(response)


def alternate_hit(hit):
    return replace(hit, chunk=replace(hit.chunk, id="79c80936-d26e-48bb-8746-4111ad56d49f"))
