from types import SimpleNamespace

from legal_rag.generation import generate_openai_answer
from legal_rag.models import Passage, RankedPassage


class FakeClient:
    class responses:
        @staticmethod
        def create(**kwargs):
            assert kwargs["model"] == "gpt-4o-mini"
            return SimpleNamespace(output_text="Section 438 permits anticipatory bail. [case:holding:1]")


def test_openai_generation_accepts_retrieved_citations() -> None:
    context = [RankedPassage(Passage("case:holding:1", "case", "Section 438 permits anticipatory bail."), 1.0, 1)]
    answer = generate_openai_answer("What does Section 438 permit?", context, client=FakeClient())
    assert not answer.abstained
    assert answer.citations == ("case:holding:1",)


def test_openai_generation_rejects_invented_citations() -> None:
    class InvalidClient:
        class responses:
            @staticmethod
            def create(**kwargs):
                return SimpleNamespace(output_text="Unsupported assertion. [made:up]")
    context = [RankedPassage(Passage("case:holding:1", "case", "Section 438 permits anticipatory bail."), 1.0, 1)]
    answer = generate_openai_answer("question", context, client=InvalidClient())
    assert answer.abstained
