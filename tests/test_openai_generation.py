from legal_rag.generation import generate_openai_answer
from legal_rag.models import Passage, RankedPassage


class FakeModel:
    def invoke(self, prompt, **kwargs):
        assert "Section 438 permits anticipatory bail." in prompt.to_messages()[-1].content
        return "Section 438 permits anticipatory bail. [case:holding:1]"


def test_openai_generation_accepts_retrieved_citations() -> None:
    context = [RankedPassage(Passage("case:holding:1", "case", "Section 438 permits anticipatory bail."), 1.0, 1)]
    answer = generate_openai_answer("What does Section 438 permit?", context, llm=FakeModel())
    assert not answer.abstained
    assert answer.citations == ("case:holding:1",)


def test_openai_generation_rejects_invented_citations() -> None:
    class InvalidModel:
        def invoke(self, prompt, **kwargs):
            return "Unsupported assertion. [made:up]"
    context = [RankedPassage(Passage("case:holding:1", "case", "Section 438 permits anticipatory bail."), 1.0, 1)]
    answer = generate_openai_answer("question", context, llm=InvalidModel())
    assert answer.abstained
