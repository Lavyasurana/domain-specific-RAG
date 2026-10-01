from types import SimpleNamespace

from legal_rag.judge import judge_answer
from legal_rag.models import BenchmarkQuestion, Passage, RankedPassage


class FakeJudgeClient:
    class responses:
        @staticmethod
        def create(**kwargs):
            assert kwargs["model"] == "gpt-4o-mini"
            return SimpleNamespace(output_text='{"faithfulness": 1, "key_fact_recall": 0.5, "citation_support": 1, "notes": "supported"}')


def test_gpt4o_mini_judge_marks_same_family() -> None:
    question = BenchmarkQuestion("q", "What does Section 438 do?", True, ("case:1",), {"case:1": 3}, "holding", key_facts=("anticipatory bail",))
    context = [RankedPassage(Passage("case:1", "case", "Section 438 permits anticipatory bail."), 1, 1)]
    result = judge_answer(question, "It permits anticipatory bail. [case:1]", ("case:1",), context, client=FakeJudgeClient())
    assert result.faithfulness == 1
    assert result.same_family_as_generator
