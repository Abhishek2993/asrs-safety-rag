from asrs_rag.llm import LLMError
from asrs_rag.retrieve import Filters
from asrs_rag.summarise import build_summary_prompt, compute_statistics, summarise


class FakeLLM:
    name = "fake"

    def __init__(self, text):
        self.text = text
        self.prompts = []

    def generate(self, prompt, system=None, max_tokens=512, json_mode=False):
        self.prompts.append(prompt)
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


def test_statistics_are_deterministic_counts(toy_index):
    stats = compute_statistics(toy_index.reports)
    assert stats["n_reports"] == 6
    assert stats["counts_by_year"] == {2018: 1, 2019: 1, 2020: 1, 2021: 1, 2022: 2}
    assert stats["label_counts"]["weather"] == 2
    assert stats["label_counts"]["aircraft_equipment"] == 2


def test_template_summary_without_llm_is_fully_cited(toy_index):
    s = summarise(toy_index, Filters(phases=["approach"]))
    assert s.n_reports == 2 and s.generator == "template"
    assert {e["acn"] for e in s.evidence} == {"2000001", "2000002"}
    assert s.citation_check["passed"] and s.citation_check["citation_precision"] == 1.0
    assert "Weather was coded as a contributing factor in 2 of 2 reports" in s.text


def test_llm_summary_hallucinated_citation_is_caught_and_removed(toy_index):
    llm = FakeLLM(
        "- Wind shear and icing affected approaches [ACN 2000001, 2000002]\n"
        "- A bird strike occurred [ACN 1234567]\n"
        "- Crews were unprepared"
    )
    s = summarise(toy_index, Filters(phases=["approach"]), llm=llm)
    assert s.generator == "fake"
    check = s.citation_check
    assert not check["passed"]
    assert check["invalid_citations"] == ["1234567"]
    assert check["citation_precision"] == round(2 / 3, 4)
    assert s.text == "- Wind shear and icing affected approaches [ACN 2000001, 2000002]"
    assert len(s.removed_statements) == 2
    # The prompt lists only the evidence reports and forbids numbers.
    assert "[ACN 2000001]" in llm.prompts[0] and "[ACN 2000003]" not in llm.prompts[0]
    assert "Do not state counts" in llm.prompts[0]


def test_llm_failure_falls_back_to_template(toy_index):
    s = summarise(toy_index, Filters(), llm=FakeLLM(LLMError("down")))
    assert s.generator == "template" and s.citation_check["passed"]


def test_query_driven_evidence(toy_index):
    s = summarise(toy_index, Filters(), query="fatigue long duty day", n_evidence=1)
    assert [e["acn"] for e in s.evidence] == ["2000003"]


def test_empty_filter_result(toy_index):
    s = summarise(toy_index, Filters(year_from=2030))
    assert s.n_reports == 0 and s.evidence == [] and s.text == ""


def test_prompt_format():
    ev = [
        {
            "acn": "1",
            "date": "2020-01",
            "flight_phase": "Cruise",
            "aircraft_type": "A320",
            "synopsis": "Crew reported turbulence.",
        }
    ]
    p = build_summary_prompt("all reports", ev)
    assert "[ACN 1] 2020-01 | Cruise | A320: Crew reported turbulence." in p
