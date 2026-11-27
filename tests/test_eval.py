import pytest

from asrs_rag import eval as ev


def test_make_split_is_deterministic_and_disjoint():
    acns = [str(1000 + i) for i in range(50)]
    a, b = ev.make_split(acns, 0.2, seed=1), ev.make_split(acns, 0.2, seed=1)
    assert a == b
    assert len(a["test"]) == 10 and len(a["train"]) == 40
    assert not set(a["test"]) & set(a["train"])
    assert ev.make_split(acns, 0.2, seed=2)["test"] != a["test"]


def test_retrieval_proxy_eval_on_toy_index(toy_index):
    res = ev.retrieval_proxy_eval(toy_index, n_queries=6, ks=(1, 3), modes=("hybrid", "bm25"))
    assert res["n_queries"] == 6
    for mode in ("hybrid", "bm25"):
        assert 0.0 <= res[mode]["P@1"] <= 1.0
        assert 0.0 <= res[mode]["MRR"] <= 1.0
    # 6 reports, 3 primary problems with 2 reports each -> chance P = 1/5
    assert res["chance"]["P@1"] == pytest.approx(1 / 5)


def test_labelled_eval(toy_index):
    queries = [
        {"id": "q1", "query": "wind shear on final", "relevant": ["2000001", "2000002"]},
        {"id": "q2", "query": "tired crew long duty day", "relevant": ["2000003"]},
    ]
    res = ev.retrieval_labelled_eval(toy_index, queries, ks=(1, 2), modes=("hybrid",))
    assert res["hybrid"]["P@1"] == 1.0
    per_query = {q["id"]: q for q in res["per_query"]["hybrid"]}
    assert per_query["q2"]["R@1"] == 1.0
    assert per_query["q1"]["R@2"] in (0.5, 1.0)  # 1 or 2 of the 2 relevant reports
    assert res["hybrid"]["R@2"] == pytest.approx(
        (per_query["q1"]["R@2"] + per_query["q2"]["R@2"]) / 2
    )
    assert res["mean_relevant_per_query"] == 1.5


def test_labelled_eval_rejects_unknown_acns(toy_index):
    with pytest.raises(ValueError):
        ev.retrieval_labelled_eval(toy_index, [{"id": "x", "query": "q", "relevant": ["1"]}])


class EchoLLM:
    """Cites the first ACN it sees in the prompt, plus one invented ACN."""

    name = "echo"

    def generate(self, prompt, system=None, max_tokens=512, json_mode=False):
        first = prompt.split("[ACN ")[1].split("]")[0]
        return f"- Pattern seen [ACN {first}]\n- Invented [ACN 7777777]"


def test_faithfulness_eval_counts_invalid_citations(toy_index):
    res = ev.faithfulness_eval(toy_index, EchoLLM(), [{"phases": ["approach"]}, {}])
    assert res["n_summaries"] == 2
    assert res["citation_precision"] == pytest.approx(0.5)
    assert res["statement_coverage"] == pytest.approx(0.5)
    assert res["summaries_fully_grounded"] == 0
    assert all("7777777" not in p["enforced_text"] for p in res["per_summary"])


def test_faithfulness_eval_template_is_fully_grounded(toy_index):
    res = ev.faithfulness_eval(toy_index, None, [{"phases": ["approach"]}])
    assert res["citation_precision"] == 1.0 and res["summaries_fully_grounded"] == 1
