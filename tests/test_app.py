"""Headless smoke test of the Streamlit app on a toy index (no model downloads)."""

from __future__ import annotations

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


@pytest.fixture()
def toy_index_dir(tmp_path, toy_index, monkeypatch):
    toy_index.save(tmp_path / "index")
    from asrs_rag import config

    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "index")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # hide any local sample index
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setenv("ASRS_RAG_LLM", "none")
    return tmp_path / "index"


def test_app_renders_query_and_trends(toy_index_dir):
    st_testing = pytest.importorskip("streamlit.testing.v1")
    at = st_testing.AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "ACN 200000" in text  # retrieved toy reports are listed
    assert "Supervised (TF-IDF + logistic regression)" in text
    assert "Citation check" in " ".join(s.value for s in at.success) + " ".join(
        w.value for w in at.warning
    )
