from datetime import datetime, timezone

from paper_picker.core import Paper, _format_authors_apa, format_apa_citation


def test_format_authors_apa_multiple():
    out = _format_authors_apa(["Ada Lovelace", "Alan Turing", "Grace Hopper"])
    assert out == "Lovelace, A., Turing, A., & Hopper, G."


def test_format_apa_citation():
    paper = Paper(
        arxiv_id="1234.5678v1",
        title="A Great Paper",
        abstract="Test",
        authors=["Ada Lovelace", "Alan Turing"],
        published=datetime(2025, 1, 1, tzinfo=timezone.utc),
        abs_url="https://arxiv.org/abs/1234.5678v1",
        pdf_url="https://arxiv.org/pdf/1234.5678v1.pdf",
    )
    citation = format_apa_citation(paper)
    assert citation.startswith("Lovelace, A., & Turing, A. (2025). A Great Paper")
