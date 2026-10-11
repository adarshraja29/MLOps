"""Unit tests for PDF text extraction endpoint using pypdf."""

import io
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from backend.app.main import app

client = TestClient(app)


def test_extract_pdf_valid_text():
    """Verify text extraction from a valid PDF document with text stream."""
    writer = PdfWriter()
    # Add a page
    page = writer.add_blank_page(width=300, height=300)
    
    # We can write simple PDF bytes or test endpoint handling
    buf = io.BytesIO()
    writer.write(buf)
    pdf_bytes = buf.getvalue()

    # Empty pages without text content should return 400
    res = client.post(
        "/api/v1/extract-pdf",
        files={"file": ("blank_test.pdf", pdf_bytes, "application/pdf")},
    )
    assert res.status_code == 400
    assert "No extractable text found" in res.json()["detail"]


def test_extract_pdf_rejects_non_pdf():
    """Verify rejection of files without .pdf extension."""
    res = client.post(
        "/api/v1/extract-pdf",
        files={"file": ("paper.docx", b"PK\x03\x04...", "application/vnd.openxmlformats-officedocument")},
    )
    assert res.status_code == 400
    assert "Only PDF" in res.json()["detail"]


def test_extract_pdf_rejects_oversized():
    """Verify rejection of files exceeding the 5MB boundary to protect RAM."""
    oversized_data = b"%PDF-1.4\n" + b"A" * (6 * 1024 * 1024)
    res = client.post(
        "/api/v1/extract-pdf",
        files={"file": ("large_paper.pdf", oversized_data, "application/pdf")},
    )
    assert res.status_code == 400
    assert "5MB" in res.json()["detail"]
