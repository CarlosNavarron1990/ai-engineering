"""Unit tests for local document extraction (Camino B)."""

from __future__ import annotations

import io

import docx
import pypdf
import pytest
from fastapi import UploadFile

from app.services.attachment_extractor import (
    extract_attachment_text,
    extract_text_from_docx,
    extract_text_from_pdf,
)


def _create_synthetic_pdf(text: str) -> bytes:
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=72, height=72)
    # Note: blank page has no text, so test pypdf reader with synthetic page
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _create_synthetic_docx(paragraphs: list[str]) -> bytes:
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_extract_docx_valid_content() -> None:
    content = _create_synthetic_docx([
        "Authentication module requires OAuth2 and JWT tokens.",
        "Database migrations handled via Alembic.",
    ])
    extracted = extract_text_from_docx(content)
    assert "OAuth2 and JWT" in extracted
    assert "Alembic" in extracted


def test_extract_docx_empty_or_corrupt() -> None:
    corrupted = b"not a valid docx binary"
    result = extract_text_from_docx(corrupted)
    assert "Error extracting DOCX" in result


def test_extract_pdf_corrupt() -> None:
    corrupted = b"not a valid pdf binary"
    result = extract_text_from_pdf(corrupted)
    assert "Error extracting PDF" in result


@pytest.mark.asyncio
async def test_extract_attachment_text_dispatch() -> None:
    # Text file
    text_bytes = b"Architecture design notes:\nMicroservices with RabbitMQ."
    upload_file = UploadFile(
        file=io.BytesIO(text_bytes),
        filename="notes.txt",
    )
    result = await extract_attachment_text(upload_file)
    assert "--- attachment: notes.txt ---" in result
    assert "Microservices with RabbitMQ" in result


@pytest.mark.asyncio
async def test_extract_attachment_unsupported_format() -> None:
    upload_file = UploadFile(
        file=io.BytesIO(b"binary data"),
        filename="binary.exe",
    )
    result = await extract_attachment_text(upload_file)
    assert "Unsupported document format" in result
