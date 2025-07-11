
import pytest
from fastapi.testclient import TestClient
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

# Set a dummy API key before importing the app to avoid OpenAIError
os.environ["OPENAI_API_KEY"] = "test_api_key"

from app.main import app

client = TestClient(app)

@pytest.fixture
def mock_openai():
    with patch("app.document_parser.client") as mock_client:
        mock_choice = MagicMock()
        # Correctly format the JSON string content
        mock_choice.message.content = json.dumps([{"source": "System A", "target": "System B", "confidence": 0.9}])
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        yield mock_client

@pytest.fixture
def mock_spacy():
    with patch("app.document_parser.nlp") as mock_nlp:
        mock_ent = MagicMock()
        mock_ent.text = "System A"
        mock_ent.label_ = "ORG"
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc
        yield mock_nlp

@pytest.fixture
def temp_pdf_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "test.pdf"
        import pypdf
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=8.5 * 72, height=11 * 72)
        writer.write(pdf_path)
        yield tmpdir

@patch("app.document_parser.extract_text_from_pdf", return_value="This is a test document about System A and System B.")
def test_process_documents_success(mock_extract_text, temp_pdf_dir, mock_openai, mock_spacy):
    response = client.post("/process-documents/", json={"path": temp_pdf_dir})
    assert response.status_code == 200
    assert response.json() == [{"source": "System A", "target": "System B", "confidence": 0.9}]
    db_path = Path(temp_pdf_dir) / "edge_candidates.db"
    assert db_path.exists()

def test_process_documents_invalid_path():
    response = client.post("/process-documents/", json={"path": "/invalid/path"})
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid directory path provided."}

@patch("app.document_parser.extract_text_from_pdf", return_value="This is a test document about System A and System B.")
def test_process_documents_no_edges_found(mock_extract_text, temp_pdf_dir, mock_openai, mock_spacy):
    mock_choice = MagicMock()
    mock_choice.message.content = '[]'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai.chat.completions.create.return_value = mock_response
    response = client.post("/process-documents/", json={"path": temp_pdf_dir})
    assert response.status_code == 200
    assert response.json() == {"message": "No causal edges found."}
