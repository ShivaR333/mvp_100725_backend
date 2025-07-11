import logging
from pathlib import Path
import sqlite3
from typing import List, Dict, Any
import os
import json

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize text splitter for intelligent chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Initialize OpenAI client (optional for testing)
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts text from a PDF file."""
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    return "\n".join([doc.page_content for doc in documents])


def get_causal_statements_from_text(text_chunk: str) -> List[Dict[str, Any]]:
    """Uses GPT-4o to identify causal relationships directly from text."""
    prompt = f"""
    Analyze the following text and identify causal relationships between variables, processes, or entities.

    For each causal relationship you find, extract:
    - source: The cause variable/entity (what influences something else)
    - target: The effect variable/entity (what is being influenced)  
    - confidence: Your confidence in this relationship (0.0 to 1.0)

    Focus on:
    - Engineering/technical variables (temperature, pressure, flow rate, etc.)
    - Process relationships (increasing X causes Y to decrease)
    - System dependencies (A affects B, B influences C)
    - Quality/performance relationships

    Return a JSON object with a "relationships" array containing objects with "source", "target", and "confidence" fields.

    Text: "{text_chunk}"
    """
    
    if not client or not client.api_key:
        logger.error("OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable.")
        return []
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in causal analysis and engineering systems. Analyze text to identify causal relationships between variables."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content:
            parsed = json.loads(content)
            # Extract relationships array from the response
            return parsed.get("relationships", [])
        return []
    except Exception as e:
        logger.error(f"Error calling OpenAI API or parsing response: {e}")
        return []


def extract_edges_from_docs(doc_dir: Path) -> pd.DataFrame:
    """
    Extracts causal edges from documents in a directory using GPT-4o.

    Algorithm:
    1. Split documents into ~800-character chunks with intelligent separators.
    2. For each chunk, use GPT-4o to directly identify causal relationships.
    3. GPT-4o returns JSON with source, target, and confidence for each relationship.
    4. Aggregate into DataFrame and write to SQLite table edge_candidates.
    """
    all_edges = []

    for doc_path in doc_dir.glob("*.pdf"):
        logger.info(f"Processing document: {doc_path.name}")
        full_text = extract_text_from_pdf(doc_path)
        chunks = text_splitter.split_text(full_text)

        for i, chunk in enumerate(chunks):
            logger.info(f"  Processing chunk {i+1}/{len(chunks)}")
            
            # Skip very short chunks that are unlikely to contain meaningful relationships
            if len(chunk.strip()) < 100:
                continue

            causal_statements = get_causal_statements_from_text(chunk)
            if causal_statements:
                # Add document source to each relationship
                for statement in causal_statements:
                    statement['document'] = doc_path.name
                    statement['chunk_id'] = i
                all_edges.extend(causal_statements)

    if not all_edges:
        logger.warning("No causal edges found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_edges)
    logger.info(f"Extracted {len(df)} causal edges.")

    # Database operations
    db_path = doc_dir / "edge_candidates.db"
    conn = sqlite3.connect(db_path)
    df.to_sql("edge_candidates", conn, if_exists="replace", index=False)
    conn.close()
    logger.info(f"Saved edge candidates to '{db_path}'")

    return df