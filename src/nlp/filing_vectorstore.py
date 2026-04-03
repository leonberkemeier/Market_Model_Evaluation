"""
Filing Vector Store — LLM-Normalized Embedding Pipeline

Two-stage pipeline for embedding SEC filing sections into ChromaDB:

Stage 1 (deterministic, no LLM needed):
    Raw filing text → regex section extraction → deterministic cleaning
    (strip XBRL, fonts, page headers, boilerplate) → chunking

Stage 2 (requires LLM):
    Raw chunks → LLM normalization (rewrite into clean, structured
    financial prose) → embed via nomic-embed-text → store in ChromaDB

This module handles both stages. Stage 2 is optional — if the LLM is
unavailable, cleaned-but-unnormalized chunks are stored instead.

The resulting 'sec_filings_normalized' collection replaces the noisy
'sec_filings' full-document chunks with clean, section-specific embeddings.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chromadb
import httpx
from loguru import logger

from src.config.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    EMBEDDING_MODEL,
    EMBEDDING_COLLECTION,
    EMBEDDING_CHUNK_SIZE,
    EMBEDDING_CHUNK_OVERLAP,
    RAG_CHROMA_PATH,
)
from src.nlp.filing_analyzer import FilingNLPAnalyzer


# ── Deterministic text cleaning ───────────────────────────────────────────

# Patterns to strip before embedding (XBRL, HTML, formatting junk)
_NOISE_PATTERNS = [
    # XBRL/iXBRL tags and namespaces
    re.compile(r"<[^>]*>", re.DOTALL),
    # US-GAAP taxonomy references
    re.compile(r"(?:us-gaap|dei|srt|aapl|msft|nvda):[A-Za-z0-9]+", re.IGNORECASE),
    # HTTP URLs (FASB, SEC, etc.)
    re.compile(r"https?://\S+"),
    # EDGAR accession numbers and CIK
    re.compile(r"\b\d{10}-\d{2}-\d{6}\b"),
    re.compile(r"\bCIK[:\s]*\d+\b", re.IGNORECASE),
    # Page headers/footers like "Apple Inc. | 2025 Form 10-K | 14"
    re.compile(r"(?:Apple|Microsoft|NVIDIA)\s+Inc\.?\s*\|[^|]*\|\s*\d+", re.IGNORECASE),
    # Standalone page numbers on their own
    re.compile(r"(?<=\n)\s*\d{1,3}\s*(?=\n)"),
    # Font/style declarations that leak from HTML stripping
    re.compile(r"font-(?:family|size|weight|style)[^;]*;", re.IGNORECASE),
    # Table-of-contents page references ("Business 1", "Risk Factors 5")
    re.compile(r"(?<=\w)\s+\d{1,3}\s*$", re.MULTILINE),
]

# Boilerplate phrases to remove
_BOILERPLATE = [
    "This report should be read in conjunction with",
    "incorporated herein by reference",
    "See accompanying notes to",
    "The following table",
    "Table of Contents",
]


def clean_section_text(text: str) -> str:
    """
    Deterministic cleaning of raw filing section text.

    Strips XBRL artifacts, HTML tags, page headers, font declarations,
    and common boilerplate. Collapses whitespace.

    This runs BEFORE any LLM normalization and does not require
    network access.
    """
    if not text:
        return ""

    # Apply noise pattern removal
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub(" ", text)

    # Remove boilerplate lines
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip lines that are just numbers (page numbers)
        if stripped.isdigit():
            continue
        # Skip very short lines (likely artifacts)
        if len(stripped) < 5:
            continue
        # Skip boilerplate
        if any(bp.lower() in stripped.lower() for bp in _BOILERPLATE):
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # Collapse excessive whitespace
    text = re.sub(r"\s{3,}", "  ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────


@dataclass
class FilingChunk:
    """A single chunk ready for embedding."""

    chunk_id: str             # Unique ID: {ticker}_{filing_type}_{date}_{section}_{idx}
    text: str                 # Cleaned (and optionally LLM-normalized) text
    metadata: Dict = field(default_factory=dict)

    # Metadata fields:
    #   ticker, filing_type, filing_date, section, chunk_index,
    #   filing_id, is_llm_normalized


def chunk_text(
    text: str,
    chunk_size: int = EMBEDDING_CHUNK_SIZE,
    overlap: int = EMBEDDING_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks at sentence boundaries.

    Tries to break at sentence ends (period + space/newline) within
    the chunk_size window. Falls back to hard split if no sentence
    boundary found.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to find a sentence boundary near the end
        # Look backwards from `end` for ". " or ".\n"
        search_window = text[start + chunk_size - 300 : end]
        last_period = max(
            search_window.rfind(". "),
            search_window.rfind(".\n"),
        )

        if last_period > 0:
            # Found a sentence boundary
            split_at = start + chunk_size - 300 + last_period + 1
        else:
            # No sentence boundary, hard split at space
            last_space = text[start:end].rfind(" ")
            split_at = start + last_space if last_space > 0 else end

        chunks.append(text[start:split_at].strip())
        start = split_at - overlap  # Overlap for context continuity

    return [c for c in chunks if len(c) > 50]


# ── LLM Normalization Prompt ──────────────────────────────────────────────

NORMALIZE_PROMPT = """\
You are a financial data engineer. Rewrite the following SEC filing excerpt into clean, structured financial prose. Remove all formatting artifacts, legal boilerplate, and non-substantive text. Keep ONLY financially meaningful content: numbers, trends, risks, and management commentary.

Preserve all dollar amounts, percentages, dates, and metric names exactly. Output clean paragraphs, no bullet points.

FILING EXCERPT:
{chunk_text}

CLEANED VERSION (financial content only, no explanation):"""


# ── Vector Store ──────────────────────────────────────────────────────────

class FilingVectorStore:
    """
    Manages the normalized filing embeddings in ChromaDB.

    Workflow:
        1. extract_and_clean() — deterministic section extraction + cleaning
        2. normalize_with_llm() — optional LLM rewrite of each chunk
        3. embed_and_store() — embed via Ollama + store in ChromaDB
        4. query() — semantic search over normalized filings
    """

    def __init__(
        self,
        chroma_path: str = RAG_CHROMA_PATH,
        collection_name: str = EMBEDDING_COLLECTION,
        ollama_url: str = OLLAMA_BASE_URL,
        llm_model: str = OLLAMA_MODEL,
        embed_model: str = EMBEDDING_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.ollama_url = ollama_url.rstrip("/")
        self.llm_model = llm_model
        self.embed_model = embed_model
        self.timeout = timeout
        self.logger = logger.bind(module="filing_vectorstore")

        # Lazy-init ChromaDB client
        self._client = None
        self._collection = None

        # Reuse section extraction from FilingNLPAnalyzer
        self._analyzer = FilingNLPAnalyzer(
            base_url=ollama_url, model=llm_model, timeout=timeout
        )

    # ── ChromaDB setup ────────────────────────────────────────────────

    def _ensure_collection(self) -> chromadb.Collection:
        """Get or create the normalized filings collection."""
        if self._collection is not None:
            return self._collection

        self._client = chromadb.PersistentClient(path=self.chroma_path)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "LLM-normalized SEC filing sections"},
        )
        self.logger.info(
            f"ChromaDB collection '{self.collection_name}': "
            f"{self._collection.count()} existing documents"
        )
        return self._collection

    # ── Stage 1: Extract & Clean (no LLM needed) ─────────────────────

    def extract_and_clean(
        self,
        filing_text: str,
        ticker: str,
        filing_type: str,
        filing_date: str,
        filing_id: int,
    ) -> List[FilingChunk]:
        """
        Extract sections, clean deterministically, and chunk.

        Returns a list of FilingChunk objects ready for optional
        LLM normalization and embedding.
        """
        normalized = self._analyzer._normalize_text(filing_text)
        chunks = []

        sections_to_extract = [
            ("mda", self._analyzer.extract_mda(normalized, filing_type)),
            ("risk_factors", self._analyzer.extract_section(normalized, "risk_factors")),
        ]

        for section_name, section_text in sections_to_extract:
            if not section_text:
                self.logger.debug(
                    f"{ticker} {filing_type}: {section_name} not found"
                )
                continue

            # Deterministic cleaning
            cleaned = clean_section_text(section_text)
            if len(cleaned) < 100:
                self.logger.debug(
                    f"{ticker}: {section_name} too short after cleaning "
                    f"({len(cleaned)} chars)"
                )
                continue

            # Chunk
            text_chunks = chunk_text(cleaned)
            self.logger.info(
                f"{ticker} {filing_type} {section_name}: "
                f"{len(cleaned):,} clean chars → {len(text_chunks)} chunks"
            )

            for idx, chunk_text_str in enumerate(text_chunks):
                chunk_id = (
                    f"{ticker}_{filing_type}_{filing_date}_{section_name}_{idx}"
                )
                chunks.append(FilingChunk(
                    chunk_id=chunk_id,
                    text=chunk_text_str,
                    metadata={
                        "ticker": ticker,
                        "filing_type": filing_type,
                        "filing_date": filing_date,
                        "filing_id": filing_id,
                        "section": section_name,
                        "chunk_index": idx,
                        "is_llm_normalized": False,
                    },
                ))

        return chunks

    # ── Stage 2: LLM Normalization (requires Ollama) ─────────────────

    def normalize_with_llm(self, chunks: List[FilingChunk]) -> List[FilingChunk]:
        """
        Send each chunk through the LLM for normalization.

        Rewrites raw financial text into clean, structured prose —
        stripping any remaining formatting noise that deterministic
        cleaning missed.

        If the LLM is unavailable, returns chunks unchanged
        (is_llm_normalized stays False).
        """
        # Quick connectivity check
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(f"{self.ollama_url}/api/tags")
                resp.raise_for_status()
        except Exception:
            self.logger.warning(
                "LLM server unreachable — skipping normalization, "
                "using deterministic cleaning only"
            )
            return chunks

        normalized = []
        failed = 0

        for chunk in chunks:
            prompt = NORMALIZE_PROMPT.format(chunk_text=chunk.text)
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(
                        f"{self.ollama_url}/api/generate",
                        json={
                            "model": self.llm_model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.1,
                                "num_predict": 2048,
                            },
                        },
                    )
                    resp.raise_for_status()
                    llm_text = resp.json().get("response", "").strip()

                    # Strip any <think> tags (Qwen3 compat)
                    llm_text = re.sub(
                        r"<think>.*?</think>", "", llm_text, flags=re.DOTALL
                    ).strip()

                    if len(llm_text) > 50:
                        chunk.text = llm_text
                        chunk.metadata["is_llm_normalized"] = True
                    else:
                        failed += 1

            except Exception as e:
                self.logger.debug(f"LLM normalization failed for {chunk.chunk_id}: {e}")
                failed += 1

            normalized.append(chunk)

        n_ok = sum(1 for c in normalized if c.metadata.get("is_llm_normalized"))
        self.logger.info(
            f"LLM normalization: {n_ok}/{len(chunks)} chunks normalized, "
            f"{failed} fell back to deterministic cleaning"
        )
        return normalized

    # ── Stage 3: Embed & Store ────────────────────────────────────────

    def _embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Get embeddings from Ollama's embedding endpoint."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.ollama_url}/api/embed",
                    json={"model": self.embed_model, "input": texts},
                )
                resp.raise_for_status()
                return resp.json().get("embeddings")
        except Exception as e:
            self.logger.error(f"Embedding failed: {e}")
            return None

    def embed_and_store(
        self,
        chunks: List[FilingChunk],
        batch_size: int = 32,
    ) -> int:
        """
        Embed chunks and upsert into ChromaDB.

        Returns number of chunks successfully stored.
        """
        if not chunks:
            return 0

        collection = self._ensure_collection()
        stored = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]

            embeddings = self._embed_texts(texts)
            if embeddings is None:
                self.logger.error(
                    f"Embedding batch {i // batch_size} failed, skipping"
                )
                continue

            collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=texts,
                embeddings=embeddings,
                metadatas=[c.metadata for c in batch],
            )
            stored += len(batch)

        self.logger.info(
            f"Stored {stored}/{len(chunks)} chunks in '{self.collection_name}'"
        )
        return stored

    # ── Full pipeline ─────────────────────────────────────────────────

    def process_filing(
        self,
        filing_text: str,
        ticker: str,
        filing_type: str,
        filing_date: str,
        filing_id: int,
        use_llm: bool = True,
    ) -> int:
        """
        Full pipeline: extract → clean → (optionally normalize) → embed → store.

        Args:
            filing_text: Raw filing text.
            ticker: Stock ticker.
            filing_type: '10-K' or '10-Q'.
            filing_date: Filing date string.
            filing_id: DB filing_id for metadata.
            use_llm: If True, attempt LLM normalization. If False or
                      LLM unavailable, use deterministic cleaning only.

        Returns:
            Number of chunks stored.
        """
        # Stage 1: deterministic
        chunks = self.extract_and_clean(
            filing_text, ticker, filing_type, filing_date, filing_id
        )

        if not chunks:
            self.logger.warning(f"{ticker} {filing_type}: no chunks extracted")
            return 0

        # Stage 2: LLM normalization (optional)
        if use_llm:
            chunks = self.normalize_with_llm(chunks)

        # Stage 3: embed & store
        return self.embed_and_store(chunks)

    # ── Query ─────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        ticker: Optional[str] = None,
        section: Optional[str] = None,
        n_results: int = 5,
    ) -> List[Dict]:
        """
        Semantic search over normalized filing embeddings.

        Args:
            query_text: Natural language query.
            ticker: Filter by ticker (optional).
            section: Filter by section — 'mda' or 'risk_factors' (optional).
            n_results: Number of results to return.

        Returns:
            List of dicts with keys: text, metadata, distance.
        """
        collection = self._ensure_collection()

        # Build Chroma where filter
        where = {}
        if ticker:
            where["ticker"] = ticker
        if section:
            where["section"] = section

        # Embed the query
        embeddings = self._embed_texts([query_text])
        if not embeddings:
            self.logger.error("Failed to embed query")
            return []

        results = collection.query(
            query_embeddings=embeddings,
            n_results=n_results,
            where=where if where else None,
        )

        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            })

        return output
