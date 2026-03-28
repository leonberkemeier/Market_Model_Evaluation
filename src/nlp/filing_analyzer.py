"""
SEC Filing NLP Analyzer

Extracts Item 1A (Risk Factors) and Item 7/2 (MD&A) from SEC filings,
then sends the extracted sections to a remote Ollama LLM server to produce
structured features: management sentiment, risk count, guidance tone,
and revenue/operating income figures.

Pipeline position: runs offline (script) → writes to fact_filing_analysis →
    BayesianEvaluator reads at backtest time via get_latest_nlp_features().
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import httpx
from loguru import logger

from src.config.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT


# ── Section-extraction regex (mirrored from aggregator's FilingAnalyzer) ──

SECTION_PATTERNS = {
    "risk_factors": r"item\s*1a\s*[\.:\-\u2013\u2014]*\s*risk\s*factors",
    "mda": r"item\s*7\s*[\.:\-\u2013\u2014]*\s*management.*?discussion.*?analysis",
    "mda_quarterly": r"item\s*2\s*[\.:\-\u2013\u2014]*\s*management.*?discussion.*?analysis",
}

# Boundary: next "Item <number>." heading marks end of current section.
# Requires period + capital letter after item number to avoid matching
# inline cross-references like "see Item 7, 'Management's...'"
_NEXT_ITEM_RE = re.compile(
    r"\b[Ii]tem\s+\d+[A-Za-z]?\s*\.\s+[A-Z]"
)

MAX_SECTION_CHARS = 48_000  # ~12k tokens for llama3.1:8b context


# ── Prompt templates ──────────────────────────────────────────────────────

MDA_SENTIMENT_PROMPT = """\
You are a financial analyst. Read the following MD&A (Management's Discussion and Analysis) section from a SEC filing and produce a JSON object with these fields:

1. "mgmt_sentiment": a float from -1.0 (very negative tone) to 1.0 (very positive tone) reflecting management's overall tone about the company's performance and outlook.
2. "revenue_millions": the most recent total revenue/net sales figure mentioned, in millions of USD. Use null if not found.
3. "operating_income_millions": the most recent operating income/EBIT figure mentioned, in millions of USD. Use null if not found.
4. "guidance_tone": a float from -1.0 to 1.0 reflecting the tone of any forward-looking statements or guidance. Use 0.0 if no forward-looking statements found.

Respond ONLY with valid JSON, no explanation.

MD&A TEXT:
{section_text}
"""

RISK_ANALYSIS_PROMPT = """\
You are a financial analyst. Read the following Risk Factors section (Item 1A) from a SEC filing and produce a JSON object with these fields:

1. "risk_count": the number of distinct, material risk categories identified.
2. "risk_summary": a JSON array of short strings (max 10 words each) naming each distinct risk category. Maximum 20 categories.

Respond ONLY with valid JSON, no explanation.

RISK FACTORS TEXT:
{section_text}
"""


@dataclass
class FilingAnalysisResult:
    """Structured output from LLM analysis of a single filing."""

    mgmt_sentiment: Optional[float] = None
    risk_count: Optional[int] = None
    risk_summary: Optional[List[str]] = None
    revenue_extracted: Optional[float] = None
    operating_income_extracted: Optional[float] = None
    guidance_tone: Optional[float] = None
    llm_model: str = ""
    sections_found: List[str] = None

    def __post_init__(self):
        if self.sections_found is None:
            self.sections_found = []

    def to_dict(self) -> Dict:
        return {
            "mgmt_sentiment": self.mgmt_sentiment,
            "risk_count": self.risk_count,
            "risk_summary": json.dumps(self.risk_summary) if self.risk_summary else None,
            "revenue_extracted": self.revenue_extracted,
            "operating_income_extracted": self.operating_income_extracted,
            "guidance_tone": self.guidance_tone,
            "llm_model": self.llm_model,
        }


class FilingNLPAnalyzer:
    """
    Extracts targeted sections from SEC filing text and scores them
    using a remote Ollama LLM server.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.logger = logger.bind(module="filing_nlp")

    # ── Ollama HTTP helpers ───────────────────────────────────────────

    def test_connectivity(self) -> bool:
        """Quick health-check + model availability test."""
        try:
            with httpx.Client(timeout=15) as client:
                # Check server is up
                resp = client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                models = [m["name"] for m in resp.json().get("models", [])]
                self.logger.info(f"Ollama server reachable. Models: {models}")

                if not any(self.model in m for m in models):
                    self.logger.error(
                        f"Model '{self.model}' not found on server. "
                        f"Available: {models}"
                    )
                    return False

                # Quick generation test
                resp = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": "Respond with exactly: OK",
                        "stream": False,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                self.logger.info("Ollama connectivity test passed")
                return True

        except Exception as e:
            self.logger.error(f"Ollama connectivity test failed: {e}")
            return False

    def _generate(self, prompt: str) -> Optional[str]:
        """Send a prompt to Ollama and return the response text."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temp for deterministic extraction
                            "num_predict": 1024,
                        },
                    },
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
        except httpx.TimeoutException:
            self.logger.error(
                f"Ollama request timed out after {self.timeout}s"
            )
            return None
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            return None

    # ── Section extraction ────────────────────────────────────────────

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize filing text for robust regex matching."""
        if not text:
            return ""
        text = text.replace("\xa0", " ")
        text = text.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
        text = re.sub(r"[\s\u00A0]+", " ", text)
        return text

    def extract_section(self, text: str, pattern_key: str) -> Optional[str]:
        """
        Extract a section from filing text using regex.

        Iterates through all regex matches and returns the first one
        that yields substantial content (>200 chars). This skips
        Table-of-Contents entries which match the heading pattern but
        contain only page numbers between section boundaries.

        Returns the section text (truncated to MAX_SECTION_CHARS) or None.
        """
        pattern = SECTION_PATTERNS.get(pattern_key)
        if not pattern:
            return None

        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = match.end()
            next_item = _NEXT_ITEM_RE.search(text[start:])
            end = start + next_item.start() if next_item else len(text)

            section = text[start:end].strip()

            # Skip ToC entries and cross-references (short matches)
            if len(section) < 200:
                continue

            # Truncate to stay within context limits
            if len(section) > MAX_SECTION_CHARS:
                section = section[:MAX_SECTION_CHARS]
                self.logger.debug(
                    f"Truncated {pattern_key} to {MAX_SECTION_CHARS} chars"
                )

            return section

        return None

    def extract_mda(self, text: str, filing_type: str) -> Optional[str]:
        """Extract MD&A section — Item 7 for 10-K, Item 2 for 10-Q."""
        if filing_type == "10-K":
            return self.extract_section(text, "mda")
        else:
            # 10-Q uses Item 2
            return self.extract_section(text, "mda_quarterly")

    # ── JSON response parsing ─────────────────────────────────────────

    @staticmethod
    def _parse_json_response(raw: str) -> Optional[Dict]:
        """Parse JSON from LLM response, with fallback extraction."""
        if not raw:
            return None

        # Try direct parse
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting JSON block from markdown fences
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding first {...} block
        brace_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _clamp(value: Optional[float], lo: float = -1.0, hi: float = 1.0) -> Optional[float]:
        if value is None:
            return None
        try:
            return max(lo, min(hi, float(value)))
        except (TypeError, ValueError):
            return None

    # ── High-level analysis ───────────────────────────────────────────

    def analyze_filing(
        self,
        filing_text: str,
        ticker: str,
        filing_type: str,
        filing_date: str,
    ) -> FilingAnalysisResult:
        """
        Run full NLP analysis on a single SEC filing.

        1. Extract Item 1A (Risk Factors) and Item 7/2 (MD&A)
        2. Send each to Ollama with structured prompts
        3. Parse and return structured results
        """
        result = FilingAnalysisResult(llm_model=self.model)
        normalized = self._normalize_text(filing_text)

        self.logger.info(
            f"Analyzing {filing_type} for {ticker} ({filing_date}), "
            f"{len(filing_text):,} chars"
        )

        # ── MD&A analysis ──
        mda_text = self.extract_mda(normalized, filing_type)
        if mda_text:
            result.sections_found.append("mda")
            self.logger.info(
                f"  MD&A extracted: {len(mda_text):,} chars"
            )

            prompt = MDA_SENTIMENT_PROMPT.format(section_text=mda_text)
            raw_response = self._generate(prompt)
            parsed = self._parse_json_response(raw_response)

            if parsed:
                result.mgmt_sentiment = self._clamp(parsed.get("mgmt_sentiment"))
                result.guidance_tone = self._clamp(parsed.get("guidance_tone"))
                result.revenue_extracted = parsed.get("revenue_millions")
                result.operating_income_extracted = parsed.get(
                    "operating_income_millions"
                )
                self.logger.info(
                    f"  MD&A scored: sentiment={result.mgmt_sentiment}, "
                    f"guidance={result.guidance_tone}, "
                    f"revenue={result.revenue_extracted}M"
                )
            else:
                self.logger.warning(
                    f"  MD&A: failed to parse LLM response for {ticker}"
                )
        else:
            self.logger.warning(f"  MD&A section not found in {filing_type}")

        # ── Risk Factors analysis ──
        risk_text = self.extract_section(normalized, "risk_factors")
        if risk_text:
            result.sections_found.append("risk_factors")
            self.logger.info(
                f"  Risk Factors extracted: {len(risk_text):,} chars"
            )

            prompt = RISK_ANALYSIS_PROMPT.format(section_text=risk_text)
            raw_response = self._generate(prompt)
            parsed = self._parse_json_response(raw_response)

            if parsed:
                result.risk_count = parsed.get("risk_count")
                risk_summary = parsed.get("risk_summary", [])
                if isinstance(risk_summary, list):
                    result.risk_summary = risk_summary[:20]
                self.logger.info(
                    f"  Risks scored: count={result.risk_count}, "
                    f"categories={len(result.risk_summary or [])}"
                )
            else:
                self.logger.warning(
                    f"  Risk Factors: failed to parse LLM response for {ticker}"
                )
        else:
            self.logger.warning(
                f"  Risk Factors section not found in {filing_type}"
            )

        return result
