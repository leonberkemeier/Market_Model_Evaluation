"""
Gemma 3 "Decision Agent" Loop

Orchestrates the advisory decision through a 3-step funnel:

    Step 1 (no LLM):  Pull Top 10 candidates via SQL.
    Step 2 (no LLM):  Pre-fetch detailed profiles for all candidates
                       (MCMC stats, News Pulse, Key Risks from ChromaDB).
    Step 3 (LLM):     Gemma 3 receives one enriched prompt with all data
                       and produces structured JSON allocation decisions.

This "pre-fetch" design keeps Gemma focused on evaluation only, avoids
native tool-calling requirements, and is more efficient for Gemma 3:12B's
context window.  The system prompt defines the "Investment Committee"
persona — a Rational Risk Manager — and forces all final output into
valid JSON.

Usage:
    from src.advisory.gemma_advisor_logic import run_advisory_cycle

    decisions = run_advisory_cycle(user_id=1, category_id=3)
    # [{"ticker": "AAPL", "decision": "BUY", "sizing": 0.15, "reasoning": "..."}]
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from src.config.config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT

# Import tool implementations directly (no MCP transport overhead)
from src.advisory.mcp_server import get_candidates, get_ticker_details, execute_rebalance


# ── System Prompt ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are the Chief Risk Officer on an Investment Committee for a quantitative \
robo-advisory platform.  Your role is **Rational Risk Manager**: you make \
allocation decisions grounded in Bayesian posterior statistics, regime \
detection, and news sentiment — never on hype or speculation.

## Key Metric Definitions
- **mu_posterior**: Expected return.  Positive = bullish, negative = bearish.
- **sigma_posterior**: Uncertainty around mu.  Lower = more confident.
- **es5 (Expected Shortfall 5%)**: The average loss in the worst 5% of \
  scenarios.  ES is ALWAYS negative (it measures downside).  A value \
  closer to 0 (e.g. -0.03) means LESS tail risk; a more negative value \
  (e.g. -0.15) means MORE tail risk.  The user's risk category sets the \
  maximum acceptable ES — all candidates have already passed this filter.
- **win_probability**: P(positive return).  Higher = better odds.
- **news_pulse**: Time-decay-weighted sentiment.  0 = neutral/no data, \
  positive = bullish headlines, negative = bearish headlines.
- **regime**: Bull / Sideways / Bear from HMM regime detection.

## Rules
1. NEVER allocate to an asset whose mu_posterior is negative.
2. The ES threshold has already been enforced by the pre-filter — do NOT \
   reject assets simply because es5 is negative (it always is).  Instead, \
   prefer assets with es5 closer to zero (less tail risk).
3. Always diversify: final portfolio MUST include ≥5 distinct BUY tickers.
4. Weight each position using win_probability as a quality signal; cap any \
   single position at 25 % of the total allocation.
5. When the regime is "Bear" for a ticker, halve the sizing.
6. Negative news_pulse (< -0.5) is a red flag — downgrade sizing by 50 % \
   or reject entirely if key_risks are severe.
7. When in doubt, prefer HOLD over BUY.  But a candidate with positive \
   mu, high win_prob, and Bull regime is a strong BUY signal.

## Workflow
You will be given detailed profiles for candidate assets pre-filtered by \
their Expected-Shortfall score.  Each profile includes MCMC posterior \
stats, News Pulse score, and Key Risks from SEC filings.

Review all candidates and output your decisions as a JSON array.

## Output Format (strict)
Respond ONLY with a JSON array.  Each element:
```json
{"ticker": "SYMBOL", "decision": "BUY"|"HOLD"|"SELL", "sizing": 0.00, "reasoning": "one sentence"}
```
- `sizing` is a float 0.0–0.25 representing portfolio fraction.
- Sum of all BUY sizings must be ≤ 1.0.
- Do NOT include any text outside the JSON array.
"""




# ── Data classes ──────────────────────────────────────────────────────────

@dataclass
class AdvisoryDecision:
    """Single asset decision from the Gemma advisor."""
    ticker: str
    decision: str  # BUY, HOLD, SELL
    sizing: float
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "decision": self.decision,
            "sizing": self.sizing,
            "reasoning": self.reasoning,
        }


@dataclass
class AdvisoryCycleResult:
    """Full result from one advisory cycle."""
    user_id: int
    category_id: int
    timestamp: str
    candidates_count: int
    decisions: List[AdvisoryDecision] = field(default_factory=list)
    details_fetched: int = 0
    raw_llm_response: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "category_id": self.category_id,
            "timestamp": self.timestamp,
            "candidates_count": self.candidates_count,
            "decisions": [d.to_dict() for d in self.decisions],
            "details_fetched": self.details_fetched,
            "error": self.error,
        }


# ── Ollama chat helpers ───────────────────────────────────────────────────

def _ollama_chat(
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
) -> dict:
    """
    Call Ollama /api/chat and return the full response dict.

    Uses low temperature for deterministic, risk-averse output.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 4096,
        },
    }

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"

    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def _prefetch_details(candidates: List[dict]) -> List[dict]:
    """
    Pre-fetch detailed profiles for all candidates (no LLM needed).

    Calls get_ticker_details for each candidate and returns enriched
    dicts.  Failed lookups are skipped silently.
    """
    enriched = []
    for c in candidates:
        ticker = c.get("ticker", "")
        try:
            detail_json = get_ticker_details(ticker)
            detail = json.loads(detail_json)
            if "error" not in detail:
                enriched.append(detail)
            else:
                # Fall back to the candidate summary
                enriched.append(c)
        except Exception as e:
            logger.warning(f"Detail fetch failed for {ticker}: {e}")
            enriched.append(c)
    return enriched


# ── JSON extraction ───────────────────────────────────────────────────────

def _parse_decisions(text: str) -> List[AdvisoryDecision]:
    """
    Extract decision JSON from Gemma's response.

    Tries json.loads first, falls back to regex extraction.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Try direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return _validate_decisions(data)
        if isinstance(data, dict) and "ticker" in data:
            return _validate_decisions([data])
    except json.JSONDecodeError:
        pass

    # Fallback: extract JSON array with regex
    match = re.search(r"\[[\s\S]*?\]", text)
    if match:
        try:
            data = json.loads(match.group())
            return _validate_decisions(data)
        except json.JSONDecodeError:
            pass

    # Last resort: extract individual JSON objects
    objects = re.findall(r"\{[^{}]+\}", text)
    decisions = []
    for obj_str in objects:
        try:
            obj = json.loads(obj_str)
            if "ticker" in obj and "decision" in obj:
                decisions.append(obj)
        except json.JSONDecodeError:
            continue

    return _validate_decisions(decisions) if decisions else []


def _validate_decisions(raw: List[dict]) -> List[AdvisoryDecision]:
    """Validate and normalize decision dicts into AdvisoryDecision objects."""
    decisions = []
    for item in raw:
        ticker = str(item.get("ticker", "")).upper().strip()
        decision = str(item.get("decision", "HOLD")).upper().strip()
        sizing = float(item.get("sizing", 0.0))
        reasoning = str(item.get("reasoning", ""))

        if not ticker:
            continue
        if decision not in ("BUY", "HOLD", "SELL"):
            decision = "HOLD"
        sizing = max(0.0, min(0.25, sizing))

        decisions.append(AdvisoryDecision(
            ticker=ticker,
            decision=decision,
            sizing=sizing,
            reasoning=reasoning[:200],  # cap reasoning length
        ))
    return decisions


# ── Main orchestration ────────────────────────────────────────────────────

def run_advisory_cycle(
    user_id: int,
    category_id: int,
    auto_rebalance: bool = False,
) -> AdvisoryCycleResult:
    """
    Run a full advisory decision cycle for a user.

    Pre-fetch approach (Gemma-compatible, no native tool-calling needed):
        Step 1: Pull Top 10 candidates via SQL (no LLM).
        Step 2: Pre-fetch detailed profiles for all candidates (no LLM).
        Step 3: Send one enriched prompt to Gemma → JSON decisions.

    Args:
        user_id: User to advise.
        category_id: Risk category (1–5) for candidate filtering.
        auto_rebalance: If True, automatically call execute_rebalance after
                        decisions are made.

    Returns:
        AdvisoryCycleResult with decisions and metadata.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    result = AdvisoryCycleResult(
        user_id=user_id,
        category_id=category_id,
        timestamp=timestamp,
        candidates_count=0,
    )

    # ── Step 1: Pull candidates (no LLM) ──────────────────────────────
    logger.info(f"Advisory cycle: user={user_id}, category={category_id}")

    try:
        candidates_json = get_candidates(category_id)
        candidates = json.loads(candidates_json)
    except Exception as e:
        result.error = f"Failed to fetch candidates: {e}"
        logger.error(result.error)
        return result

    if isinstance(candidates, dict) and "error" in candidates:
        result.error = candidates["error"]
        return result

    result.candidates_count = len(candidates)

    if not candidates:
        result.error = "No candidates found for this risk category"
        logger.warning(result.error)
        return result

    logger.info(f"Step 1: {len(candidates)} candidates retrieved")

    # ── Step 2: Pre-fetch all details (no LLM) ────────────────────────
    enriched = _prefetch_details(candidates)
    result.details_fetched = len(enriched)
    logger.info(f"Step 2: {len(enriched)} detailed profiles fetched")

    # ── Step 3: Single Gemma call with enriched data ──────────────────
    profiles_text = json.dumps(enriched, indent=2)
    user_msg = (
        f"Risk category: {category_id} (user {user_id})\n\n"
        f"Below are the detailed profiles for {len(enriched)} candidate "
        f"assets, including MCMC posterior stats (mu, sigma), Expected "
        f"Shortfall (es5), win probability, HMM regime, News Pulse "
        f"score, and Key Risks from SEC filings where available.\n\n"
        f"{profiles_text}\n\n"
        f"Evaluate each candidate. Apply the Investment Committee rules. "
        f"Output your final allocation decisions as a JSON array."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        response = _ollama_chat(messages)
        result.raw_llm_response = response.get("message", {}).get("content", "")
        logger.info("Step 3: Gemma produced allocation decisions")
    except Exception as e:
        result.error = f"Ollama chat failed: {e}"
        logger.error(result.error)
        return result

    # ── Parse decisions ────────────────────────────────────────────────
    if result.raw_llm_response:
        result.decisions = _parse_decisions(result.raw_llm_response)

    if not result.decisions:
        result.error = "Failed to parse any valid decisions from LLM response"
        logger.warning(result.error)
    else:
        buy_decisions = [d for d in result.decisions if d.decision == "BUY"]
        total_sizing = sum(d.sizing for d in buy_decisions)
        logger.info(
            f"Decisions: {len(result.decisions)} total, "
            f"{len(buy_decisions)} BUY (total sizing={total_sizing:.2%})"
        )

    # ── Optional: auto-rebalance ──────────────────────────────────────
    if auto_rebalance and result.decisions:
        try:
            rebalance_result = execute_rebalance(user_id)
            logger.info(f"Auto-rebalance result: {rebalance_result[:200]}")
        except Exception as e:
            logger.error(f"Auto-rebalance failed: {e}")

    return result


# ── CLI entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Gemma 3 Advisory Cycle")
    parser.add_argument("--user-id", type=int, default=1, help="User ID")
    parser.add_argument("--category", type=int, default=3, help="Risk category (1-5)")
    parser.add_argument("--auto-rebalance", action="store_true", help="Auto-execute rebalance")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logger.enable("")

    result = run_advisory_cycle(
        user_id=args.user_id,
        category_id=args.category,
        auto_rebalance=args.auto_rebalance,
    )

    print(json.dumps(result.to_dict(), indent=2))
