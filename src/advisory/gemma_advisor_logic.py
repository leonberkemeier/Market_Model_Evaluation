"""
Gemma 3 "Decision Agent" Loop

Orchestrates the advisory decision through a 3-step funnel:

    Step 1 (no LLM):  Pull Top 10 candidates via SQL.
    Step 2 (LLM):     Gemma 3 receives the list and calls get_ticker_details
                       for each candidate via Ollama tool-calling.
    Step 3 (LLM):     Gemma 3 evaluates MCMC math vs News/RAG narrative
                       and produces structured JSON decisions.

Gemma 3 is accessed through Ollama's /api/chat endpoint with native
tool-calling support.  The system prompt defines the "Investment Committee"
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

## Rules
1. NEVER allocate to an asset whose mu_posterior is negative.
2. NEVER exceed the user's Expected-Shortfall threshold for ANY position.
3. Always diversify: final portfolio MUST include ≥5 distinct tickers.
4. Weight each position using win_probability as a quality signal; cap any \
   single position at 25 % of the total allocation.
5. When the regime is "Bear" for a ticker, halve the sizing.
6. Negative news_pulse (< -0.5) is a red flag — downgrade sizing by 50 % \
   or reject entirely if key_risks are severe.
7. When in doubt, prefer HOLD over BUY.

## Workflow
You will be given a table of candidate assets pre-filtered by their \
Expected-Shortfall score.  For each candidate, call `get_ticker_details` \
to retrieve its full profile (MCMC stats, News Pulse, Key Risks).

After reviewing all candidates, output your decisions as a JSON array.

## Output Format (strict)
Respond ONLY with a JSON array.  Each element:
```json
{"ticker": "SYMBOL", "decision": "BUY"|"HOLD"|"SELL", "sizing": 0.00, "reasoning": "one sentence"}
```
- `sizing` is a float 0.0–0.25 representing portfolio fraction.
- Sum of all BUY sizings must be ≤ 1.0.
- Do NOT include any text outside the JSON array.
"""


# ── Ollama tool definitions (matches Ollama /api/chat "tools" schema) ─────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_ticker_details",
            "description": (
                "Get detailed MCMC posterior stats, News Pulse score, and "
                "top 3 Key Risks from SEC filings for a single ticker."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g. AAPL)",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_rebalance",
            "description": (
                "Compare current portfolio holdings vs target weights and "
                "compute a buy/sell order plan for a user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "User ID to rebalance for",
                    }
                },
                "required": ["user_id"],
            },
        },
    },
]

# Map tool names to their Python implementations
_TOOL_DISPATCH = {
    "get_ticker_details": get_ticker_details,
    "execute_rebalance": execute_rebalance,
}


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
    tool_calls_made: int = 0
    raw_llm_response: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "category_id": self.category_id,
            "timestamp": self.timestamp,
            "candidates_count": self.candidates_count,
            "decisions": [d.to_dict() for d in self.decisions],
            "tool_calls_made": self.tool_calls_made,
            "error": self.error,
        }


# ── Ollama chat helpers ───────────────────────────────────────────────────

def _ollama_chat(
    messages: List[Dict[str, Any]],
    tools: Optional[List[dict]] = None,
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
    if tools:
        payload["tools"] = tools

    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"

    with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def _execute_tool_call(tool_call: dict) -> str:
    """
    Dispatch a tool call from Ollama's response to the local implementation.

    Args:
        tool_call: Dict with 'function' -> {'name': ..., 'arguments': {...}}

    Returns:
        Tool result as a string (JSON).
    """
    func_info = tool_call.get("function", {})
    name = func_info.get("name", "")
    args = func_info.get("arguments", {})

    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        result = handler(**args)
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return json.dumps({"error": f"Tool {name} failed: {str(e)}"})


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
    max_tool_rounds: int = 15,
    auto_rebalance: bool = False,
) -> AdvisoryCycleResult:
    """
    Run a full advisory decision cycle for a user.

    Args:
        user_id: User to advise.
        category_id: Risk category (1–5) for candidate filtering.
        max_tool_rounds: Maximum number of tool-calling iterations before
                         forcing a final answer.
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

    # ── Step 2 + 3: Gemma tool-calling loop ───────────────────────────
    # Build the initial user message with the candidate table
    candidate_table = json.dumps(candidates, indent=2)
    user_msg = (
        f"Here are the Top {len(candidates)} candidate assets for risk "
        f"category {category_id}:\n\n{candidate_table}\n\n"
        f"For each candidate, call get_ticker_details to get the full "
        f"profile.  Then evaluate all candidates and provide your "
        f"final allocation decisions as a JSON array."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    # Tool-calling loop
    for round_num in range(max_tool_rounds):
        try:
            response = _ollama_chat(messages, tools=TOOL_DEFINITIONS)
        except Exception as e:
            result.error = f"Ollama chat failed (round {round_num}): {e}"
            logger.error(result.error)
            return result

        msg = response.get("message", {})
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        # Append assistant message to history
        messages.append(msg)

        # If no tool calls, Gemma has produced its final answer
        if not tool_calls:
            result.raw_llm_response = content
            logger.info(
                f"Step 3: Gemma produced final answer after "
                f"{result.tool_calls_made} tool calls"
            )
            break

        # Process tool calls
        for tc in tool_calls:
            result.tool_calls_made += 1
            tool_result = _execute_tool_call(tc)

            # Append tool result as a tool message
            messages.append({
                "role": "tool",
                "content": tool_result,
            })

            func_name = tc.get("function", {}).get("name", "?")
            logger.debug(f"  Tool call #{result.tool_calls_made}: {func_name}")

    else:
        # Exceeded max rounds — force final answer without tools
        logger.warning(f"Max tool rounds ({max_tool_rounds}) reached, forcing final answer")
        messages.append({
            "role": "user",
            "content": (
                "You have used all available tool calls.  Based on the "
                "information gathered so far, provide your final allocation "
                "decisions now as a JSON array."
            ),
        })
        try:
            response = _ollama_chat(messages, tools=None)
            result.raw_llm_response = response.get("message", {}).get("content", "")
        except Exception as e:
            result.error = f"Final answer extraction failed: {e}"
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
