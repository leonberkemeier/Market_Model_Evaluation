"""
Signal Generator for Trading Integration.

Converts model ScoreResults into actionable trade signals.
Key principle: ONLY generate signals for high-conviction opportunities.
No forced trades - it's perfectly valid to return zero signals.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging

from ..data_structures import ScoreResult, ScorerOutput
from ..portfolio.kelly_optimizer import KellyOptimizer


logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Trade signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"  # Explicit hold (have position, keep it)
    NO_ACTION = "NO_ACTION"  # No position, no action needed


class SignalReason(str, Enum):
    """Why a signal was generated (or not)."""
    HIGH_EV_OPPORTUNITY = "high_ev_opportunity"
    SCORE_THRESHOLD_MET = "score_threshold_met"
    STRONG_CONVICTION = "strong_conviction"
    EXIT_DETERIORATING = "exit_deteriorating_position"
    STOP_LOSS = "stop_loss_triggered"
    TAKE_PROFIT = "take_profit_triggered"
    
    # Rejection reasons
    INSUFFICIENT_EV = "insufficient_ev"
    LOW_CONFIDENCE = "low_confidence"
    BELOW_SCORE_THRESHOLD = "below_score_threshold"
    KELLY_TOO_SMALL = "kelly_fraction_too_small"
    MAX_POSITIONS_REACHED = "max_positions_reached"


@dataclass
class TradeSignal:
    """
    A validated trade signal ready for execution.
    Only generated when quality thresholds are met.
    """
    # Required fields (no defaults) must come first
    ticker: str
    signal_type: SignalType
    model_name: str
    date: date
    
    # Position sizing (from Kelly)
    suggested_weight: float  # As fraction of portfolio (0-1)
    
    # Quality metrics that justified this signal
    score: float  # 0-100
    ev: float  # Expected value
    p_win: float  # Probability of win
    confidence: float  # Model confidence
    kelly_fraction: float  # Raw Kelly fraction
    
    # Optional fields (with defaults) must come after
    suggested_quantity: Optional[int] = None  # Shares (calculated with capital)
    reason: SignalReason = SignalReason.HIGH_EV_OPPORTUNITY
    current_price: Optional[float] = None
    generated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API transmission."""
        return {
            "ticker": self.ticker,
            "signal_type": self.signal_type.value,
            "model_name": self.model_name,
            "date": self.date.isoformat(),
            "suggested_weight": self.suggested_weight,
            "suggested_quantity": self.suggested_quantity,
            "score": self.score,
            "ev": self.ev,
            "p_win": self.p_win,
            "confidence": self.confidence,
            "kelly_fraction": self.kelly_fraction,
            "reason": self.reason.value,
            "current_price": self.current_price,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class SignalGeneratorConfig:
    """
    Configuration for signal generation thresholds.
    All thresholds are MINIMUM requirements - signal only generated if ALL are met.
    """
    # === BUY Signal Thresholds ===
    min_score_buy: float = 70.0  # Minimum score (0-100) to consider buying
    min_ev_buy: float = 0.005  # Minimum expected value (0.5% = 0.005)
    min_confidence_buy: float = 0.6  # Minimum model confidence (0-1)
    min_p_win_buy: float = 0.55  # Minimum win probability
    min_kelly_buy: float = 0.01  # Minimum Kelly fraction (1% position)
    
    # === SELL Signal Thresholds ===
    max_score_sell: float = 30.0  # Maximum score to trigger sell
    max_ev_sell: float = -0.002  # Negative EV threshold for exit
    min_confidence_sell: float = 0.5  # Confidence needed to act on sell signal
    
    # === Portfolio Constraints ===
    max_positions: int = 20  # Maximum concurrent positions
    max_single_position: float = 0.15  # Max 15% in single position
    min_position_size: float = 0.02  # Min 2% position (don't bother with tiny positions)
    
    # === Kelly Settings ===
    kelly_fraction: float = 0.25  # Fractional Kelly for safety
    
    # === Quality Gates ===
    require_positive_ev: bool = True  # Hard requirement: EV must be positive for BUY
    require_data_points: int = 20  # Minimum data points supporting the score


class SignalGenerator:
    """
    Generates trade signals from model scores.
    
    Philosophy: Quality over quantity. 
    - It's better to miss a good trade than take a bad one.
    - Only signal when multiple quality gates are passed.
    - Returning zero signals is a valid and often correct outcome.
    """
    
    def __init__(self, config: Optional[SignalGeneratorConfig] = None):
        """
        Initialize signal generator.
        
        Args:
            config: Signal generation configuration. Uses defaults if not provided.
        """
        self.config = config or SignalGeneratorConfig()
        self.kelly = KellyOptimizer(
            kelly_fraction=self.config.kelly_fraction,
            max_position=self.config.max_single_position,
            min_position=self.config.min_position_size
        )
    
    def generate_signals(
        self,
        scorer_output: ScorerOutput,
        current_holdings: Optional[List[str]] = None,
        capital: Optional[float] = None,
        prices: Optional[Dict[str, float]] = None
    ) -> List[TradeSignal]:
        """
        Generate trade signals from scorer output.
        
        Only returns signals for opportunities that pass ALL quality gates.
        May return an empty list if no opportunities meet the criteria.
        
        Args:
            scorer_output: Output from a model scorer (contains all ScoreResults)
            current_holdings: List of tickers currently held (for SELL signals)
            capital: Total portfolio capital (for quantity calculation)
            prices: Current prices dict (ticker -> price)
            
        Returns:
            List of TradeSignal objects. May be empty if no quality opportunities.
        """
        current_holdings = current_holdings or []
        prices = prices or {}
        
        signals: List[TradeSignal] = []
        
        # Separate into potential buys and sells
        buy_candidates = []
        sell_candidates = []
        
        for score in scorer_output.scores:
            # Evaluate each score against thresholds
            buy_assessment = self._assess_buy_opportunity(score)
            sell_assessment = self._assess_sell_opportunity(score, current_holdings)
            
            if buy_assessment[0]:
                buy_candidates.append((score, buy_assessment[1]))
            
            if sell_assessment[0]:
                sell_candidates.append((score, sell_assessment[1]))
        
        # Process SELL signals first (free up capital)
        for score, reason in sell_candidates:
            signal = self._create_sell_signal(
                score=score,
                model_name=scorer_output.model_name,
                reason=reason,
                current_price=prices.get(score.ticker)
            )
            signals.append(signal)
            logger.info(f"SELL signal: {score.ticker} - {reason.value}")
        
        # Process BUY signals (limited by max positions)
        current_position_count = len(current_holdings) - len(sell_candidates)
        available_slots = self.config.max_positions - current_position_count
        
        if available_slots > 0 and buy_candidates:
            # Sort by EV (best opportunities first)
            buy_candidates.sort(key=lambda x: x[0].ev, reverse=True)
            
            # Take top opportunities up to available slots
            for score, reason in buy_candidates[:available_slots]:
                # Skip if already holding
                if score.ticker in current_holdings:
                    continue
                
                signal = self._create_buy_signal(
                    score=score,
                    model_name=scorer_output.model_name,
                    reason=reason,
                    capital=capital,
                    current_price=prices.get(score.ticker)
                )
                signals.append(signal)
                logger.info(f"BUY signal: {score.ticker} - EV: {score.ev:.4f}, Score: {score.score:.1f}")
        
        # Log summary
        if signals:
            logger.info(f"Generated {len(signals)} signals ({len([s for s in signals if s.signal_type == SignalType.BUY])} BUY, {len([s for s in signals if s.signal_type == SignalType.SELL])} SELL)")
        else:
            logger.info("No signals generated - no opportunities met quality thresholds")
        
        return signals
    
    def _assess_buy_opportunity(self, score: ScoreResult) -> Tuple[bool, Optional[SignalReason]]:
        """
        Assess if a score qualifies as a buy opportunity.
        All thresholds must be met.
        
        Returns:
            Tuple of (is_qualified, reason)
        """
        # Gate 1: Positive EV (hard requirement)
        if self.config.require_positive_ev and score.ev <= 0:
            return False, None
        
        # Gate 2: Minimum EV threshold
        if score.ev < self.config.min_ev_buy:
            return False, None
        
        # Gate 3: Score threshold
        if score.score < self.config.min_score_buy:
            return False, None
        
        # Gate 4: Win probability
        if score.p_win < self.config.min_p_win_buy:
            return False, None
        
        # Gate 5: Model confidence
        if score.confidence < self.config.min_confidence_buy:
            return False, None
        
        # Gate 6: Data points (statistical significance)
        if score.data_points < self.config.require_data_points:
            return False, None
        
        # Gate 7: Kelly fraction (position must be worth taking)
        kelly_f = self.kelly.compute_kelly_fraction(
            score.p_win, score.avg_win, score.avg_loss
        )
        if kelly_f < self.config.min_kelly_buy:
            return False, None
        
        # All gates passed - determine primary reason
        if score.ev >= 0.02:  # Very high EV
            return True, SignalReason.HIGH_EV_OPPORTUNITY
        elif score.confidence >= 0.8:  # Very high confidence
            return True, SignalReason.STRONG_CONVICTION
        else:
            return True, SignalReason.SCORE_THRESHOLD_MET
    
    def _assess_sell_opportunity(
        self, 
        score: ScoreResult, 
        current_holdings: List[str]
    ) -> Tuple[bool, Optional[SignalReason]]:
        """
        Assess if a held position should be sold.
        
        Returns:
            Tuple of (should_sell, reason)
        """
        # Only consider positions we actually hold
        if score.ticker not in current_holdings:
            return False, None
        
        # Gate 1: Negative EV (position expected to lose money)
        if score.ev < self.config.max_ev_sell:
            return True, SignalReason.EXIT_DETERIORATING
        
        # Gate 2: Score dropped below threshold
        if score.score < self.config.max_score_sell:
            if score.confidence >= self.config.min_confidence_sell:
                return True, SignalReason.EXIT_DETERIORATING
        
        return False, None
    
    def _create_buy_signal(
        self,
        score: ScoreResult,
        model_name: str,
        reason: SignalReason,
        capital: Optional[float] = None,
        current_price: Optional[float] = None
    ) -> TradeSignal:
        """Create a validated BUY signal."""
        kelly_f = self.kelly.compute_kelly_fraction(
            score.p_win, score.avg_win, score.avg_loss
        )
        
        # Cap at max position size
        weight = min(kelly_f, self.config.max_single_position)
        
        # Calculate quantity if we have capital and price
        quantity = None
        if capital and current_price and current_price > 0:
            dollar_amount = capital * weight
            quantity = int(dollar_amount / current_price)
        
        return TradeSignal(
            ticker=score.ticker,
            signal_type=SignalType.BUY,
            model_name=model_name,
            date=score.date,
            suggested_weight=weight,
            suggested_quantity=quantity,
            score=score.score,
            ev=score.ev,
            p_win=score.p_win,
            confidence=score.confidence,
            kelly_fraction=kelly_f,
            reason=reason,
            current_price=current_price,
            metadata=score.metadata
        )
    
    def _create_sell_signal(
        self,
        score: ScoreResult,
        model_name: str,
        reason: SignalReason,
        current_price: Optional[float] = None
    ) -> TradeSignal:
        """Create a SELL signal for an existing position."""
        kelly_f = self.kelly.compute_kelly_fraction(
            score.p_win, score.avg_win, score.avg_loss
        )
        
        return TradeSignal(
            ticker=score.ticker,
            signal_type=SignalType.SELL,
            model_name=model_name,
            date=score.date,
            suggested_weight=0.0,  # Selling entire position
            suggested_quantity=None,  # Will sell all shares
            score=score.score,
            ev=score.ev,
            p_win=score.p_win,
            confidence=score.confidence,
            kelly_fraction=kelly_f,
            reason=reason,
            current_price=current_price,
            metadata=score.metadata
        )
    
    def get_opportunity_summary(self, scorer_output: ScorerOutput) -> Dict:
        """
        Get a summary of opportunity quality without generating signals.
        Useful for debugging and monitoring.
        """
        scores = scorer_output.scores
        
        if not scores:
            return {"status": "no_scores", "opportunities": 0}
        
        # Analyze distribution
        evs = [s.ev for s in scores]
        positive_ev = [s for s in scores if s.ev > 0]
        high_score = [s for s in scores if s.score >= self.config.min_score_buy]
        qualified = [s for s in scores if self._assess_buy_opportunity(s)[0]]
        
        return {
            "status": "analyzed",
            "total_scores": len(scores),
            "positive_ev_count": len(positive_ev),
            "high_score_count": len(high_score),
            "qualified_opportunities": len(qualified),
            "best_ev": max(evs) if evs else 0,
            "worst_ev": min(evs) if evs else 0,
            "avg_ev": sum(evs) / len(evs) if evs else 0,
            "model_name": scorer_output.model_name,
            "date": scorer_output.date.isoformat(),
            "recommendation": "trade" if qualified else "wait"
        }


# === Convenience Functions ===

def create_conservative_generator() -> SignalGenerator:
    """Create a very conservative signal generator (fewer trades, higher quality)."""
    config = SignalGeneratorConfig(
        min_score_buy=80.0,
        min_ev_buy=0.01,  # 1% minimum EV
        min_confidence_buy=0.7,
        min_p_win_buy=0.60,
        min_kelly_buy=0.02,
        max_positions=10,
        max_single_position=0.10,
        kelly_fraction=0.20,  # More conservative Kelly
    )
    return SignalGenerator(config)


def create_moderate_generator() -> SignalGenerator:
    """Create a moderate signal generator (balanced)."""
    return SignalGenerator()  # Uses defaults


def create_aggressive_generator() -> SignalGenerator:
    """Create a more aggressive signal generator (more trades, accept lower quality)."""
    config = SignalGeneratorConfig(
        min_score_buy=60.0,
        min_ev_buy=0.003,
        min_confidence_buy=0.5,
        min_p_win_buy=0.52,
        min_kelly_buy=0.005,
        max_positions=30,
        max_single_position=0.20,
        kelly_fraction=0.35,
    )
    return SignalGenerator(config)
