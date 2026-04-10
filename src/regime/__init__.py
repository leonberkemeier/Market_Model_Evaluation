"""Regime detection module (Layer II)."""

from .hmm_detector import HMMRegimeDetector
from .markov_chain_detector import MarkovChainRegimeDetector

__all__ = [
    "HMMRegimeDetector",              # ⚠️ LEGACY - use MarkovChainRegimeDetector for new code
    "MarkovChainRegimeDetector"       # ✅ NEW - Primary regime detector for all new code
]
