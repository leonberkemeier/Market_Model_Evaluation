#!/usr/bin/env python3
"""
Train HMM Regime Detection Model

One-time training script for market-wide regime detection.
Trains on SPY (S&P 500) or specified market index.

Usage:
    python scripts/train_hmm_regime.py
    python scripts/train_hmm_regime.py --ticker QQQ --start 2018-01-01
    python scripts/train_hmm_regime.py --validate
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.regime import HMMRegimeDetector

# Try to import data loader
try:
    from src.data.data_loader import DataLoader
    HAS_DATA_LOADER = True
except ImportError:
    HAS_DATA_LOADER = False
    logger.warning("DataLoader not available, will try yfinance")

# Try yfinance as fallback
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


def load_market_data(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Load market index data for training.
    
    Args:
        ticker: Market index ticker (e.g., SPY, QQQ)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Series of closing prices
    """
    logger.info(f"Loading {ticker} data from {start_date} to {end_date}...")
    
    # Try DataLoader first (if available)
    if HAS_DATA_LOADER:
        try:
            loader = DataLoader()
            prices_df = loader.load_prices(
                ticker, 
                start_date=datetime.strptime(start_date, "%Y-%m-%d"),
                end_date=datetime.strptime(end_date, "%Y-%m-%d")
            )
            
            if not prices_df.empty:
                logger.info(f"Loaded {len(prices_df)} days from DataLoader")
                return prices_df['close']
        except Exception as e:
            logger.warning(f"DataLoader failed: {e}, trying yfinance...")
    
    # Fallback to yfinance
    if HAS_YFINANCE:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            logger.info(f"Loaded {len(data)} days from yfinance")
            return data['Close']
            
        except Exception as e:
            logger.error(f"yfinance failed: {e}")
            raise
    
    raise ImportError(
        "No data source available. Install yfinance: pip install yfinance"
    )


def train_hmm(
    ticker: str = "SPY",
    start_date: str = "2020-01-01",
    end_date: str = None,
    n_states: int = 3,
    n_iter: int = 100,
    model_path: str = "models/hmm_regime_model.pkl",
    validate: bool = False
):
    """
    Train HMM regime detection model.
    
    Args:
        ticker: Market index ticker
        start_date: Training start date
        end_date: Training end date (default: today)
        n_states: Number of regimes (default: 3)
        n_iter: Training iterations
        model_path: Where to save model
        validate: Run validation after training
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info("=" * 60)
    logger.info("HMM Regime Detection - Training Script")
    logger.info("=" * 60)
    
    # Load data
    try:
        prices = load_market_data(ticker, start_date, end_date)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    if len(prices) < 100:
        logger.error(f"Insufficient data: {len(prices)} days (need at least 100)")
        sys.exit(1)
    
    logger.info(f"\nTraining data summary:")
    logger.info(f"  Ticker: {ticker}")
    logger.info(f"  Period: {start_date} to {end_date}")
    logger.info(f"  Days: {len(prices)}")
    logger.info(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")
    
    # Train HMM
    logger.info(f"\nTraining HMM with {n_states} states ({n_iter} iterations)...")
    
    detector = HMMRegimeDetector(
        n_states=n_states,
        n_iter=n_iter,
        random_state=42,
        model_path=Path(model_path)
    )
    
    try:
        detector.fit(prices, verbose=True)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)
    
    # Get regime statistics
    logger.info("\n" + "=" * 60)
    logger.info("Regime Statistics")
    logger.info("=" * 60)
    
    stats = detector.get_regime_statistics(prices)
    
    for regime, metrics in stats.items():
        logger.info(f"\n{regime.upper()} Regime:")
        logger.info(f"  Mean Daily Return: {metrics['mean_return']:.4f} ({metrics['mean_return']*252:.1%} annualized)")
        logger.info(f"  Volatility: {metrics['volatility']:.4f} ({metrics['volatility']*np.sqrt(252):.1%} annualized)")
        logger.info(f"  Frequency: {metrics['frequency']:.1%}")
        logger.info(f"  Avg Duration: {metrics['avg_duration_days']:.1f} days")
        logger.info(f"  Max Duration: {metrics['max_duration_days']:.0f} days")
    
    # Current regime
    current_regime = detector.predict_regime(prices, return_probabilities=True)
    logger.info("\n" + "=" * 60)
    logger.info("Current Market Regime (as of training end)")
    logger.info("=" * 60)
    logger.info(f"  Regime: {current_regime['regime'].upper()}")
    logger.info(f"  Confidence: {current_regime['confidence']:.1%}")
    logger.info(f"  Probabilities:")
    for regime, prob in current_regime['probabilities'].items():
        logger.info(f"    {regime.capitalize():10s} {prob:.1%}")
    
    # Validation
    if validate:
        logger.info("\n" + "=" * 60)
        logger.info("Validation")
        logger.info("=" * 60)
        
        # Get regime history
        history = detector.predict_regime_history(prices)
        
        # Calculate average confidence per regime
        for regime in ["bull", "bear", "sideways"]:
            regime_history = history[history["regime"] == regime]
            if len(regime_history) > 0:
                avg_confidence = regime_history["confidence"].mean()
                logger.info(f"  {regime.capitalize()} avg confidence: {avg_confidence:.1%}")
        
        # Check for reasonable regime distribution
        regime_counts = history["regime"].value_counts()
        logger.info(f"\n  Regime distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(history) * 100
            logger.info(f"    {regime.capitalize():10s} {count:5d} days ({pct:.1f}%)")
        
        # Sanity checks
        logger.info("\n  Sanity checks:")
        
        # Check 1: Bear market shouldn't be > 50% of time
        bear_pct = regime_counts.get("bear", 0) / len(history)
        if bear_pct > 0.5:
            logger.warning("    ⚠️  Bear market > 50% of time - unusual")
        else:
            logger.success("    ✓ Bear market frequency looks reasonable")
        
        # Check 2: Bull market should exist
        if "bull" not in regime_counts:
            logger.warning("    ⚠️  No bull market detected - check data")
        else:
            logger.success("    ✓ Bull market detected")
        
        # Check 3: Average confidence should be > 60%
        avg_confidence = history["confidence"].mean()
        if avg_confidence < 0.6:
            logger.warning(f"    ⚠️  Low average confidence ({avg_confidence:.1%}) - model may be uncertain")
        else:
            logger.success(f"    ✓ Average confidence is good ({avg_confidence:.1%})")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"✅ Training complete! Model saved to: {model_path}")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Use in daily pipeline: detector.load_model()")
    logger.info("  2. Predict regime: detector.predict_regime(prices)")
    logger.info("  3. Optional: Retrain monthly/quarterly for adaptation")
    

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train HMM regime detection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on SPY (default)
  python scripts/train_hmm_regime.py
  
  # Train on QQQ (Nasdaq)
  python scripts/train_hmm_regime.py --ticker QQQ
  
  # Custom date range
  python scripts/train_hmm_regime.py --start 2018-01-01 --end 2023-12-31
  
  # With validation
  python scripts/train_hmm_regime.py --validate
        """
    )
    
    parser.add_argument(
        "--ticker",
        type=str,
        default="SPY",
        help="Market index ticker (default: SPY)"
    )
    
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Training start date YYYY-MM-DD (default: 2020-01-01)"
    )
    
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Training end date YYYY-MM-DD (default: today)"
    )
    
    parser.add_argument(
        "--states",
        type=int,
        default=3,
        help="Number of regime states (default: 3)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Training iterations (default: 100)"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/hmm_regime_model.pkl",
        help="Model save path (default: models/hmm_regime_model.pkl)"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after training"
    )
    
    args = parser.parse_args()
    
    # Train
    train_hmm(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        n_states=args.states,
        n_iter=args.iterations,
        model_path=args.model_path,
        validate=args.validate
    )


if __name__ == "__main__":
    main()
