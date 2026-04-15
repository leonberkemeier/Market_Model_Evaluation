import yfinance as yf
import pandas as pd
from loguru import logger
from src.regime.hmm_detector import HMMRegimeDetector
from src.risk.monte_carlo import MonteCarloSimulator

def main():
    logger.info("Downloading recent SPY data to test HMM + Monte Carlo")
    spy = yf.download("SPY", start="2025-01-01", progress=False)['Close']
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
        
    # 1. Detect Regime with 14-day model
    detector = HMMRegimeDetector()
    detector.load_model()
    regime_output = detector.predict_regime(spy)
    current_regime = regime_output['regime']
    
    logger.info(f"Detected Regime: {current_regime.upper()} (Confidence: {regime_output['confidence']:.1%})")
    
    # 2. Run Monte Carlo Simulator
    logger.info(f"Running Monte Carlo Simulator for {current_regime.upper()} market...")
    mc = MonteCarloSimulator(n_simulations=10000)
    
    # Example prediction: Assuming SPY expected return is slightly positive
    predicted_return = 0.005  
    
    mc_result = mc.simulate(
        ticker="SPY",
        predicted_return=predicted_return,
        regime=current_regime
    )
    
    logger.info("Monte Carlo Results for SPY:")
    logger.info(f"  Predicted Return: {predicted_return:.2%}")
    logger.info(f"  Simulated Mean:   {mc_result.mean_return:.2%}")
    logger.info(f"  95% VaR (Risk):   {mc_result.var_95:.2%}")
    logger.info(f"  99% ES (Tail):    {mc_result.cvar_95:.2%}")
    logger.info(f"  Prob of Loss:     {mc_result.prob_loss:.2%}")

if __name__ == "__main__":
    main()
