import data_loader
import portfolio_math
import visualizer
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
TICKERS = ['SPY', 'TLT', 'GLD', 'NVDA', 'XLP'] 
START_DATE = '2020-01-01' # Need enough history for the lookback window
END_DATE = '2025-01-01'

def main():
    print("-----------------------------------")
    print("   ROBO-ADVISOR 2.0 (AI BACKTEST)  ")
    print("-----------------------------------")

    # 1. LOAD DATA
    prices = data_loader.fetch_data(TICKERS, START_DATE, END_DATE)
    if prices is None: return
    
    # 2. CALCULATE RETURNS
    print("\n[+] Calculating Returns...")
    log_returns = portfolio_math.calculate_log_returns(prices)
    
    # 3. STATIC OPTIMIZATION (The "Ideal" Portfolio)
    print("[+] Calculating Static Optimal Weights...")
    optimal_weights = portfolio_math.optimize_portfolio(log_returns)
    
    # 4. WALK-FORWARD BACKTEST (The "Real World" Test)
    # Rebalance every quarter (63 days) using 1 year (252 days) of history
    print("[+] Running Walk-Forward Backtest (This may take 10-20 seconds)...")
    strategy_log_rets = portfolio_math.rolling_backtest(log_returns, window_days=252, rebalance_days=63)
    
    # 5. RISK ANALYSIS (VaR)
    # We calculate VaR on the STRATEGY returns, not the raw assets
    var_95 = portfolio_math.calculate_var(strategy_log_rets)
    
    # 6. RESULTS OUTPUT
    print("\n-----------------------------------")
    print(f"   RESULTS (Strategy vs SPY)")
    print("-----------------------------------")
    
    # Total Return Calculation
    total_strat_return = np.exp(strategy_log_rets.sum()) - 1
    # Benchmark (SPY) Return over same period
    spy_log_rets = log_returns['SPY'].loc[strategy_log_rets.index]
    total_spy_return = np.exp(spy_log_rets.sum()) - 1
    
    print(f"Strategy Total Return : {total_strat_return:.2%}")
    print(f"Benchmark (SPY) Return: {total_spy_return:.2%}")
    print(f"95% Daily VaR         : {var_95:.2%}")
    print(f" (Meaning: 95% of days, loss < {abs(var_95):.2%})")

    # 7. VISUALIZE
    # Chart 1: The Efficient Frontier (Static view)
    sim_results = portfolio_math.run_simulation(log_returns, num_simulations=1000)
    visualizer.plot_efficient_frontier(sim_results, optimal_weights, log_returns, TICKERS)
    
    # Chart 2: The Backtest (Dynamic view)
    visualizer.plot_backtest(strategy_log_rets, spy_log_rets)
    
    # Chart 3: Risk Distribution (VaR)
    visualizer.plot_var_distribution(strategy_log_rets, var_95)
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()