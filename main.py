import data_loader
import portfolio_math
import visualizer

# --- CONFIGURATION (Change these to test different strategies) ---
# Tickers: Standard finance mix (S&P500, Bonds, Gold, Tech, Defensive)
TICKERS = ['META', 'NVDA', 'RKLB', 'GOOGL', 'AAPL', 'MSFT'] 
START_DATE = '2025-01-01'
END_DATE = '2026-01-01'

def main():
    print("-----------------------------------")
    print("   ROBO-ADVISOR PORTFOLIO TOOL     ")
    print("-----------------------------------")

    # 1. LOAD DATA
    prices = data_loader.fetch_data(TICKERS, START_DATE, END_DATE)
    if prices is None: return
    
    # 2. CALCULATE RETURNS
    print("\n[+] Calculating Returns...")
    log_returns = portfolio_math.calculate_log_returns(prices)
    
    # 3. OPTIMIZE (The "Brain")
    print("[+] Optimizing Portfolio Weights...")
    optimal_weights = portfolio_math.optimize_portfolio(log_returns)
    
    # 4. SIMULATE (The "Context")
    print("[+] Running Monte Carlo Simulation...")
    sim_results = portfolio_math.run_simulation(log_returns, num_simulations=2000)
    
    # 5. DISPLAY RESULTS
    print("\n-----------------------------------")
    print("   OPTIMAL ASSET ALLOCATION")
    print("-----------------------------------")
    for ticker, weight in zip(TICKERS, optimal_weights):
        print(f"{ticker.ljust(6)}: {weight:.2%}")
        
    # 6. VISUALIZE
    visualizer.plot_efficient_frontier(sim_results, optimal_weights, log_returns, TICKERS)
    visualizer.plot_weights_pie(optimal_weights, TICKERS)
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()