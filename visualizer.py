import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_efficient_frontier(sim_results, optimal_weights, log_returns, tickers):
    """
    Plots the Monte Carlo simulation results and highlights the Optimal Portfolio.
    
    Inputs:
    - sim_results: Numpy array (3 rows: Returns, Volatility, Sharpe)
    - optimal_weights: The calculated best weights
    - log_returns: The history of returns (needed to calc optimal point metrics)
    """
    import portfolio_math # Lazy import to avoid circular dependency issues
    
    # 1. Calculate metrics for the Optimal Portfolio (The Red Star)
    opt_ret, opt_vol, opt_sharpe = portfolio_math.analyze_portfolio(optimal_weights, log_returns)
    
    # 2. Setup the Plot
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-darkgrid') # Professional financial look
    
    # 3. Plot the Cloud (Simulations)
    # X-axis = Volatility (Risk), Y-axis = Return
    # c (Color) = Sharpe Ratio (Darker = Better)
    plt.scatter(sim_results[1,:], sim_results[0,:], c=sim_results[2,:], 
                cmap='viridis', marker='o', s=10, alpha=0.5, label='Simulated Portfolios')
    
    plt.colorbar(label='Sharpe Ratio')
    
    # 4. Plot the Star (Optimal)
    plt.scatter(opt_vol, opt_ret, marker='*', color='r', s=250, 
                edgecolors='black', label=f'Max Sharpe Ratio ({opt_sharpe:.2f})')
    
    # 5. Labels
    plt.title('Efficient Frontier: Risk vs. Return Analysis', fontsize=16)
    plt.xlabel('Annualized Volatility (Risk)', fontsize=12)
    plt.ylabel('Expected Annual Return', fontsize=12)
    plt.legend(loc='upper left')
    
    print("Displaying Efficient Frontier...")
    plt.show()

def plot_weights_pie(weights, tickers):
    """
    Plots a pie chart of the optimal asset allocation.
    """
    # Filter out tiny weights (e.g., 0.0001%) to keep chart clean
    clean_weights = []
    clean_labels = []
    
    for w, label in zip(weights, tickers):
        if w > 0.01: # Only show assets with > 1% allocation
            clean_weights.append(w)
            clean_labels.append(label)
    
    plt.figure(figsize=(8, 8))
    plt.pie(clean_weights, labels=clean_labels, autopct='%1.1f%%', startangle=140, 
            colors=plt.cm.Paired.colors)
    
    plt.title('Optimal Portfolio Allocation')
    print("Displaying Allocation Pie Chart...")
    plt.show()

# --- TEST CODE ---
if __name__ == "__main__":
    import data_loader
    import portfolio_math
    
    print("--- Testing Visualizer ---")
    
    # 1. Get Data & Run Math
    tickers = ['SPY', 'TLT', 'GLD', 'NVDA']
    prices = data_loader.fetch_data(tickers, '2022-01-01', '2024-01-01')
    log_rets = portfolio_math.calculate_log_returns(prices)
    sim_results = portfolio_math.run_simulation(log_rets, 1000)
    opt_weights = portfolio_math.optimize_portfolio(log_rets)
    
    # 2. Test Plots
    plot_efficient_frontier(sim_results, opt_weights, log_rets, tickers)
    plot_weights_pie(opt_weights, tickers)