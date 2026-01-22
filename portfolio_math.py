import numpy as np
import pandas as pd
import scipy.optimize as scipy_opt

def calculate_log_returns(prices):
    """
    Calculates natural log returns from price data.
    Formula: ln(Price_t / Price_t-1)
    """
    # We use log returns because they are additive over time and easier to model mathematically
    return np.log(prices / prices.shift(1)).dropna()

def analyze_portfolio(weights, log_returns):
    """
    Calculates the performance of a specific portfolio given its weights.
    
    Returns:
    - Annualized Return
    - Annualized Volatility (Risk)
    - Sharpe Ratio
    """
    weights = np.array(weights)
    
    # 1. Calculate Expected Annual Return
    # mean() gives daily return, * 252 scales it to a trading year
    annual_return = np.sum(log_returns.mean() * weights) * 252
    
    # 2. Calculate Expected Annual Volatility (Standard Deviation)
    # This involves Matrix Algebra: sqrt(weights * covariance_matrix * weights)
    cov_matrix = log_returns.cov() * 252
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    annual_volatility = np.sqrt(portfolio_variance)
    
    # 3. Calculate Sharpe Ratio (Risk-Adjusted Return)
    # We assume a Risk-Free Rate of 2% (0.02) for this calculation
    risk_free_rate = 0.02
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    return annual_return, annual_volatility, sharpe_ratio

def run_simulation(log_returns, num_simulations=5000):
    """
    Runs a Monte Carlo simulation to generate thousands of random portfolios.
    """
    num_assets = len(log_returns.columns)
    
    # Create an array to hold results:
    # Row 0 = Returns
    # Row 1 = Volatility
    # Row 2 = Sharpe Ratio
    results = np.zeros((3, num_simulations))
    
    print(f"Running {num_simulations} simulations...")
    
    for i in range(num_simulations):
        # Generate random weights
        weights = np.random.random(num_assets)
        weights /= np.sum(weights) # Normalize so they sum to 1.0 (100%)
        
        # Calculate metrics for this random portfolio
        ret, vol, sharpe = analyze_portfolio(weights, log_returns)
        
        # Store results
        results[0,i] = ret
        results[1,i] = vol
        results[2,i] = sharpe
        
    print("Simulation complete.")
    return results
def negative_sharpe(weights, log_returns):
    """
    Helper function for Scipy.
    Scipy's 'minimize' function tries to find the lowest number.
    Since we want the HIGHEST Sharpe Ratio, we return the NEGATIVE Sharpe.
    Minimizing a negative number is the same as Maximizing a positive one.
    """
    return -analyze_portfolio(weights, log_returns)[2]

def optimize_portfolio(log_returns):
    """
    Uses the SLSQP method (Sequential Least Squares Programming) to find the
    optimal weights.
    """
    num_assets = len(log_returns.columns)
    
    # 1. Guess: Start by assuming equal weights (e.g., 25% each)
    initial_weights = [1./num_assets for _ in range(num_assets)]
    
    # 2. Constraint: Sum of weights must equal 1 (100%)
    # 'eq' means equality constraint. 'fun' is the function that must equal 0.
    # sum(x) - 1 = 0  ->  sum(x) = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # 3. Bounds: Each weight must be between 0 and 1 (0% to 100%)
    # This prevents "Short Selling" (negative weights).
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # 4. The Solver
    print("Optimizing portfolio weights...")
    result = scipy_opt.minimize(
        negative_sharpe, 
        initial_weights, 
        args=(log_returns,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    return result.x  # This is the array of optimal weights
# --- TEST CODE (Runs only if you execute this file directly) ---
# --- TEST CODE ---
if __name__ == "__main__":
    import data_loader
    
    print("--- Testing Portfolio Math & Optimization ---")
    
    # 1. Get dummy data
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN'] # Added AMZN to make it harder
    prices = data_loader.fetch_data(tickers, '2023-01-01', '2024-01-01')
    log_rets = calculate_log_returns(prices)
    
    # 2. Run Optimization
    optimal_weights = optimize_portfolio(log_rets)
    
    print("\nOPTIMAL WEIGHTS FOUND:")
    for ticker, weight in zip(tickers, optimal_weights):
        print(f"{ticker}: {weight:.2%}")
        
    print("\nTest Passed!")