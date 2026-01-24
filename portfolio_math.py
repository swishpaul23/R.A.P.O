import numpy as np
import pandas as pd
import scipy.optimize as scipy_opt
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

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

def run_simulation(log_returns, num_simulations=10000):
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
def calculate_var(portfolio_returns, confidence_level=0.95):
    """
    Calculates Historical Value at Risk (VaR).
    If confidence is 0.95, it returns the 5th percentile worst loss.
    """
    # 1. Convert log returns to simple percentage returns for easier reading
    simple_returns = np.exp(portfolio_returns) - 1
    
    # 2. Find the cutoff point
    cutoff = (1 - confidence_level) * 100
    var_value = np.percentile(simple_returns, cutoff)
    
    return var_value

def rolling_backtest(log_returns, window_days=252, rebalance_days=63):
    """
    Simulates a 'Walk-Forward' strategy.
    - Look back 'window_days' (e.g., 1 year) to calculate optimal weights.
    - Hold those weights for 'rebalance_days' (e.g., 1 quarter).
    - Repeat.
    """
    num_days = len(log_returns)
    num_assets = len(log_returns.columns)
    
    # Create an empty array to store the strategy's daily returns
    strategy_log_returns = pd.Series(index=log_returns.index, data=0.0)
    
    # Start the loop after the first window (e.g., Day 253)
    # We step forward by 'rebalance_days' chunks
    print(f"Starting Walk-Forward Backtest (Window={window_days}, Rebalance={rebalance_days})...")
    
    for t in range(window_days, num_days, rebalance_days):
        # 1. Slice the "Lookback" data (What we knew THEN)
        start_idx = t - window_days
        end_idx = t
        lookback_data = log_returns.iloc[start_idx:end_idx]
        
        # 2. Optimize weights based ONLY on that past data
        # (This is the slow part - running Scipy inside a loop)
        current_weights = optimize_portfolio(lookback_data)
        
        # 3. Apply weights to the "Future" (next rebalance period)
        # Handle the end of the dataset
        next_rebalance = min(t + rebalance_days, num_days)
        future_returns = log_returns.iloc[t:next_rebalance]
        
        # Calculate returns for this period: Sum(Asset_Return * Weight)
        # Note: In log returns, we can't just multiply, but for daily small steps, 
        # weighted sum is a close enough approximation for this simulation.
        period_ret = (future_returns * current_weights).sum(axis=1)
        strategy_log_returns.iloc[t:next_rebalance] = period_ret
        
    # Drop the first 'window_days' because we were just watching, not trading
    return strategy_log_returns.iloc[window_days:]
# --- HIERARCHICAL RISK PARITY (HRP) LOGIC ---

def get_ivp(cov):
    """Calculates Inverse Variance Portfolio (Risk Parity)"""
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def get_cluster_var(cov, c_items):
    """Calculates variance of a cluster"""
    cov_slice = cov.loc[c_items, c_items]   # <--- Change to .loc
    w = get_ivp(cov_slice).reshape(-1, 1)
    c_var = np.dot(np.dot(w.T, cov_slice), w)[0, 0]
    return c_var

def get_quasi_diag(link):
    """Sorts the cluster to place similar assets next to each other"""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]
    
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= num_items]
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
        
    return sort_ix.tolist()

def get_rec_bisection(cov, sort_ix):
    """Recursive Bisection: The core logic that allocates weights"""
    w = pd.Series(1, index=sort_ix)
    c_items = [sort_ix]
    
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
        
        for i in range(0, len(c_items), 2):
            c0 = c_items[i]
            c1 = c_items[i + 1]
            
            c_var0 = get_cluster_var(cov, c0)
            c_var1 = get_cluster_var(cov, c1)
            
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c0] *= alpha
            w[c1] *= 1 - alpha
            
    return w

def optimize_hrp(log_returns):
    """
    Main HRP Function.
    1. Cluster the assets based on correlation.
    2. Sort them (Quasi-Diagonalization).
    3. Allocate risk recursively.
    """
    # 1. Compute Correlation & Covariance
    corr = log_returns.corr()
    cov = log_returns.cov()
    
    # 2. Clustering (Single Linkage)
    dist = ssd.squareform(((1 - corr) / 2)**0.5)
    link = sch.linkage(dist, 'single')
    
    # 3. Sort (Quasi-Diagonalization)
    sort_ix = get_quasi_diag(link)
    sort_ix = corr.index[sort_ix].tolist()
    
    # 4. Allocation (Recursive Bisection)
    # We re-order the covariance matrix to match the sorted cluster
    cov_sorted = cov.loc[sort_ix, sort_ix]
    hrp_weights = get_rec_bisection(cov_sorted, sort_ix)
    
    # Re-sort weights to match the original ticker order (alphabetical/input order)
    return hrp_weights[log_returns.columns].values, link