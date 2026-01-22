import matplotlib.pyplot as plt
import numpy as np
import portfolio_math # Lazy import

def plot_efficient_frontier(sim_results, optimal_weights, log_returns, tickers):
    """
    Plots the Efficient Frontier and the Capital Market Line (CML).
    Returns the matplotlib Figure object for UI rendering.
    """
    # 1. Calculate metrics for the Optimal Portfolio
    opt_ret, opt_vol, opt_sharpe = portfolio_math.analyze_portfolio(optimal_weights, log_returns)
    risk_free_rate = 0.02
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use('seaborn-v0_8-darkgrid')

    # 3. Plot the Cloud
    sc = ax.scatter(sim_results[1,:], sim_results[0,:], c=sim_results[2,:], 
                    cmap='viridis', marker='o', s=10, alpha=0.5)
    plt.colorbar(sc, label='Sharpe Ratio', ax=ax)

    # 4. Plot the Optimal Portfolio (Red Star)
    ax.scatter(opt_vol, opt_ret, marker='*', color='r', s=250, edgecolors='black', 
               label=f'Tangency Portfolio (Sharpe: {opt_sharpe:.2f})', zorder=5)

    # 5. Add the Capital Market Line (CML)
    # The line starts at (0, Risk_Free_Rate) and goes through the Optimal Portfolio
    # Equation: y = mx + c  ->  Return = Sharpe * Volatility + Risk_Free
    x_cml = np.linspace(0, opt_vol * 1.5, 100) # Draw line from 0 to 1.5x the risk
    y_cml = risk_free_rate + opt_sharpe * x_cml
    
    ax.plot(x_cml, y_cml, color='green', linestyle='--', linewidth=2, label='Capital Market Line')

    # Formatting
    ax.set_title('Efficient Frontier & Capital Market Line')
    ax.set_xlabel('Annualized Volatility (Risk)')
    ax.set_ylabel('Annualized Return')
    ax.set_xlim(left=0) # Start x-axis at 0 to show the Risk-Free connection
    ax.legend()

    return fig  # Return the figure so the UI can draw it

def plot_weights_pie(weights, tickers):
    """
    Returns a pie chart figure.
    """
    clean_weights = []
    clean_labels = []
    
    for w, label in zip(weights, tickers):
        if w > 0.01:
            clean_weights.append(w)
            clean_labels.append(label)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(clean_weights, labels=clean_labels, autopct='%1.1f%%', startangle=140)
    ax.set_title('Optimal Portfolio Allocation')
    
    return fig