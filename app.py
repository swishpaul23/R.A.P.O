import streamlit as st
import pandas as pd
import data_loader
import portfolio_math
import visualizer

# --- PAGE CONFIG ---
st.set_page_config(page_title="Robo-Advisor", layout="wide")

# --- SIDEBAR (User Inputs) ---
st.sidebar.header("🔧 Portfolio Settings")

# 1. Ticker Input
default_tickers = "SPY, TLT, GLD, NVDA, GOOGL"
ticker_input = st.sidebar.text_input("Enter Tickers (comma separated):", value=default_tickers)
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# 2. Date Input
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))

# 3. Run Button
if st.sidebar.button("🚀 Optimize Portfolio"):
    
    # --- MAIN DASHBOARD ---
    st.title("📈 Portfolio Optimizer")
    
    with st.spinner("Downloading Financial Data..."):
        prices = data_loader.fetch_data(tickers, str(start_date), str(end_date))
        
    if prices is None:
        st.error("Failed to fetch data. Please check ticker symbols.")
    else:
        # Run Math
        log_returns = portfolio_math.calculate_log_returns(prices)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Mathematical Optimization")
            with st.spinner("Calculating Max Sharpe Ratio..."):
                optimal_weights = portfolio_math.optimize_portfolio(log_returns)
                
                # Display Weights as a nice dataframe
                df_weights = pd.DataFrame({
                    "Ticker": tickers,
                    "Weight": [f"{w:.2%}" for w in optimal_weights]
                })
                st.dataframe(df_weights, hide_index=True)

        with col2:
            st.subheader("2. Asset Allocation")
            fig_pie = visualizer.plot_weights_pie(optimal_weights, tickers)
            st.pyplot(fig_pie)

        st.subheader("3. Efficient Frontier & Capital Market Line")
        with st.spinner("Running Monte Carlo Simulation (This may take a moment)..."):
            sim_results = portfolio_math.run_simulation(log_returns, num_simulations=2000)
            fig_frontier = visualizer.plot_efficient_frontier(sim_results, optimal_weights, log_returns, tickers)
            st.pyplot(fig_frontier)
            
        st.success("Analysis Complete!")

else:
    # Landing Page State
    st.title("Welcome to your Personal Robo-Advisor")
    st.markdown("""
    👈 **Get started by entering your stock symbols in the sidebar.**
    
    This tool uses **Modern Portfolio Theory** to:
    1. Download historical data.
    2. Simulate thousands of portfolio combinations.
    3. Calculate the **Optimal Tangency Portfolio** (Max Sharpe Ratio).
    """)