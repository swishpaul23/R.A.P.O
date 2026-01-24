import streamlit as st
import pandas as pd
import numpy as np
import data_loader
import portfolio_math
import visualizer

# --- PAGE CONFIG ---
st.set_page_config(page_title="Robo-Advisor Pro", layout="wide")

# --- SIDEBAR ---
st.sidebar.header("🔧 Settings")
default_tickers = "SPY, TLT, GLD, NVDA, XLP"
ticker_input = st.sidebar.text_input("Tickers (comma separated):", value=default_tickers)
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))

if st.sidebar.button("🚀 Run Analysis"):
    
    st.title("📈 AI Portfolio Optimizer")
    
    # 1. GET DATA
    with st.spinner("Downloading Data..."):
        prices = data_loader.fetch_data(tickers, str(start_date), str(end_date))
    
    if prices is None:
        st.error("Error fetching data.")
    else:
        # Sync the ticker list with the actual downloaded data columns
        tickers = prices.columns.tolist()
        # 2. CALC RETURNS
        log_returns = portfolio_math.calculate_log_returns(prices)
        
        # 3. SETUP TABS
        tab1, tab2, tab3 = st.tabs(["🏛️ Static MPT", "🔮 Backtest", "🤖 ML Optimization (HRP)"])
        
        # --- TAB 1: The Classic MPT Optimization ---
        with tab1:
            st.subheader("Modern Portfolio Theory (MPT) Optimization")
            col1, col2 = st.columns(2)
            
            with col1:
                # Run Optimization
                optimal_weights = portfolio_math.optimize_portfolio(log_returns)
                
                # Display Weights
                df_weights = pd.DataFrame({
                    "Ticker": tickers,
                    "Weight": [f"{w:.2%}" for w in optimal_weights]
                })
                st.dataframe(df_weights, hide_index=True)
                
            with col2:
                # Pie Chart
                fig_pie = visualizer.plot_weights_pie(optimal_weights, tickers)
                st.pyplot(fig_pie)
            
            # Efficient Frontier
            with st.spinner("Running Monte Carlo Simulation..."):
                sim_results = portfolio_math.run_simulation(log_returns, num_simulations=1000)
                fig_frontier = visualizer.plot_efficient_frontier(sim_results, optimal_weights, log_returns, tickers)
                st.pyplot(fig_frontier)

        # --- TAB 2: The Reality Check (Backtest + VaR) ---
        with tab2:
            st.subheader("Dynamic Walk-Forward Analysis")
            st.markdown("This simulates rebalancing the portfolio every quarter (63 days) based on the previous year's data.")
            
            with st.spinner("Running Backtest (This takes a moment)..."):
                # Run the Rolling Backtest
                strategy_log_rets = portfolio_math.rolling_backtest(log_returns, window_days=252, rebalance_days=63)
                
                # Calculate Metrics
                var_95 = portfolio_math.calculate_var(strategy_log_rets)
                
                # Compare to Benchmark (First ticker in list)
                benchmark_ticker = tickers[0]
                benchmark_rets = log_returns[benchmark_ticker].loc[strategy_log_rets.index]
                
                total_strat_ret = np.exp(strategy_log_rets.sum()) - 1
                total_bench_ret = np.exp(benchmark_rets.sum()) - 1
                
                # Display Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Strategy Return", f"{total_strat_ret:.2%}")
                m2.metric(f"Benchmark ({benchmark_ticker})", f"{total_bench_ret:.2%}")
                m3.metric("95% VaR (Daily)", f"{var_95:.2%}")
                
                # Plot Backtest Chart
                fig_backtest = visualizer.plot_backtest(strategy_log_rets, benchmark_rets)
                st.pyplot(fig_backtest)
                
                # Plot VaR Histogram
                st.subheader("Risk Distribution")
                fig_var = visualizer.plot_var_distribution(strategy_log_rets, var_95)
                st.pyplot(fig_var)

        # --- TAB 3: Hierarchical Risk Parity (HRP) ---
        with tab3:
            st.subheader("Hierarchical Risk Parity (HRP)")
            st.markdown("""
            **Why Use This?** Standard MPT fails when stocks are highly correlated. 
            HRP uses **Machine Learning (Clustering)** to group similar assets and diversifies risk *across clusters*.
            """)
            
            with st.spinner("Running HRP Algorithm..."):
                # 1. Run HRP
                hrp_weights, linkage_matrix = portfolio_math.optimize_hrp(log_returns)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**HRP Optimized Weights**")
                    df_hrp = pd.DataFrame({
                        "Ticker": tickers,
                        "HRP Weight": [f"{w:.2%}" for w in hrp_weights]
                    })
                    st.dataframe(df_hrp, hide_index=True)
                    
                with col2:
                    # HRP Pie Chart
                    fig_hrp_pie = visualizer.plot_weights_pie(hrp_weights, tickers)
                    st.pyplot(fig_hrp_pie)
                
                # 2. Dendrogram
                st.subheader("Asset Clustering Tree")
                fig_dendro = visualizer.plot_dendrogram(linkage_matrix, tickers)
                st.pyplot(fig_dendro)

else:
    st.info("👈 Select tickers and click 'Run Analysis' to start.")