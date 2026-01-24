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

# 1. Portfolio Tickers
default_tickers = "SPY, TLT, GLD, NVDA, XLP"
ticker_input = st.sidebar.text_input("Portfolio Tickers (comma separated):", value=default_tickers)
portfolio_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# 2. Benchmark Selection
st.sidebar.subheader("Benchmark Settings")
benchmark_options = ["SPY (S&P 500)", "QQQ (Nasdaq 100)", "DIA (Dow Jones)", "IWM (Russell 2000)", "Custom"]
benchmark_choice = st.sidebar.selectbox("Select Benchmark for Comparison:", benchmark_options)

if benchmark_choice == "Custom":
    custom_bench = st.sidebar.text_input("Enter Custom Benchmark Ticker:", value="SPY")
    benchmark_ticker = custom_bench.strip().upper()
else:
    # Extract ticker from string like "SPY (S&P 500)"
    benchmark_ticker = benchmark_choice.split(" ")[0]

# 3. Dates
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))

if st.sidebar.button("🚀 Run Analysis"):
    
    st.title("📈 AI Portfolio Optimizer")
    
    # 1. GET DATA (Portfolio + Benchmark)
    with st.spinner("Downloading Data..."):
        # We need to fetch the benchmark data even if it's not in the portfolio
        all_tickers_to_fetch = list(set(portfolio_tickers + [benchmark_ticker]))
        prices = data_loader.fetch_data(all_tickers_to_fetch, str(start_date), str(end_date))
    
    if prices is None:
        st.error("Error fetching data.")
    else:
        # Sync lists with actual downloaded data
        downloaded_tickers = prices.columns.tolist()
        
        # Verify we have the benchmark and at least some portfolio assets
        if benchmark_ticker not in downloaded_tickers:
            st.error(f"Error: Could not download data for benchmark '{benchmark_ticker}'.")
        else:
            # Filter portfolio tickers to only those that downloaded successfully
            valid_portfolio_tickers = [t for t in portfolio_tickers if t in downloaded_tickers]
            
            if not valid_portfolio_tickers:
                st.error("Error: No valid portfolio tickers found.")
            else:
                # 2. SEPARATE DATA
                # Portfolio Data (For Optimization)
                portfolio_prices = prices[valid_portfolio_tickers]
                portfolio_log_returns = portfolio_math.calculate_log_returns(portfolio_prices)
                
                # Benchmark Data (For Comparison)
                benchmark_prices = prices[[benchmark_ticker]] # Double bracket to keep as DataFrame
                benchmark_log_returns = portfolio_math.calculate_log_returns(benchmark_prices)
                
                # 3. SETUP TABS
                tab1, tab2, tab3 = st.tabs(["🏛️ Static MPT", "🔮 Backtest", "🤖 ML Optimization (HRP)"])
                
                # --- TAB 1: Static MPT ---
                with tab1:
                    st.subheader("Modern Portfolio Theory (MPT) Optimization")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        optimal_weights = portfolio_math.optimize_portfolio(portfolio_log_returns)
                        
                        df_weights = pd.DataFrame({
                            "Ticker": valid_portfolio_tickers,
                            "Weight": [f"{w:.2%}" for w in optimal_weights]
                        })
                        st.dataframe(df_weights, hide_index=True)
                        
                    with col2:
                        fig_pie = visualizer.plot_weights_pie(optimal_weights, valid_portfolio_tickers)
                        st.pyplot(fig_pie)
                    
                    with st.spinner("Running Monte Carlo Simulation..."):
                        sim_results = portfolio_math.run_simulation(portfolio_log_returns, num_simulations=1000)
                        fig_frontier = visualizer.plot_efficient_frontier(sim_results, optimal_weights, portfolio_log_returns, valid_portfolio_tickers)
                        st.pyplot(fig_frontier)

                # --- TAB 2: Backtest vs Benchmark ---
                with tab2:
                    st.subheader(f"Dynamic Analysis vs {benchmark_ticker}")
                    st.markdown("This simulates rebalancing the portfolio every quarter (63 days).")
                    
                    with st.spinner("Running Backtest..."):
                        # Run Strategy
                        strategy_log_rets = portfolio_math.rolling_backtest(portfolio_log_returns, window_days=252, rebalance_days=63)
                        
                        # Align Benchmark with Strategy dates
                        aligned_bench_rets = benchmark_log_returns[benchmark_ticker].loc[strategy_log_rets.index]
                        
                        # Calc Metrics
                        var_95 = portfolio_math.calculate_var(strategy_log_rets)
                        total_strat_ret = np.exp(strategy_log_rets.sum()) - 1
                        total_bench_ret = np.exp(aligned_bench_rets.sum()) - 1
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Strategy Return", f"{total_strat_ret:.2%}")
                        m2.metric(f"Benchmark ({benchmark_ticker})", f"{total_bench_ret:.2%}")
                        m3.metric("95% VaR (Daily)", f"{var_95:.2%}")
                        
                        fig_backtest = visualizer.plot_backtest(strategy_log_rets, aligned_bench_rets)
                        st.pyplot(fig_backtest)
                        
                        st.subheader("Risk Distribution")
                        fig_var = visualizer.plot_var_distribution(strategy_log_rets, var_95)
                        st.pyplot(fig_var)

                # --- TAB 3: HRP ---
                with tab3:
                    st.subheader("Hierarchical Risk Parity (HRP)")
                    with st.spinner("Running HRP Algorithm..."):
                        hrp_weights, linkage_matrix = portfolio_math.optimize_hrp(portfolio_log_returns)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            df_hrp = pd.DataFrame({
                                "Ticker": valid_portfolio_tickers,
                                "HRP Weight": [f"{w:.2%}" for w in hrp_weights]
                            })
                            st.dataframe(df_hrp, hide_index=True)
                        with col2:
                            fig_hrp_pie = visualizer.plot_weights_pie(hrp_weights, valid_portfolio_tickers)
                            st.pyplot(fig_hrp_pie)
                        
                        st.subheader("Asset Clustering Tree")
                        fig_dendro = visualizer.plot_dendrogram(linkage_matrix, valid_portfolio_tickers)
                        st.pyplot(fig_dendro)

else:
    st.info("👈 Select tickers and click 'Run'")