# 📈 Algo-Advisor: Modern Portfolio Theory Optimizer

### A Python-based quantitative tool that automates asset allocation using Monte Carlo simulations and the Sharpe Ratio maximization algorithm.

![Project Status](https://img.shields.io/badge/status-active-success)
![Python Version](https://img.shields.io/badge/python-3.x-blue)
![Finance](https://img.shields.io/badge/domain-Quantitative%20Finance-green)

## 📖 Project Overview
Constructing an investment portfolio that balances risk and reward is often done manually or with basic heuristics. This project applies **Modern Portfolio Theory (MPT)** to mathematically solve for the "Optimal Portfolio."

It uses a **Monte Carlo simulation** (10,000+ iterations) to visualize the Efficient Frontier and utilizes **SciPy's optimization engine** to calculate the exact asset weights that maximize the Sharpe Ratio.

**Key Features:**
* **Automated Data Pipeline:** Fetches real-time adjusted close prices via `yfinance`.
* **Mathematical Optimization:** Uses `scipy.optimize.minimize` (SLSQP method) to solve for max risk-adjusted returns.
* **Risk Analysis:** Calculates annualized volatility, expected returns, and covariance matrices.
* **Visual Analytics:** Generates an "Efficient Frontier" scatter plot and an allocation pie chart.

---

## 🛠️ Tech Stack
* **Core Logic:** `Python`, `NumPy`, `Pandas`
* **Optimization:** `SciPy` (Linear Programming/Minimization)
* **Visualization:** `Matplotlib`, `Seaborn`
* **Data Source:** `Yahoo Finance API (yfinance)`

---

## 📊 How It Works
The application follows a modular "Separation of Concerns" design pattern:

1.  **Data Ingestion (`data_loader.py`):** Downloads historical price data and cleans it (handling missing values/holidays).
2.  **Financial Math (`portfolio_math.py`):**
    * Calculates **Log Returns** (more additive/statistically sound than simple returns).
    * Constructs the **Covariance Matrix** to measure how assets move together.
    * Runs the **Monte Carlo Simulation** to generate the risk/return cloud.
    * Runs the **Optimizer** to find the specific weights for the Max Sharpe Ratio.
3.  **Visualization (`visualizer.py`):** Plots the results using a professional Seaborn style.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/Portfolio_Optimizer.git](https://github.com/YOUR_USERNAME/Portfolio_Optimizer.git)
cd Portfolio_Optimizer