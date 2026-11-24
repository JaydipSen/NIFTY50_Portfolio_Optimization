\# NIFTY 50 Portfolio Optimization: MVP vs HRP



A Python-based interactive portfolio optimization and backtesting application built using \*\*Streamlit\*\*. The application compares two portfolio construction strategies:



1\. \*\*Minimum Variance Portfolio (MVP)\*\*

2\. \*\*Hierarchical Risk Parity (HRP)\*\*



The application uses historical daily closing prices of selected NIFTY 50 stocks to construct portfolios, evaluate their out-of-sample performance, and compare their risk-return characteristics.



\## 1. Project Overview



Portfolio optimization plays an important role in investment management by balancing expected returns against portfolio risk.



This project implements and compares two portfolio allocation approaches:



\- \*\*MVP:\*\* Minimizes portfolio variance using covariance-based optimization.

\- \*\*HRP:\*\* Uses hierarchical clustering and recursive bisection to allocate portfolio weights based on the risk structure of assets.



The application provides an interactive dashboard through which users can select stocks, configure analysis periods, optimize portfolios, visualize allocations, and evaluate investment performance.



\## 2. Dataset and Study Period



The application retrieves historical stock price data from Yahoo Finance using the `yfinance` Python library.



| Parameter | Description |

|---|---|

| Stock universe | Selected NIFTY 50 constituents |

| Data frequency | Daily |

| Price variable | Close |

| Data source | Yahoo Finance |

| Overall period | January 2025 – October 2026 |

| Training period | January 2025 – April 2026 |

| Testing period | May 2026 – October 2026 |

| Return calculation | Daily percentage returns |



Only the \*\*Close price\*\* is used for portfolio construction and backtesting.



\### Default Stock Universe



The application initially includes the following ten stocks:



\- HDFC Bank (HDFCBANK.NS)

\- ICICI Bank (ICICIBANK.NS)

\- Reliance Industries (RELIANCE.NS)

\- Bharti Airtel (BHARTIARTL.NS)

\- Larsen \& Toubro (LT.NS)

\- State Bank of India (SBIN.NS)

\- Axis Bank (AXISBANK.NS)

\- Infosys (INFY.NS)

\- Kotak Mahindra Bank (KOTAKBANK.NS)

\- Mahindra \& Mahindra (M\&M.NS)



Users can modify the stock universe through the Streamlit sidebar.



\*\*Note:\*\* The default stock universe is an illustrative selection of large NIFTY 50 constituents, not a dynamically reconstructed historical index universe.



\## 3. Portfolio Optimization Methodologies



\### 3.1 Minimum Variance Portfolio (MVP)



The Minimum Variance Portfolio minimizes portfolio variance subject to investment constraints.



The optimization problem is:



Minimize:



\\\[

\\sigma\_p^2 = w^T \\Sigma w

\\]



Subject to:



\\\[

\\sum\_{i=1}^{N}w\_i=1

\\]



\\\[

0 \\leq w\_i \\leq 1

\\]



Where:



\- \\(w\_i\\) represents the portfolio weight of asset \\(i\\).

\- \\(\\Sigma\\) represents the covariance matrix of asset returns.

\- \\(N\\) represents the number of selected stocks.



The optimization is performed using the Sequential Least Squares Programming (SLSQP) algorithm.



\### 3.2 Hierarchical Risk Parity (HRP)



Hierarchical Risk Parity constructs portfolios using the correlation structure of assets.



The implementation follows these steps:



1\. Calculate the covariance and correlation matrices.

2\. Convert correlations into a distance matrix.

3\. Apply hierarchical clustering using single linkage.

4\. Determine the hierarchical ordering of assets.

5\. Perform recursive bisection.

6\. Allocate portfolio weights based on cluster variances.



Unlike conventional mean-variance optimization, HRP does not require direct inversion of the covariance matrix.



\## 4. Backtesting Framework



The study follows a train-test framework.



\*\*Training period: January 2025 – April 2026\*\*



\- Calculate daily stock returns.

\- Estimate covariance and correlation matrices.

\- Optimize MVP portfolio weights.

\- Construct HRP portfolio weights.



\*\*Testing period: May 2026 – October 2026\*\*



\- Apply the optimized training-period weights.

\- Calculate daily portfolio returns.

\- Evaluate cumulative portfolio growth.

\- Compare portfolio risk and performance.



Portfolio weights remain fixed during the testing period to avoid look-ahead bias from re-optimizing with future observations.



\## 5. Performance Evaluation Metrics



The application calculates the following performance indicators:



| Metric | Description |

|---|---|

| Total Return | Cumulative portfolio return during the testing period |

| CAGR | Compound annual growth rate |

| Annualized Risk | Annualized standard deviation of daily returns |

| Sharpe Ratio | Risk-adjusted return relative to the risk-free rate |

| Maximum Drawdown | Maximum observed peak-to-trough portfolio decline |



The annualized volatility is calculated as:



\\\[

\\sigma\_{annual}=\\sigma\_{daily}\\sqrt{252}

\\]



The annualized Sharpe ratio is:



\\\[

Sharpe=\\frac{(\\bar r\_d-r\_f/252)}{\\sigma\_d}\\sqrt{252}

\\]



Where:



\- \\(\\bar r\_d\\) is the average daily portfolio return.

\- \\(r\_f\\) is the annual risk-free rate.

\- \\(\\sigma\_d\\) is the standard deviation of daily portfolio returns.



The default annual risk-free rate is 6%, which can be modified through the application sidebar.



\## 6. Dashboard Features



The Streamlit application provides the following visualizations and analytical outputs:



\### Performance Analysis

\- MVP versus HRP total returns.

\- Cumulative portfolio growth.

\- Annualized return comparison.

\- Annualized volatility comparison.

\- Sharpe ratio comparison.

\- Rolling 21-day annualized volatility.



\### Portfolio Allocation Analysis

\- MVP portfolio allocation pie chart.

\- HRP portfolio allocation pie chart.

\- Comparative portfolio weight table.



\### Risk-Return Analysis

\- Risk-return scatter plot.

\- Comparative evaluation of portfolio strategies.



\### Data Export

Users can download:



\- Portfolio allocation weights in CSV format.

\- Historical closing prices.

\- Out-of-sample portfolio returns.

\- Cumulative portfolio wealth series.



\## 7. Technologies Used



The project uses the following Python libraries:



| Library | Purpose |

|---|---|

| Python | Programming language |

| Streamlit | Interactive dashboard |

| yfinance | Historical stock price retrieval |

| Pandas | Data processing |

| NumPy | Numerical computations |

| SciPy | Portfolio optimization and hierarchical clustering |

| Plotly | Interactive visualizations |



\## 8. Installation and Execution



\### Step 1: Clone the repository



```bash

git clone https://github.com/YOUR\_USERNAME/YOUR\_REPOSITORY.git

```



\### Step 2: Navigate to the project directory



```bash

cd YOUR\_REPOSITORY

```



\### Step 3: Install dependencies



```bash

pip install -r requirements.txt

```



\### Step 4: Run the Streamlit application



```bash

streamlit run app.py

```



The application will generally be accessible at:



http://localhost:8501



\## 9. Deployment on Streamlit Community Cloud



The application can be deployed online using Streamlit Community Cloud.



1\. Upload `app.py` and `requirements.txt` to a GitHub repository.

2\. Log in to \[Streamlit Community Cloud](https://share.streamlit.io/).

3\. Connect your GitHub account.

4\. Select the repository containing the application.

5\. Specify `app.py` as the main application file.

6\. Click Deploy.



Once deployed, the application will be accessible through a public Streamlit URL, subject to the repository and deployment settings.



\## 10. Project Structure



```text

NIFTY50-Portfolio-Optimization/

│

├── app.py

│   └── Main Streamlit application

│

├── requirements.txt

│   └── Required Python libraries

│

└── README.md

&#x20;   └── Project documentation

```



\## 11. Important Limitations



1\. Historical prices are retrieved from Yahoo Finance and depend on data availability.

2\. Unadjusted Close prices are used, so dividends are excluded and stock splits may affect historical price returns.

3\. The selected stocks may introduce survivorship and selection bias if used to represent the historical NIFTY 50 universe.

4\. Transaction costs, brokerage charges, taxes, liquidity constraints and slippage are not incorporated.

5\. Portfolio weights are estimated once using training data and are not rebalanced during the testing period.

6\. Optimization results are sensitive to the selected training period and historical covariance estimates.



The application is intended for academic research, educational purposes, and comparative portfolio analysis. It should not be interpreted as personalized investment advice.



\## 12. Research Applications



This project can be extended to investigate:



\- Comparative performance of classical and hierarchical portfolio optimization.

\- Risk diversification across NIFTY 50 stocks.

\- Stability of portfolio weights under changing market conditions.

\- Portfolio performance under different training and testing windows.

\- The impact of market volatility on portfolio allocation.

\- Comparative performance of alternative portfolio construction techniques.



\## 13. Acknowledgements



\- Yahoo Finance for historical stock price data.

\- Streamlit for the interactive application framework.

\- Python scientific computing ecosystem for numerical optimization and statistical analysis.



\---



\*\*Keywords:\*\* Portfolio Optimization, Minimum Variance Portfolio, Hierarchical Risk Parity, NIFTY 50, Financial Analytics, Portfolio Diversification, Risk Management, Python, Streamlit, Backtesting.

