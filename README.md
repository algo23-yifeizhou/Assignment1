# Assignment1

### What is it?

This is a research based on
 
  "Han, & Kong, L. (2022). A trend factor in commodity futures markets: Any economic gains from using information over investment horizons? The Journal of Futures Markets, 42(5), 803–822. https://doi.org/10.1002/fut.22291"


### Abstract
In this paper, we provide a trend factor that captures simultaneously all three stock price trends: the short-, intermediate-, and long-term, by exploiting information in moving average prices of various time lengths whose predictive power is justified by a proposed general equilibrium model. It outperforms substantially the well-known short-term reversal, momentum, and long-term reversal factors, which are based on the three price trends separately, by more than doubling their Sharpe ratios. During the recent financial crisis, the trend factor earns 0.75% per month, while the market loses −2.03% per month, the short-term reversal factor loses −0.82%, the momentum factor loses −3.88%, and the long-term reversal factor barely gains 0.03%. The performance of the trend factor is robust to alternative formations and to a variety of control variables. From an asset pricing perspective, it also performs well in explaining cross-section stock returns.

### Get Started
This project is based on Python
1. conda create -n assign3 python=3.7
2. conda activate assign3 pip install -r requirements.txt

### My efforts and improvement
The above model used Moving average prices of different lags to construct a trend-following factor, and this factor could capture simultaneously all three asset price trends: the short-term, intermediate-term, and long-term, by exploiting information in moving average prices of various time lengths whose predictive power is justified by a proposed general equilibrium model.

The original research included all domestic common stocks listed on the NYSE, AMEX, and Nasdaq stock markets, with monthly data sets. I change the data set to China's Share Price Index Futures (SPIF), with intraday 1-min bar price, and I found this trend-factor is hardly profitable. And more work need to be done in the future procedure.

### Data sets
1. IH: SPIF of SSE 50
2. IC: SPIF of CSI 500
3. IF: SPIF of CSI 300



### About the strategy:
I used the intraday 1-min stock index future bar to construct this trend factor. The main parameters are moving windows when calculate the OLS regression betas and target returns period length.

### Phased outcomes:
#### Parameters
Symbol: IH
predict future X min return: X=5
Moving Average window Length: (5, 10, 15, 30)
Beta OLS Regression Window Length: 30
Beta Expectation Window Length: 10
#### Backtesting results (2016-06-01)
![image](https://github.com/algo23-yifeizhou/Assignment1/assets/125112527/8e882c10-75ff-442d-a833-9009488e71ef)
#### Backtesting results (2016-01-20)
![image](https://github.com/algo23-yifeizhou/Assignment1/assets/125112527/e8338234-4afe-404a-8bfd-83ebac55402a)

### Further works to be done
1. Then I will extend the one-day intraday strategy to days intraday strategy and try to use the factor in a fixed time zone during an intraday trading period
2. Switch data sets and parameters to those mentioned in the paper, and then testify the model again
3. Change the Parameters and find a generalized methods to avoid over-fit 
