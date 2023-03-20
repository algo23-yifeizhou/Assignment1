# Assignment1

### This is a course assignment
My research is based on (Yufeng Han, 2016)A trend factor: Any economic gains from using information over investment horizons? Which used Moving average prices of different lags to construct a trend-following factor, and this factor could capture simultaneously all three stock price trends: the short-, intermediate-, and long-term, by exploiting information in moving average prices of various time lengths whose predictive power is justified by a proposed general equilibrium model

### About the data sets:
The original research included all domestic common stocks listed on the NYSE, AMEX, and Nasdaq stock markets, with monthly data sets. I change the data set to China's stock index futures, with intraday 1-min bar price, and I found this trend-factor still profitable.

### About the strategy:
I used the intraday 1-min stock index future bar to construct this trend factor. The main parameters are moving windows when calculate the OLS betas and target returns period length.

### phased outcomes:

![image](https://user-images.githubusercontent.com/125112527/226239910-6f78f7e2-deaa-4fdc-b519-13bc5f195da6.png)

then I will extend the one-day intraday strategy to days intraday strategy and try to use the factor in a fixed time zone during an intraday trading period
