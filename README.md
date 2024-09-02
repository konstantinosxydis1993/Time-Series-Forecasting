This repository concerns time series forecasting, specifically the data is sales in an airport and the passengers per month are used as an exogenous parameter to aid in forcasting. 

Two methods are used, a) SARIMAX and b) the prophet library. SARIMAX stands for Seasonal Auto-Regressive Integrated Moving Average.

ARIMA is made up of three key components:
AR (AutoRegressive): Refers to the relationship between an observation and a certain number of lagged observations (previous time steps). It uses the dependency between an observation and a number of prior observations.
I (Integrated): Refers to the differencing of raw observations (subtracting an observation from an observation at the previous time step) to make the time series stationary, i.e., to remove trends and seasonality. Stationarity means that the statistical properties of the series, such as mean and variance, are constant over time.
MA (Moving Average): Refers to the dependency between an observation and a residual error from a moving average model applied to lagged observations. It captures the error of the model as a linear combination of error terms from past predictions.

ARIMA models are defined by three parameters (p, d, q):
p (Order of AR term): The number of lag observations included in the model (i.e., the number of lagged values to consider). It represents the order of the autoregressive part.
d (Order of Differencing): The number of times that the raw observations are differenced to make the time series stationary. This represents the number of times that the data have had past values subtracted.
q (Order of MA term): The number of lagged forecast errors in the prediction equation. It represents the order of the moving average part.

The auto-arima method is used whcih automatically calcualtes p, d and q, while seasonality is set to 12 months.

As far as the prophet library is concerned, Prophet models time series data as an additive model, where the time series is decomposed into several components:

y(t)=g(t)+s(t)+h(t)+ϵ(t)

Where:
g(t): Trend component, which models non-periodic changes in the value of the time series.
s(t): Seasonal component, which captures periodic changes (e.g., daily, weekly, yearly).
h(t): Holiday component, which accounts for the effects of holidays or special events.
ϵ(t): Error term, which represents any noise or unexplained variance in the time series.

