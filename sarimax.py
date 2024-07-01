import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
#import missingno as msno
import statsmodels.api as sm
import pmdarima as pm
import datetime as dt
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from tkinter import filedialog as fd
import tkinter as tk

# Pop-up window to choose input data
root = tk.Tk()
Path = fd.askopenfilename(title = 'Select Data')
root.destroy()

# Import Excel Files
df = pd.read_excel(Path)
exogenous_csv = pd.read_excel(Path)
export_file = r'C:\Users\Xydis\Documents\Arima\Github\sarimax_output.xlsx'

# Choose timeseries dataframe
df_output = df.dropna(subset=['Airport Sales'])
# Choose timeseries column
Sales = 'Airport Sales' ##############
print(Sales)

# Choose Forecasted Period
n_periods =  12 # forecasted months
n = len(df_output) # train periods

# Choose Forecasted Period for TEST
n_periods_test = 12 # forecasted months for tests
n_test=len(df_output) - n_periods_test 

# Stationarity Check - Decompose
decomposition = sm.tsa.seasonal_decompose(df_output[Sales][:n], model='additive', period=12)
decomp_trend = decomposition.trend
decomp_seas = decomposition.seasonal
decomp_resid = decomposition.resid
fig = decomposition.plot()
plt.show()

# Choose Exogenous Parameters
exogenous = exogenous_csv[[
                "Airport Passengers"
                ]]

# Seasonal Arima with Exogenous Variables - Train Model

train=df_output[:n]
SARIMAX_model = pm.auto_arima(train[[Sales]],
                             exogenous=exogenous[:n],
                           start_p=1,
                           start_q=1,
                           test='adf',
                           max_p=5,
                           max_q=5, 
                           m=12,
                           start_P=0, 
                           seasonal=True,
                           d=None, 
                           D=1, 
                           trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

fitted_test, confint = SARIMAX_model.predict(n_periods=n_periods,
                                            return_conf_int=True,                                            
                                            exogenous=exogenous[n:n+n_periods],                                             
                                            alpha=0.05)

# SARIMAX Model Diagnostics

print(SARIMAX_model.summary())
SARIMAX_model.plot_diagnostics(lags=25,figsize=(7,5))
plt.show



## Tests for Autocorelation and Heteroskedasticity

## Residuals
residual = SARIMAX_model.resid()
plt.plot(residual)
plt.title('residuals')
plt.show()

#  Ljung Box Test for Autocorrelation
Ljung1 = acorr_ljungbox(residual, return_df=True, lags=15)

# Breusch Pagan for Heteroskedasticity
names = ['Lagrange multiplier statistic', 'p-value','f-value', 'f p-value']
exogenousBP = exogenous[~exogenous.isin([np.nan, np.inf, -np.inf]).any(axis=1)] 
exogenousBP = exogenousBP[:len(residual)]
# Get the test result
test_result = sms.het_breuschpagan(residual, exogenousBP)
result = lzip(names, test_result)



# Create Export Dataframe
fitted_test_df = pd.DataFrame(fitted_test, columns=['a'])

# Extract the 'Sales' series from df_output
sales_series = df_output[Sales][:n]

# Concatenate the two series
ts = pd.concat([sales_series, fitted_test_df['a']])

# Initialize period
period = n

# Create Real_Forecast list
Real_Forecast = ['Real' if i < period else 'Forecast' for i in range(len(ts))]

a_list = list(range(0, (len(ts))))
Export = pd.DataFrame(index=a_list)
Export['Sales']= ts.tolist()
Export['DATE']= pd.date_range('2014-12-01 00:00:00', periods=len(ts), freq='M') + dt.timedelta(days=1)
Export['Real-Forecast'] = Real_Forecast


# Include Trend, Seasonality and Cofidence Intervals
Export['Trend'] = decomp_trend
Export['Seasonality'] = decomp_seas
Export['Residual'] = decomp_resid
Export['Conf_Interval Upper'] = np.append(np.full((1, len(Export)-len(confint)),np.nan),confint[:,0])      
Export['Conf_Interval Lower'] = np.append(np.full((1, len(Export)-len(confint)),np.nan),confint[:,1])   




# SARIMAX Test Model for Accuracy

train=df_output[:n_test]
test=df_output[n_test:(n_test+n_periods_test)]

SARIMAX_model = pm.auto_arima(train[[Sales]],
                              exogenous=exogenous[:n_test],
                           start_p=1,
                           start_q=1,
                           test='adf',
                           max_p=10, max_q=10, 
                           m=12,
                           start_P=0, 
                           seasonal=True,
                           d=None, 
                           D=1, 
                           trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

fitted_test2, confint2 = SARIMAX_model.predict(n_periods=n_periods_test,
                                            return_conf_int=True,                                            
                                            exogenous=exogenous[n_test:n_test+n_periods_test],                                             
                                            alpha=0.05)

fitted_test_df2 = pd.DataFrame(fitted_test2, columns=['a'])
sales_series = df_output[Sales][:n_test]
a_series = fitted_test_df2['a']
ts2 = pd.concat([sales_series, a_series])
ts2_list = ts2.tolist()
num_nans = len(Export) - len(ts2)
nan_padding = np.full((num_nans,), np.nan)
test_sales_combined = np.append(ts2_list, nan_padding)
Export['Test Sales'] = test_sales_combined


## Model Evaluation ##

#Mean Absolute Percentage Average
mape = np.mean(np.abs(test[Sales].to_numpy()-fitted_test2)/np.abs(test[Sales].to_numpy()))  # MAPE - Mean Absolute Percentage Error
#Residual Mean Squared Error
rmse = sqrt(mean_squared_error(test[Sales].to_numpy(),fitted_test2))
#Residual Mean Squared Percentage Error
rmspe = np.sqrt(np.mean(np.square((test[Sales].to_numpy() - fitted_test2) / test[Sales].to_numpy()), axis=0))


Export['MAPE'] = mape
Export['MAPE'] = Export['MAPE'].iloc[:1]
Export['RMSPE'] = rmspe
Export['RMSPE'] = Export['RMSPE'].iloc[:1]
Export['RMSE'] = rmse
Export['RMSE'] = Export['RMSE'].iloc[:1]

#Export
Export.to_excel(export_file)


