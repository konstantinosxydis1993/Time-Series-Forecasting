import prophet
import pandas as pd
from prophet import Prophet
from pandas import to_datetime
import datetime
from tkinter import filedialog as fd
import tkinter as tk

# Pop-up window to choose input data
root = tk.Tk()
Path = fd.askopenfilename(title = 'Select Data')
root.destroy()
df = pd.read_excel(Path)

# Output data location
export_file = r'C:\Users\Xydis\Documents\Arima\Github\prophet_output.xlsx'

# Choose and format timeseries dataframe
df_output = df 
Sales = 'Airport Sales'
df_output = df_output[['DATE',Sales]].dropna(subset=[Sales])
df_output['DATE'] = pd.to_datetime(df_output['DATE']).dt.strftime('%Y-%m')
df_output = df_output.rename({'DATE': 'ds', Sales: 'y'}, axis='columns')

# Choose Exogenous
exogenous1 = "Airport Passengers"
df_output['exogenous'] = df[exogenous1]

# Create forecasted exogenous df
exogenous_fc = df.iloc[len(df_output):len(df_output)+12].reset_index()

# Create futurte dates df
future_list = (pd.date_range(df_output['ds'].iloc[-1] , periods=12, freq='M')+ datetime.timedelta(days=1)).tolist()
future = pd.DataFrame(future_list, columns = ['ds'])
future['ds'] = future['ds'].dt.strftime('%Y-%m')
                                        
# Create forecasted exogenous df
future['exogenous'] = exogenous_fc[exogenous1]

# inlcude exogenous parameters
model = Prophet()
model.add_regressor('exogenous')

# Fit the model
model.fit(df_output)

# Forecast test (present)
test = model.predict()

# Forecast future
train = model.predict(future)

# Output
output = pd.concat([test, train], ignore_index=True)
output.to_excel(export_file)


