import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

# Read train.csv
train = pd.read_csv('train.csv')

# Convert the Date column to datetime type
train['Date'] = pd.to_datetime(train['Date'])

# Filtering out only store #1 and department #1 to a new dataframe
# with the required column names
store1_dept1_sales = train[(train['Store'] == 1) & (
    train['Dept'] == 1)][['Date', 'Weekly_Sales']]
store1_dept1_sales.columns = ['ds', 'y']

# Fit Prophet with the data and carry out the forecast
# model = Prophet()
# model.fit(store1_dept1_sales)
# future = model.make_future_dataframe(periods=52, freq='W')
# forecast = model.predict(future)
# figure = model.plot(forecast)
# plt.show()
# forecast.to_csv('sales_forecast.csv')

# Build the holiday dataframe combining the current and future holiday information
train_store1_dep1_holiday = train[(train['Store'] == 1) & (
    train['Dept'] == 1) & (train['IsHoliday'] == True)][['Date', 'IsHoliday']]
test = pd.read_csv('test.csv')
test_store1_dep1_holiday = test[(test['Store'] == 1) & (
    test['Dept'] == 1) & (test['IsHoliday'] == True)][['Date', 'IsHoliday']]
combined_holiday = pd.concat(
    [train_store1_dep1_holiday, test_store1_dep1_holiday])
combined_holiday['Date'] = pd.to_datetime(combined_holiday['Date'])
combined_holiday.columns = ['ds', 'holiday']

# Prophet requires the holiday column to be string so we need to convert the boolean
# to string
combined_holiday['holiday'] = combined_holiday['holiday'].map(
    {True: 'Yes', False: 'No'})

# Pass in the holiday information
model = Prophet(holidays=combined_holiday)
model.fit(store1_dept1_sales)
future = model.make_future_dataframe(periods=52, freq='W')
forecast_holiday = model.predict(future)
figure = model.plot(forecast_holiday)
figure2 = model.plot_components(forecast_holiday)
plt.show()
forecast_holiday.to_csv('sales_forecast_holiday.csv')
