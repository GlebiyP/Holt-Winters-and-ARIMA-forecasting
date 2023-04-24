import pandas as pd
import matplotlib
matplotlib.use('TkAgg',force=True)
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


#international-airline-passengers, Thousands of Passengers, :120
forecast_data = pd.read_csv('C:/Users/Gleb/Desktop/international-airline-passengers.csv', index_col='Month', parse_dates=True)

forecast_data.index.freq = 'MS'

# Split into train and test set
train_airline = forecast_data[:120]
test_airline = forecast_data[120:]

fitted_model = ExponentialSmoothing(train_airline['Thousands of Passengers'], trend='mul', seasonal='mul',
                                    seasonal_periods=12).fit()
test_predictions = fitted_model.forecast(24)

train_airline['Thousands of Passengers'].plot(legend=True, label='TRAIN')
test_airline['Thousands of Passengers'].plot(legend=True, label='TEST', figsize=(9,6))
test_predictions.plot(legend=True, label='PREDICTION')
#test_predictions.plot(legend=True,label='PREDICTION',xlim=['1959-01-01','1961-01-01'])

plt.title('Train, Test and Predicted Test using Holt Winters')
plt.show()