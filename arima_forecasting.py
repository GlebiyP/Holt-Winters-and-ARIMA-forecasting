import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima


# Зчитування даних
data = pd.read_csv('C:/Users/Gleb/Desktop/international-airline-passengers.csv', index_col='Month', parse_dates=True)

# Перетворення даних у часовий ряд
ts = data['Thousands of Passengers']

"""
# Розділення даних на тренувальні та тестові набори
train_data = data[:120]
test_data = data[120:]
"""

"""
Перевіримо стаціонарність даних. Можна використати графік ковзного среднього та графік 
ковзного стандартного відхилення для визначення тенденції та варіативності відповідно.
"""

# Визначення ковзних статистик
rolling_mean = ts.rolling(window=12).mean()
rolling_std = ts.rolling(window=12).std()

# Побудова графіка
plt.plot(ts, label='Дані')
plt.plot(rolling_mean, label='Ковзне середнє')
plt.plot(rolling_std, label='Ковзне стандартне відхилення')
plt.legend()
plt.show()

"""
Якщо на графіку видно, що скользяще середнє та стандартне відхилення змінюються з часом, 
то це свідчить про нестаціонарність даних.

Виконаємо диференціювання, якщо необхідно. Якщо дані нестаціонарні, то їх можна перетворити на 
стаціонарні за допомогою диференціювання.
"""

# Диференціювання даних
diff = ts.diff().dropna()

# Перевірка стаціонарності нових даних
rolling_mean = diff.rolling(window=12).mean()
rolling_std = diff.rolling(window=12).std()
plt.plot(diff, label='Диференційовані дані')
plt.plot(rolling_mean, label='Ковзне середнє')
plt.plot(rolling_std, label='Ковзне стандартне відхилення')
plt.legend()
plt.show()

# Розділення даних на тренувальні та тестові набори
train_data = diff[:120]
test_data = diff[120:]

#Визначення моделі та параметрів
#Використана сезонна модель ARIMA, параметри визначаються автоматично
model = auto_arima(train_data, seasonal=True, m=12, suppress_warnings=True)
print(model.order)
print(model.seasonal_order)

#Прогнозування
forecast = model.predict(n_periods=24)

#Виведення результатів
plt.plot(train_data, label='TRAIN')
plt.plot(test_data, label='TEST')
plt.plot(forecast, label='PREDICTION')
plt.title('Train, Test and Predicted Test using ARIMA')
plt.show()