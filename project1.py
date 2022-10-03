import pandas as pd
import numpy as np
data = pd.read_csv('dataset.csv')
cleaned_data = pd.concat([data[:910], data[919:]])
params_data = cleaned_data.drop(['Unnamed: 0', 'year'], axis=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(params_data))

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
scaled_data = pd.DataFrame(imp.fit_transform(scaled_data))

train = pd.concat([scaled_data[start:start+9] for start in range(0,scaled_data.shape[0], 10)])
test = pd.DataFrame([scaled_data.iloc[start+9] for start in range(0,scaled_data.shape[0], 10)])

x_train, y_train = train.iloc[:, 0:8] ,train.iloc[:, 8:]
x_test, y_test = test.iloc[:, 0:8] ,test.iloc[:, 8:]

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train, y_train)

y_pred = reg.predict(x_test)

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_pred, y_test) ** 0.5