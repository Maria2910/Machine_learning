import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


data = pd.read_csv('/Users/kechk/PycharmProjects/ml/tables/bikes_rent.csv')


X = data[['weathersit', 'temp', 'atemp', 'hum', 'windspeed(mph)', 'windspeed(ms)']]
y = data['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



def predict_demand(weather, temp, atemp, hum, windspeed_mph, windspeed_ms):
    input_data = np.array([[weather, temp, atemp, hum, windspeed_mph, windspeed_ms]])
    return model.predict(input_data)

example_prediction = predict_demand(1, 0.3, 0.3, 0.5, 10, 4.47)
print(f'Прогнозируемое количество арендованных велосипедов: {example_prediction[0]}')


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

model_pca = LinearRegression()
model_pca.fit(X_train_pca, y_train_pca)

plt.figure(figsize=(10, 6))
plt.scatter(X_test_pca[:, 0], y_test_pca, color='blue', label='Actual')
plt.scatter(X_test_pca[:, 0], model_pca.predict(X_test_pca), color='red', label='Predicted')
plt.xlabel('Principal Component 1')
plt.ylabel('Count')
plt.title('Prediction of Bike Rentals using PCA')
plt.legend()
plt.show()

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)


lasso_coef = pd.Series(lasso.coef_, index=X.columns)
print("Коэффициенты Lasso:")
print(lasso_coef)

most_influential_feature = lasso_coef.idxmax()
print(f'Наиболее влиятельный признак: {most_influential_feature}')
