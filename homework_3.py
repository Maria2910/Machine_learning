import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


try:
    df = pd.read_csv("train.csv")
except FileNotFoundError:
    print("Ошибка: Файл 'train.csv' не найден. Укажите правильный путь к файлу.")
    exit()

df = df.dropna()


TARGET = 'SalePrice'


features = [col for col in df.columns if col != TARGET and df[col].dtype in ['int64', 'float64']]
X = df[features]
y = df[TARGET]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


correlation_matrix = df[features].corr()


correlation_threshold = 0.8


correlated_features = set()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname_i = correlation_matrix.columns[i]
            colname_j = correlation_matrix.columns[j]
            correlated_features.add(colname_i) 
            correlated_features.add(colname_j) 

print(f"Обнаружены сильно коррелирующие признаки: {correlated_features}")


X_uncorrelated = df[features].drop(columns=correlated_features, errors='ignore')
features_uncorrelated = [col for col in X_uncorrelated.columns if col != TARGET]
X_scaled_uncorrelated = scaler.fit_transform(X_uncorrelated)



pca = PCA(n_components=2) 
X_pca = pca.fit_transform(X_scaled_uncorrelated)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')


ax.scatter(X_pca[:, 0], X_pca[:, 1], y, c=y, cmap='viridis')

ax.set_xlabel('Главная компонента 1')
ax.set_ylabel('Главная компонента 2')
ax.set_zlabel('SalePrice')
ax.set_title('3D график SalePrice в пространстве PCA')
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X_scaled_uncorrelated, y, test_size=0.2, random_state=42)


alphas = np.logspace(-4, 2, 20)  

rmse_values = []

for alpha in alphas:
    
    lasso = Lasso(alpha=alpha)
    lasso.fit(X_train, y_train)

    
    y_pred = lasso.predict(X_test)

    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_values.append(rmse)

    print(f"Alpha: {alpha}, RMSE: {rmse}")


best_alpha = alphas[np.argmin(rmse_values)]
best_rmse = min(rmse_values)
print(f"\nЛучший Alpha: {best_alpha}, Лучший RMSE: {best_rmse}")


plt.figure(figsize=(10, 6))
plt.plot(alphas, rmse_values, marker='o')
plt.xscale('log')
plt.xlabel('Коэффициент регуляризации (alpha)')
plt.ylabel('RMSE')
plt.title('Зависимость RMSE от коэффициента регуляризации Lasso')
plt.grid(True)
plt.show()


lasso_best = Lasso(alpha=best_alpha)
lasso_best.fit(X_train, y_train)


coefficients = lasso_best.coef_


feature_importance = pd.DataFrame({'Feature': X_uncorrelated.columns, 'Coefficient': coefficients}) # Используем X_uncorrelated.columns


feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)

print("\nНаиболее важные признаки (Lasso):")
print(feature_importance.head(10)) 


plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'].head(10), feature_importance['Abs_Coefficient'].head(10))
plt.xlabel('Абсолютное значение коэффициента Lasso')
plt.ylabel('Признак')
plt.title('Важность признаков (Lasso)')
plt.show()
