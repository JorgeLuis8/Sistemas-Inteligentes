import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Carregar dados
df = pd.read_csv("regressao/CAR DETAILS FROM CAR DEKHO.csv")

# Selecionar variáveis
X = df[["year", "km_driven", "fuel", "seller_type", "transmission", "owner"]]
y = df["selling_price"]

# Transformar variáveis categóricas em dummies
X = pd.get_dummies(X, drop_first=True)

# Separar em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Criar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Previsões
y_pred = model.predict(X_test)

# Resultados
print("Coeficientes (β):", model.coef_)
print("Intercepto (β0):", model.intercept_)
print("R² (treino):", model.score(X_train, y_train))
print("R² (teste):", r2_score(y_test, y_pred))
print("RMSE (teste):", np.sqrt(mean_squared_error(y_test, y_pred)))

# Comparar valores reais x previstos
plt.scatter(y_test, y_pred, alpha=0.5, color="purple")
plt.xlabel("Preço Real")
plt.ylabel("Preço Previsto")
plt.title("Regressão Linear Múltipla")
plt.show()
