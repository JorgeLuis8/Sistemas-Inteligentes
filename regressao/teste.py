import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Carregar dados
df = pd.read_csv("regressao\CAR DETAILS FROM CAR DEKHO.csv")

# Variáveis
X = df[["km_driven"]]       # variável explicativa
y = df["selling_price"]     # variável alvo

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
print("Coeficiente angular (β1):", model.coef_[0])
print("Intercepto (β0):", model.intercept_)
print("R² (treino):", model.score(X_train, y_train))
print("R² (teste):", r2_score(y_test, y_pred))
print("RMSE (teste):", np.sqrt(mean_squared_error(y_test, y_pred)))

# Visualização
plt.scatter(X_test, y_test, color="blue", alpha=0.5, label="Dados reais")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Reta de regressão")
plt.xlabel("Quilometragem (km_driven)")
plt.ylabel("Preço de Venda (selling_price)")
plt.title("Regressão Simples: km_driven vs selling_price")
plt.legend()
plt.show()
