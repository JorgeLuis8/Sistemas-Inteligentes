import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Carregar CSV
df = pd.read_csv("lifeexpectancy.csv")
print("Tamanho original:", len(df))
df = df.dropna(subset=["Prevelance of Undernourishment", "Life Expectancy World Bank"])
print("Tamanho depois da limpeza:", len(df))

# Remover linhas onde as colunas principais estão vazias
df = df.dropna(subset=["Prevelance of Undernourishment", "Life Expectancy World Bank"])

# Definir variáveis
X = df[["Prevelance of Undernourishment"]]   # independente
y = df["Life Expectancy World Bank"]         # dependente

# Divisão em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Criar e treinar modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar modelo
print("Coeficiente angular (β1):", model.coef_[0])
print("Intercepto (β0):", model.intercept_)
print("R² (treino):", model.score(X_train, y_train))
print("R² (teste):", model.score(X_test, y_test))
print("RMSE (teste):", np.sqrt(mean_squared_error(y_test, y_pred)))

# Gráfico
plt.scatter(X_test, y_test, color="blue", label="Valores reais (teste)")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Reta de regressão")
plt.xlabel("Prevalência de Subnutrição (%)")
plt.ylabel("Expectativa de Vida (anos)")
plt.title("Regressão Linear Simples: Subnutrição vs Expectativa de Vida")
plt.legend()
plt.show()
