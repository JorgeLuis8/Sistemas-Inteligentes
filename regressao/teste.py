import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ================================
# 1. Carregar e preparar os dados
# ================================
df = pd.read_csv("regressao/CAR DETAILS FROM CAR DEKHO.csv")

# Variáveis independentes e alvo
X = df[["year", "km_driven", "fuel", "seller_type", "transmission", "owner"]]
y = df["selling_price"].values.reshape(-1, 1)

# One-hot encoding para variáveis categóricas
X = pd.get_dummies(X, drop_first=True)

# Normalizar X
X = (X - X.mean()) / X.std()

# Normalizar y
y_mean, y_std = y.mean(), y.std()
y_norm = (y - y_mean) / y_std

# Adicionar coluna de 1s (intercepto)
X.insert(0, "intercepto", 1)

# Converter para numpy
X = X.values

# ================================
# 2. Separar treino e teste
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_norm, test_size=0.3, random_state=42
)

# ================================
# 3. Descida do Gradiente
# ================================
alpha = 0.01   # taxa de aprendizado
epochs = 1000  # iterações
n, d = X_train.shape
theta = np.zeros((d, 1))

for i in range(epochs):
    y_pred = X_train @ theta
    error = y_pred - y_train
    cost = (1/n) * np.sum(error ** 2)

    gradient = (2/n) * (X_train.T @ error)
    theta -= alpha * gradient

    if i % 100 == 0:
        print(f"Iter {i}: Custo = {cost:.4f}")

# ================================
# 4. Avaliação
# ================================
# Previsões normalizadas
y_pred_test_norm = X_test @ theta

# Desnormalizar para escala real
y_pred_test = y_pred_test_norm * y_std + y_mean
y_test_real = y_test * y_std + y_mean

# Métricas
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_test))
r2 = r2_score(y_test_real, y_pred_test)

print("\nCoeficientes (θ):", theta.flatten())
print("RMSE (teste):", rmse)
print("R² (teste):", r2)

# ================================
# 5. Visualização
# ================================
plt.figure(figsize=(8,6))
plt.scatter(y_test_real, y_pred_test, alpha=0.5, color="blue", label="Previsões")
plt.plot([y_test_real.min(), y_test_real.max()],
         [y_test_real.min(), y_test_real.max()],
         color="red", linestyle="--", linewidth=2, label="Ideal (y = x)")
plt.xlabel("Preço real (R$)")
plt.ylabel("Preço previsto (R$)")
plt.title("Regressão Linear Múltipla - Descida do Gradiente")
plt.legend()
plt.show()
