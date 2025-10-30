import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ===============================
# 1. Carregar dados
# ===============================
df = pd.read_csv("regressao/Car details v3.csv")

# ===============================
# 2. Limpeza das variáveis textuais
# ===============================
# mileage
df["mileage"] = df["mileage"].str.replace(" kmpl", "").str.replace(" km/kg", "")
df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

# engine
df["engine"] = df["engine"].str.replace(" CC", "")
df["engine"] = pd.to_numeric(df["engine"], errors="coerce")

# max_power
df["max_power"] = df["max_power"].str.replace(" bhp", "")
df["max_power"] = pd.to_numeric(df["max_power"], errors="coerce")

# torque -> pega primeiro número encontrado
df["torque"] = df["torque"].str.extract(r"(\d+\.?\d*)")
df["torque"] = pd.to_numeric(df["torque"], errors="coerce")

# seats
df["seats"] = pd.to_numeric(df["seats"], errors="coerce")

# ===============================
# 3. Features adicionais
# ===============================
df["brand"] = df["name"].str.split(" ").str[0]
df["car_age"] = 2025 - df["year"]

owner_map = {
    "Test Drive Car": 0,
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4
}
df["owner_num"] = df["owner"].map(owner_map)

# ===============================
# 4. Remover outliers acima de 1kk
# ===============================
df = df[df["selling_price"] <= 1_000_000]

print(f"📊 Após remoção de outliers: {df.shape[0]} carros restantes")

# ===============================
# 5. Definir X e y
# ===============================
X = df.drop(["selling_price", "name", "year", "owner"], axis=1)
y = df["selling_price"]

categorical_cols = ["brand", "fuel", "seller_type", "transmission"]
numeric_cols = ["km_driven", "mileage", "engine", "max_power", "torque", "seats", "car_age", "owner_num"]

# ===============================
# 6. Pré-processamento (com imputação)
# ===============================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),  # preenche NaN com média
            ("scaler", StandardScaler())
        ]), numeric_cols)
    ]
)

# ===============================
# 7. Pipeline com regressão linear
# ===============================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# ===============================
# 8. Treino/Teste
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ===============================
# 9. Avaliação
# ===============================
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n✅ Resultados com variáveis técnicas (mileage, engine, power, torque, seats):")
print(f"RMSE (teste): {rmse:.2f}")
print(f"R² (teste): {r2:.4f}")

# ===============================
# 10. Antes da regressão
# ===============================
plt.figure(figsize=(8,6))
plt.scatter(df["car_age"], df["selling_price"], alpha=0.5, edgecolors="k")
plt.xlabel("Idade do Carro (anos)")
plt.ylabel("Preço de Venda")
plt.title("Antes da Regressão - Dispersão dos Dados (≤ 1kk)")
plt.show()

# ===============================
# 11. Depois da regressão
# ===============================
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors="k")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
plt.xlabel("Preço Real")
plt.ylabel("Preço Predito")
plt.title("Depois da Regressão - Real vs Predito")
plt.show()
