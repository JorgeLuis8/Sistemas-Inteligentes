import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml


# ============================================================
# 🔹 Função auxiliar: detectar tipo de coluna
# ============================================================
def detectar_tipo_colunas(X_df):
    """Retorna uma lista booleana indicando se cada coluna é categórica."""
    tipos = []
    for col in X_df.columns:
        if pd.api.types.is_object_dtype(X_df[col]) or pd.api.types.is_categorical_dtype(X_df[col]):
            tipos.append(True)   # categórica
        else:
            tipos.append(False)  # numérica
    return tipos


# ============================================================
# 🚀 Classe Naive Bayes híbrido (numérico + categórico)
# ============================================================
class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.std = {}
        self.prob_class = {}
        self.cat_probs = {}
        self.is_categorical = []

    def fit(self, X, y):
        """Treina o modelo com dados numéricos e categóricos."""
        X_df = pd.DataFrame(X)
        y = np.array(y)
        self.is_categorical = detectar_tipo_colunas(X_df)
        self.classes = np.unique(y)

        for c in self.classes:
            Xc = X_df[y == c]
            self.prob_class[c] = len(Xc) / len(X_df)
            self.mean[c] = {}
            self.std[c] = {}
            self.cat_probs[c] = {}

            for col in X_df.columns:
                col_index = X_df.columns.get_loc(col)
                if self.is_categorical[col_index]:
                    # 🔹 Probabilidades categóricas com Laplace smoothing
                    values = X_df[col].unique()
                    counts = Xc[col].value_counts()
                    probs = {v: (counts.get(v, 0) + 1) / (len(Xc) + len(values)) for v in values}
                    self.cat_probs[c][col] = probs
                else:
                    # 🔹 Média e desvio padrão para dados numéricos
                    self.mean[c][col] = Xc[col].mean()
                    self.std[c][col] = Xc[col].std()

    def gauss(self, mean, std, x):
        """Função de densidade Gaussiana (para variáveis numéricas)."""
        std = max(std, 1e-3)  # estabilidade numérica
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    def confidence_per_class(self, X):
        """Retorna a confiança de cada classe (normalizada)."""
        X_df = pd.DataFrame(X)
        confs = []
        for _, x in X_df.iterrows():
            probs = {}
            for c in self.classes:
                prob = self.prob_class[c]
                for col in X_df.columns:
                    col_index = X_df.columns.get_loc(col)
                    if self.is_categorical[col_index]:
                        val = x[col]
                        prob *= self.cat_probs[c][col].get(val, 1e-6)
                    else:
                        prob *= self.gauss(self.mean[c][col], self.std[c][col], x[col])
                probs[c] = prob
            total = np.sum(list(probs.values()))
            confs.append({c: probs[c] / total for c in self.classes})
        return confs

    def predict(self, X):
        """Retorna a classe mais provável (maior confiança)."""
        confs = self.confidence_per_class(X)
        preds = [max(p, key=p.get) for p in confs]
        return np.array(preds)


# ============================================================
# 🧩 MAIN — Testando com a base Titanic
# ============================================================
if __name__ == "__main__":
    # 🔹 1. Carrega a base Titanic do OpenML
    titanic = fetch_openml(name="titanic", version=1, as_frame=True)
    df = titanic.frame

    # 🔹 2. Seleciona colunas mistas (numéricas + categóricas)
    df = df[["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]].dropna()

    # 🔹 3. Define X e y
    X = df.drop("survived", axis=1)
    y = df["survived"].astype(int)

    # 🔹 4. Converte colunas categóricas
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")

    # 🔹 5. (Opcional) Discretiza variáveis contínuas para reduzir outliers
    for col in ["age", "fare"]:
        X[col] = pd.cut(X[col], bins=5, labels=False)

    # 🔹 6. (Opcional) Normaliza variáveis numéricas
 
    X_num = X.select_dtypes(include=np.number)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns, index=X.index)  # ✅ mantém o índice
    X = pd.concat([X_scaled, X.select_dtypes(exclude=np.number)], axis=1)


    # 🔹 7. Divide treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 🔹 8. Treina o modelo
    model = NaiveBayes()
    model.fit(X_train, y_train)

    # 🔹 9. Faz previsões
    y_pred = model.predict(X_test)

    # 🔹 10. Avalia resultados
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

    # 🔹 11. Exibe confianças
    confs = model.confidence_per_class(X_test)
    print("\n🔹 Confiança por classe (0 = não sobreviveu, 1 = sobreviveu):")
    for i, conf in enumerate(confs[:10]):
        print(f"Amostra {i+1}: {conf}")
