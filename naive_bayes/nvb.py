import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

def detectar_tipo_colunas(X_df):
    tipos = []
    for col in X_df.columns:
        if pd.api.types.is_object_dtype(X_df[col]) or pd.api.types.is_categorical_dtype(X_df[col]):
            tipos.append(True)
        else:
            tipos.append(False)
    return tipos

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = {}
        self.std = {}
        self.prob_class = {}
        self.cat_probs = {}
        self.is_categorical = []

    def fit(self, X, y):
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
                    values = X_df[col].unique()
                    counts = Xc[col].value_counts()
                    probs = {v: (counts.get(v, 0) + 1) / (len(Xc) + len(values)) for v in values}
                    self.cat_probs[c][col] = probs
                else:
                    self.mean[c][col] = Xc[col].mean()
                    self.std[c][col] = Xc[col].std()

    def gauss(self, mean, std, x):
        std = max(std, 1e-3)
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    def confidence_per_class(self, X):
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
        confs = self.confidence_per_class(X)
        preds = [max(p, key=p.get) for p in confs]
        return np.array(preds)

if __name__ == "__main__":
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    df = adult.frame
    df = df.dropna()

    X = df.drop("class", axis=1)
    y = df["class"].astype(str)

    num_cols = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    X[cat_cols] = X[cat_cols].astype("object")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = NaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("--- Resultados com Base Adult ---")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print("Matriz de confusão:\n", confusion_matrix(y_test, y_pred))

    confs = model.confidence_per_class(X_test)
    print("\n🔹 Confiança por classe:")
    for i, conf in enumerate(confs[:10]):
        conf_fmt = {str(k): f"{v:.4f}" for k, v in conf.items()}
        print(f"Amostra {i+1}: {conf_fmt}")
