import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, KFold, StratifiedKFold, LeaveOneOut, cross_val_score
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, f1_score
from imblearn.over_sampling import SMOTE

# ==========================================================
# 1) Carregar e preparar o dataset
# ==========================================================
df = pd.read_csv("Classificacao/healthcare-dataset-stroke-data.csv")
df = df.drop(columns=["id"], errors="ignore").dropna()

# Codificar variáveis categóricas
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df.drop(columns=["stroke"])
y = df["stroke"]

# Padronizar
X = StandardScaler().fit_transform(X)

# ==========================================================
# 2) Balancear classes com SMOTE
# ==========================================================
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"\n✅ Após SMOTE: {np.bincount(y_res)} (classes balanceadas)")

# ==========================================================
# 3) Dividir em treino/val/teste (70/15/15)
# ==========================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X_res, y_res, test_size=0.3, stratify=y_res, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
print(f"\n📊 Divisão: Treino={len(X_train)}, Validação={len(X_val)}, Teste={len(X_test)}")

# ==========================================================
# 4) Modelo
# ==========================================================
model = KNeighborsClassifier(n_neighbors=7)

# ==========================================================
# 5) Função para avaliar com CV (médias e dps)
# ==========================================================
def avaliar(model, X, y, cv):
    acc  = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    prec = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(precision_score, zero_division=0))
    f1   = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(f1_score, zero_division=0))
    return {
        "acc_mean":  acc.mean(),  "acc_std":  acc.std(),
        "prec_mean": prec.mean(), "prec_std": prec.std(),
        "f1_mean":   f1.mean(),   "f1_std":   f1.std()
    }

# ==========================================================
# 6) Técnicas
# ==========================================================
# Hold-Out (usa apenas o conjunto de TESTE para as métricas)
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
acc_hold  = accuracy_score (y_test, y_pred_test)
prec_hold = precision_score(y_test, y_pred_test, zero_division=0)
f1_hold   = f1_score      (y_test, y_pred_test, zero_division=0)

# K-Fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
res_kf = avaliar(model, X_res, y_res, kfold)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res_skf = avaliar(model, X_res, y_res, skf)

# Leave-One-Out (subset para tempo)
loo = LeaveOneOut()
res_loo = avaliar(model, X_res[:500], y_res[:500], loo)

# ==========================================================
# 7) Tabela comparativa
# ==========================================================
tabela = pd.DataFrame([
    {
        "Técnica": "Hold-Out",
        "Acurácia (média)": acc_hold,    "Acurácia (dp)": 0.0,
        "Precisão (média)": prec_hold,   "Precisão (dp)": 0.0,
        "F1 (média)":       f1_hold,     "F1 (dp)":       0.0
    },
    {
        "Técnica": "K-Fold",
        "Acurácia (média)": res_kf["acc_mean"],   "Acurácia (dp)": res_kf["acc_std"],
        "Precisão (média)": res_kf["prec_mean"],  "Precisão (dp)": res_kf["prec_std"],
        "F1 (média)":       res_kf["f1_mean"],    "F1 (dp)":       res_kf["f1_std"]
    },
    {
        "Técnica": "Stratified K-Fold",
        "Acurácia (média)": res_skf["acc_mean"],  "Acurácia (dp)": res_skf["acc_std"],
        "Precisão (média)": res_skf["prec_mean"], "Precisão (dp)": res_skf["prec_std"],
        "F1 (média)":       res_skf["f1_mean"],   "F1 (dp)":       res_skf["f1_std"]
    },
    {
        "Técnica": "Leave-One-Out (500 amostras)",
        "Acurácia (média)": res_loo["acc_mean"],  "Acurácia (dp)": res_loo["acc_std"],
        "Precisão (média)": res_loo["prec_mean"], "Precisão (dp)": res_loo["prec_std"],
        "F1 (média)":       res_loo["f1_mean"],   "F1 (dp)":       res_loo["f1_std"]
    }
])

# Exibir
pd.set_option("display.max_columns", None)
print("\n📊 TABELA COMPARATIVA (Acurácia, Precisão e F1 — médias e dps)")
print("----------------------------------------------------------------")
print(tabela.to_string(index=False))
