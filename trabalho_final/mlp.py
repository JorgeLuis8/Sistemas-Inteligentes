# Instalar se faltar:
# pip install ucimlrepo pandas scikit-learn imbalanced-learn

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# ============================================================
# 1) CARREGAR DATASET DA UCI
# ============================================================

data = fetch_ucirepo(id=451)  # Breast Cancer Coimbra
X = data.data.features
y = data.data.targets

df = pd.DataFrame(X, columns=data.data.feature_names)
df["Classification"] = y

print("\nContagem das classes:")
print(df["Classification"].value_counts(), "\n")

X = df.drop("Classification", axis=1)
y = df["Classification"]

# ============================================================
# 2) DEFINIR MLP "CAMPEÃ" — MELHORES PARÂMETROS ACHADOS
# ============================================================

mlp_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="tanh",
        learning_rate="constant",
        alpha=0.0001,
        max_iter=3000,
        random_state=42
    ))
])

# ============================================================
# 3) VALIDATION: 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_acc = cross_val_score(mlp_pipeline, X, y, cv=cv, scoring="accuracy")
scores_f1  = cross_val_score(mlp_pipeline, X, y, cv=cv, scoring="f1_macro")

print("===== VALIDAÇÃO CRUZADA (5-fold) =====")
print("Acurácias:", scores_acc)
print("Acurácia média:", scores_acc.mean())
print("\nF1_macro:", scores_f1)
print("F1_macro médio:", scores_f1.mean())
print("=======================================\n")

# ============================================================
# 4) TESTE FINAL — TREINO/TESTE + SMOTE NO TREINO
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Balancear SOMENTE o treino
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Treinar modelo final
mlp_pipeline.fit(X_train_res, y_train_res)

# Predição no teste
y_pred = mlp_pipeline.predict(X_test)

print("===== AVALIAÇÃO NO CONJUNTO DE TESTE =====")
print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))
print("===========================================")
