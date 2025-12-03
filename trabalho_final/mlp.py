import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay, classification_report
)

# --- CARREGAMENTO ---
print("Baixando dados...")
dataset = fetch_ucirepo(id=451)
X = dataset.data.features
y = dataset.data.targets['Classification'].replace({1: 0, 2: 1}) # 0: Não tem, 1: Tem

# ==============================================================================
# PASSO 1: HOLD-OUT (Divisão de Treino e Teste)
# ==============================================================================
# O Hold-out separa uma fatia dos dados que o modelo NUNCA vai ver durante o treino.
# Usamos 20% para teste (test_size=0.2) e 80% para treino.
# stratify=y: Garante que a proporção de doentes seja a mesma no treino e no teste.
print("Aplicando Hold-out...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=0, 
    stratify=y
)

print(f" -> Tamanho do Treino: {len(X_train)} amostras")
print(f" -> Tamanho do Teste (Hold-out): {len(X_test)} amostras")

# ==============================================================================
# PASSO 2: NORMALIZAÇÃO (Padronização)
# ==============================================================================
# Redes Neurais não funcionam bem com números de escalas diferentes (ex: Glicose 100 vs Insulina 5).
# O StandardScaler transforma tudo para ter média 0 e desvio padrão 1.
print("Normalizando os dados...")

scaler = StandardScaler()

# ATENÇÃO: O .fit (aprender a média) é feito SÓ no treino para evitar vazamento de dados!
X_train_scaled = scaler.fit_transform(X_train) 

# No teste, nós apenas aplicamos (.transform) a régua que aprendemos no treino
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# TREINAMENTO E AVALIAÇÃO
# ==============================================================================
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50), activation='tanh', solver='lbfgs',
    alpha=0.5, max_iter=5000, random_state=42
)

print("Treinando MLP...")
mlp.fit(X_train_scaled, y_train)

# Testando na parte que separamos (Hold-out)
y_pred = mlp.predict(X_test_scaled)

# ==============================================================================
# MÉTRICAS DE AVALIAÇÃO
# ==============================================================================
print("\n" + "="*70)
print("MÉTRICAS DE DESEMPENHO NO CONJUNTO DE TESTE (HOLD-OUT)")
print("="*70)

# Métricas básicas
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Acurácia (Accuracy):            {acc:.4f} ({acc*100:.2f}%)")
print(f"Precisão (Precision):           {precision:.4f} ({precision*100:.2f}%)")
print(f"Revocação (Recall/Sensitivity): {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:                       {f1:.4f}")
print("="*70)

print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_test, y_pred, target_names=['Não tem Câncer', 'Tem Câncer']))

# Gráfico - Matriz de Confusão
fig, ax = plt.subplots(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, 
    display_labels=['Não tem Câncer', 'Tem Câncer'], 
    cmap='Blues', ax=ax, colorbar=False
)
ax.set_title("Matriz de Confusão - Hold-out", fontsize=12, fontweight='bold')
ax.grid(False)

plt.tight_layout()
plt.show()