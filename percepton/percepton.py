import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class Perceptron:
    """
    Implementação do Perceptron - Algoritmo de Aprendizado Supervisionado
    
    O Perceptron é um classificador binário linear que aprende através da
    regra de atualização: w = w + η * (y_real - y_pred) * x
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Inicializa o Perceptron
        
        Parâmetros:
        -----------
        learning_rate : float
            Taxa de aprendizado (η) - controla o tamanho dos ajustes
        n_iterations : int
            Número máximo de iterações (épocas) de treinamento
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.errors_history = []  # Histórico de erros por época
        
    def _activation(self, x):
        """
        Função de ativação: Step Function (degrau)
        Retorna 1 se x >= 0, caso contrário retorna 0
        """
        return np.where(x >= 0, 1, 0)
    
    def predict(self, X):
        """
        Faz predições para os dados X
        
        Fórmula: ŷ = activation(w·x + b)
        onde w são os pesos, x os inputs e b o bias
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self._activation(linear_output)
        return y_predicted
    
    def fit(self, X, y):
        """
        Treina o Perceptron usando os dados X e rótulos y
        
        Algoritmo:
        1. Inicializa pesos e bias com zeros
        2. Para cada época:
           - Para cada exemplo (xi, yi):
             * Calcula predição: ŷi = activation(w·xi + b)
             * Calcula erro: erro = yi - ŷi
             * Atualiza pesos: w = w + η * erro * xi
             * Atualiza bias: b = b + η * erro
        3. Repete até convergir ou atingir max iterações
        """
        n_samples, n_features = X.shape
        
        # Inicializa pesos e bias com zeros
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Treinamento
        for iteration in range(self.n_iterations):
            errors = 0
            
            # Para cada exemplo de treino
            for idx, x_i in enumerate(X):
                # Calcula predição
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self._activation(linear_output)
                
                # Calcula erro
                error = y[idx] - y_predicted
                
                # Atualiza pesos e bias (Regra do Perceptron)
                if error != 0:
                    self.weights += self.learning_rate * error * x_i
                    self.bias += self.learning_rate * error
                    errors += 1
            
            # Armazena número de erros nesta época
            self.errors_history.append(errors)
            
            # Mostra progresso a cada 100 épocas
            if (iteration + 1) % 100 == 0:
                print(f"Época {iteration + 1}/{self.n_iterations} - Erros: {errors}")
            
            # Se não houver erros, convergiu!
            if errors == 0:
                print(f"\n✓ Convergiu na iteração {iteration + 1}")
                break
        
        return self
    
    def score(self, X, y):
        """Calcula acurácia do modelo"""
        predictions = self.predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy


# =============================================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO DO DATASET STROKE
# =============================================================================

print("=" * 70)
print("PERCEPTRON - DATASET STROKE (Predição de AVC)")
print("=" * 70)

# Carrega o dataset
print("\n[1/6] Carregando dataset...")
try:
    df = pd.read_csv('percepton/healthcare-dataset-stroke-data.csv')
except FileNotFoundError:
    try:
        df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    except FileNotFoundError:
        print("❌ ERRO: Arquivo não encontrado!")
        print("Certifique-se de que o arquivo está em uma dessas locações:")
        print("  - percepton/healthcare-dataset-stroke-data.csv")
        print("  - healthcare-dataset-stroke-data.csv")
        exit()

print(f"✓ Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} features")
print(f"\nPrimeiras linhas:")
print(df.head())

print(f"\nInformações das colunas:")
print(df.info())

print(f"\nDistribuição da variável alvo (stroke):")
print(df['stroke'].value_counts())
print(f"Proporção: {df['stroke'].value_counts(normalize=True)}")

# =============================================================================
# PRÉ-PROCESSAMENTO
# =============================================================================

print("\n" + "=" * 70)
print("[2/6] Pré-processamento dos dados...")
print("=" * 70)

# Remove coluna 'id' (não é relevante)
df = df.drop('id', axis=1)

# Trata valores ausentes em 'bmi'
print(f"\nValores ausentes:")
for col in df.columns:
    missing = df[col].isna().sum()
    if missing > 0:
        print(f"  - {col}: {missing} ({missing/len(df)*100:.1f}%)")

# Substitui 'N/A' string por NaN
df['bmi'] = df['bmi'].replace('N/A', np.nan)
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

# Preenche valores ausentes com a mediana
df['bmi'].fillna(df['bmi'].median(), inplace=True)
print("✓ Valores ausentes em 'bmi' preenchidos com a mediana")

# Remove linhas com 'Unknown' em gender
df = df[df['gender'] != 'Other']
print(f"✓ Removidas amostras com gender='Other'")

# Codifica variáveis categóricas
label_encoders = {}
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

print(f"\nCodificando variáveis categóricas...")
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"  ✓ {col}: {list(le.classes_)}")

print(f"\nDataset após pré-processamento: {df.shape}")

# Separa features (X) e target (y)
X = df.drop('stroke', axis=1).values
y = df['stroke'].values

print(f"\nShape X: {X.shape}")
print(f"Shape y: {y.shape}")

# =============================================================================
# DIVISÃO TREINO/TESTE
# =============================================================================

print("\n" + "=" * 70)
print("[3/6] Dividindo em treino e teste...")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")
print(f"\nDistribuição no treino: {np.bincount(y_train)}")
print(f"Distribuição no teste: {np.bincount(y_test)}")

# =============================================================================
# NORMALIZAÇÃO
# =============================================================================

print("\n" + "=" * 70)
print("[4/6] Normalizando features...")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features normalizadas (média=0, desvio padrão=1)")

# =============================================================================
# TREINAMENTO DO PERCEPTRON
# =============================================================================

print("\n" + "=" * 70)
print("[5/6] Treinando Perceptron...")
print("=" * 70)

perceptron = Perceptron(learning_rate=0.001, n_iterations=1000)
perceptron.fit(X_train_scaled, y_train)

# =============================================================================
# AVALIAÇÃO
# =============================================================================

print("\n" + "=" * 70)
print("[6/6] Avaliando modelo...")
print("=" * 70)

# Predições
y_train_pred = perceptron.predict(X_train_scaled)
y_test_pred = perceptron.predict(X_test_scaled)

# Acurácia
train_accuracy = perceptron.score(X_train_scaled, y_train)
test_accuracy = perceptron.score(X_test_scaled, y_test)

print(f"\n📊 RESULTADOS:")
print(f"  Acurácia no TREINO: {train_accuracy:.2%}")
print(f"  Acurácia no TESTE:  {test_accuracy:.2%}")

# Matriz de confusão
from sklearn.metrics import confusion_matrix, classification_report

print(f"\n📋 MATRIZ DE CONFUSÃO (Teste):")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"\n  VP (Verdadeiro Positivo): {cm[1,1]}")
print(f"  VN (Verdadeiro Negativo): {cm[0,0]}")
print(f"  FP (Falso Positivo): {cm[0,1]}")
print(f"  FN (Falso Negativo): {cm[1,0]}")

print(f"\n📈 RELATÓRIO DE CLASSIFICAÇÃO:")
print(classification_report(y_test, y_test_pred, target_names=['Sem Stroke', 'Com Stroke']))

# =============================================================================
# VISUALIZAÇÕES
# =============================================================================

print("\n" + "=" * 70)
print("Gerando visualizações...")
print("=" * 70)

fig = plt.figure(figsize=(15, 10))

# 1. Evolução dos erros
plt.subplot(2, 3, 1)
plt.plot(perceptron.errors_history, linewidth=2)
plt.xlabel('Época', fontsize=12)
plt.ylabel('Número de Erros', fontsize=12)
plt.title('Convergência do Perceptron', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 2. Distribuição do dataset
plt.subplot(2, 3, 2)
labels = ['Sem Stroke', 'Com Stroke']
counts = [np.sum(y == 0), np.sum(y == 1)]
colors = ['lightblue', 'salmon']
plt.bar(labels, counts, color=colors, edgecolor='black')
plt.ylabel('Quantidade', fontsize=12)
plt.title('Distribuição do Dataset', fontsize=14, fontweight='bold')
for i, v in enumerate(counts):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')

# 3. Matriz de confusão
plt.subplot(2, 3, 3)
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Sem Stroke', 'Com Stroke'], rotation=45)
plt.yticks(tick_marks, ['Sem Stroke', 'Com Stroke'])

# Adiciona valores na matriz
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=16, fontweight='bold')

plt.ylabel('Real', fontsize=12)
plt.xlabel('Predito', fontsize=12)
plt.tight_layout()

# 4. Importância das features (baseado nos pesos)
plt.subplot(2, 3, 4)
feature_names = df.drop('stroke', axis=1).columns
weights = perceptron.weights
sorted_idx = np.argsort(np.abs(weights))[::-1][:10]  # Top 10

plt.barh(range(len(sorted_idx)), np.abs(weights[sorted_idx]))
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Magnitude do Peso', fontsize=12)
plt.title('Top 10 Features Mais Importantes', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

# 5. Predições vs Real (amostra)
plt.subplot(2, 3, 5)
sample_size = 50
sample_idx = np.random.choice(len(y_test), sample_size, replace=False)
x_pos = np.arange(sample_size)

plt.scatter(x_pos, y_test[sample_idx], c='blue', marker='o', s=100, alpha=0.6, label='Real')
plt.scatter(x_pos, y_test_pred[sample_idx], c='red', marker='x', s=100, alpha=0.6, label='Predito')
plt.xlabel('Amostra', fontsize=12)
plt.ylabel('Classe (0=Sem Stroke, 1=Com Stroke)', fontsize=12)
plt.title(f'Predições vs Real ({sample_size} amostras)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Métricas
plt.subplot(2, 3, 6)
metrics = ['Acurácia\nTreino', 'Acurácia\nTeste']
values = [train_accuracy * 100, test_accuracy * 100]
colors_metrics = ['lightgreen', 'lightcoral']

bars = plt.bar(metrics, values, color=colors_metrics, edgecolor='black', linewidth=2)
plt.ylabel('Porcentagem (%)', fontsize=12)
plt.title('Desempenho do Modelo', fontsize=14, fontweight='bold')
plt.ylim(0, 100)

for bar, value in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.1f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n✓ Visualizações geradas!")

# =============================================================================
# INFORMAÇÕES FINAIS
# =============================================================================

print("\n" + "=" * 70)
print("INFORMAÇÕES DO MODELO FINAL")
print("=" * 70)
print(f"Taxa de aprendizado: {perceptron.learning_rate}")
print(f"Número de features: {len(perceptron.weights)}")
print(f"Épocas treinadas: {len(perceptron.errors_history)}")
print(f"Bias aprendido: {perceptron.bias:.4f}")
print(f"\nTop 5 pesos (magnitude):")
sorted_idx = np.argsort(np.abs(perceptron.weights))[::-1][:5]
for i, idx in enumerate(sorted_idx, 1):
    print(f"  {i}. {feature_names[idx]}: {perceptron.weights[idx]:.4f}")

print("\n" + "=" * 70)
print("CONCLUSÃO")
print("=" * 70)
print("""
O Perceptron é um algoritmo simples que funciona apenas para dados
LINEARMENTE SEPARÁVEIS. Para o dataset de stroke, que é complexo e 
desbalanceado, o Perceptron pode ter limitações.

Para melhor desempenho, considere:
  • Balanceamento de classes (SMOTE, undersampling)
  • Algoritmos mais complexos (MLP, Random Forest, XGBoost)
  • Feature engineering
  • Ensemble methods
""")