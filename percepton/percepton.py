import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

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
            
            # Se não houver erros, convergiu!
            if errors == 0:
                print(f"Convergiu na iteração {iteration + 1}")
                break
        
        return self
    
    def score(self, X, y):
        """Calcula acurácia do modelo"""
        predictions = self.predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy


# =============================================================================
# EXEMPLO DE USO 1: Dataset Simples (AND Lógico)
# =============================================================================

print("=" * 60)
print("EXEMPLO 1: Porta Lógica AND")
print("=" * 60)

# Dataset: Porta AND
X_and = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])

y_and = np.array([0, 0, 0, 1])  # AND lógico

# Treina Perceptron
perceptron_and = Perceptron(learning_rate=0.1, n_iterations=10)
perceptron_and.fit(X_and, y_and)

# Testa
print("\nPredições para porta AND:")
for i, x in enumerate(X_and):
    pred = perceptron_and.predict(x.reshape(1, -1))[0]
    print(f"Input: {x} → Predição: {pred} (Real: {y_and[i]})")

print(f"\nPesos finais: {perceptron_and.weights}")
print(f"Bias final: {perceptron_and.bias}")


# =============================================================================
# EXEMPLO DE USO 2: Dataset com sklearn
# =============================================================================

print("\n" + "=" * 60)
print("EXEMPLO 2: Dataset de Classificação Binária")
print("=" * 60)

# Gera dataset sintético linearmente separável
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    flip_y=0.1,
    random_state=42
)

# Converte rótulos para 0 e 1
y = np.where(y == -1, 0, y)

# Divide em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Treina Perceptron
perceptron = Perceptron(learning_rate=0.01, n_iterations=100)
perceptron.fit(X_train, y_train)

# Avalia
train_accuracy = perceptron.score(X_train, y_train)
test_accuracy = perceptron.score(X_test, y_test)

print(f"\nAcurácia no treino: {train_accuracy:.2%}")
print(f"Acurácia no teste: {test_accuracy:.2%}")


# =============================================================================
# VISUALIZAÇÃO
# =============================================================================

# Plota evolução do erro
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(perceptron.errors_history, marker='o')
plt.xlabel('Época')
plt.ylabel('Número de Erros')
plt.title('Convergência do Perceptron')
plt.grid(True)

# Plota dados de treino
plt.subplot(1, 3, 2)
colors = ['red' if label == 0 else 'blue' for label in y_train]
plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, alpha=0.6)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Dados de Treino')
plt.legend(['Classe 0', 'Classe 1'])

# Plota fronteira de decisão
plt.subplot(1, 3, 3)
colors = ['red' if label == 0 else 'blue' for label in y_train]
plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, alpha=0.6)

# Calcula fronteira de decisão: w1*x1 + w2*x2 + b = 0
# x2 = -(w1*x1 + b) / w2
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
x_vals = np.array([x_min, x_max])
y_vals = -(perceptron.weights[0] * x_vals + perceptron.bias) / perceptron.weights[1]
plt.plot(x_vals, y_vals, 'g--', linewidth=2, label='Fronteira de Decisão')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Fronteira de Decisão do Perceptron')
plt.legend()

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("INFORMAÇÕES DO MODELO")
print("=" * 60)
print(f"Pesos aprendidos: {perceptron.weights}")
print(f"Bias aprendido: {perceptron.bias}")
print(f"Taxa de aprendizado: {perceptron.learning_rate}")
print(f"Convergiu em {len(perceptron.errors_history)} épocas")