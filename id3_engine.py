import numpy as np
from collections import Counter
import math

class DecisionNode:
    """Representa um nó ou folha da árvore."""
    def __init__(self, left=None, right=None, feature_idx=None, threshold=None, class_label=None):
        self.left = left
        self.right = right
        self.feature_idx = feature_idx  # Índice do atributo (ex: 0 para sepal length)
        self.threshold = threshold      # Valor de corte (v)
        self.class_label = class_label  # Previsão (apenas para folhas)

    def is_leaf(self):
        return self.class_label is not None

class ID3DecisionTree:
    def __init__(self, depth_limit=float('inf')):
        self.root = None
        self.depth_limit = depth_limit

    def _entropy(self, y):
        """Calcula a Entropia de um conjunto de etiquetas."""
        if len(y) == 0: return 0
        counts = Counter(y)
        probs = [counts[c] / len(y) for c in counts]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _information_gain(self, y, y_left, y_right):
        """Calcula o ganho de informação de um split."""
        parent_ent = self._entropy(y)
        n = len(y)
        n_l, n_r = len(y_left), len(y_right)
        if n_l == 0 or n_r == 0: return 0
        
        child_ent = (n_l / n) * self._entropy(y_left) + (n_r / n) * self._entropy(y_right)
        return parent_ent - child_ent

    def _best_split(self, X, y):
        """Encontra o melhor atributo e o melhor ponto de corte (Discretização)."""
        best_gain = -1
        split_idx, split_thresh = None, None

        for idx in range(X.shape[1]):
            column_values = X[:, idx]
            # Ordenar valores únicos para testar pontos médios (Discretização otimizada)
            unique_values = np.unique(column_values)
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for t in thresholds:
                left_mask = column_values <= t
                y_l, y_r = y[left_mask], y[~left_mask]
                
                gain = self._information_gain(y, y_l, y_r)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_thresh = t
        
        return split_idx, split_thresh

    def fit(self, X, y):
        """Treina a árvore."""
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # Casos base
        if len(set(y)) == 1:
            return DecisionNode(class_label=list(y)[0])
        
        if depth >= self.depth_limit or len(y) <= 1:
            most_common = Counter(y).most_common(1)[0][0]
            return DecisionNode(class_label=most_common)

        # Encontrar o melhor split (Discretização numérica)
        idx, thresh = self._best_split(X, y)
        
        if idx is None: # Se não houver ganho possível
            return DecisionNode(class_label=Counter(y).most_common(1)[0][0])

        # Criar ramos
        left_mask = X[:, idx] <= thresh
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return DecisionNode(left, right, idx, thresh)

    def predict(self, X):
        """Classifica novos exemplos."""
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.class_label
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def show(self, node=None, depth=0, feature_names=None):
        """Apresenta a árvore visualmente."""
        if node is None: node = self.root
        
        indent = "  " * depth
        if node.is_leaf():
            print(f"{indent}PREVISÃO: {node.class_label}")
            return

        feat = feature_names[node.feature_idx] if feature_names else f"Attr {node.feature_idx}"
        print(f"{indent}[{feat} <= {node.threshold:.2f}]")
        self.show(node.left, depth + 1, feature_names)
        
        print(f"{indent}[{feat} > {node.threshold:.2f}]")
        self.show(node.right, depth + 1, feature_names)

# --- Exemplo de Uso com Carregamento Manual (CSV Iris) ---

def load_iris(file_path):
    # Carregamento simples sem pandas (usando apenas numpy/csv)
    data = np.genfromtxt(file_path, delimiter=',', dtype=None, encoding='utf-8')
    # Assumindo que a última coluna é a classe e as primeiras são atributos
    X = np.array([list(row)[:-1] for row in data], dtype=float)
    y = np.array([row[-1] for row in data])
    return X, y

# Exemplo de fluxo:
# X_train, y_train = load_iris('iris.data')
# tree = ID3DecisionTree(depth_limit=5)
# tree.fit(X_train, y_train)
# tree.show(feature_names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid'])