import numpy as np
from collections import Counter
import math

class DecisionNode:
    """Esta classe serve apenas para criar as 'peças' da nossa árvore (nós ou folhas)."""
    def __init__(self, left=None, right=None, feature_idx=None, threshold=None, class_label=None):
        self.left = left            # Lado esquerdo da árvore
        self.right = right          # Lado direito da árvore
        self.feature_idx = feature_idx  # Qual coluna do tabuleiro estamos a olhar
        self.threshold = threshold      # O valor que decide se vai para a esquerda ou direita
        self.class_label = class_label  # Se for uma folha, guarda aqui a jogada final

    def is_leaf(self):
        # Se tem um class_label, é porque chegámos ao fim de um ramo
        return self.class_label is not None

class ID3DecisionTree:
    def __init__(self, depth_limit=float('inf')):
        self.root = None
        self.depth_limit = depth_limit # Para a árvore não crescer até ao infinito

    def _entropy(self, y):
        """Mede a 'confusão' dos dados. Quanto mais misturadas as jogadas, maior a entropia."""
        if len(y) == 0: return 0
        counts = Counter(y)
        probs = [counts[c] / len(y) for c in counts]
        # Fórmula da entropia: ajuda a perceber se o grupo de jogadas já é puro ou não
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _information_gain(self, y, y_left, y_right):
        """Calcula quanto é que ganhamos em organizar os dados desta maneira."""
        parent_ent = self._entropy(y)
        n = len(y)
        n_l, n_r = len(y_left), len(y_right)
        if n_l == 0 or n_r == 0: return 0
        
        # Vê se a separação ajudou a diminuir a 'confusão' (entropia)
        child_ent = (n_l / n) * self._entropy(y_left) + (n_r / n) * self._entropy(y_right)
        return parent_ent - child_ent

    def _best_split(self, X, y):
        """Aqui a árvore testa todas as jogadas e posições para ver qual a melhor pergunta a fazer."""
        best_gain = -1
        split_idx, split_thresh = None, None

        # Corre todas as colunas (atributos)
        for idx in range(X.shape[1]):
            column_values = X[:, idx]
            unique_values = np.unique(column_values)
            # Cria pontos de corte entre os valores para decidir onde separar
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for t in thresholds:
                left_mask = column_values <= t
                y_l, y_r = y[left_mask], y[~left_mask]
                
                # Vê se este corte dá um bom ganho de informação
                gain = self._information_gain(y, y_l, y_r)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_thresh = t
        
        return split_idx, split_thresh

    def fit(self, X, y):
        """Função para começar a construir a árvore com os dados de treino."""
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """Função principal que vai montando a árvore por níveis."""
        # Se todas as jogadas forem iguais, criamos uma folha com essa jogada
        if len(set(y)) == 1:
            return DecisionNode(class_label=list(y)[0])
        
        # Se atingirmos o limite ou não houver mais dados, escolhemos a jogada mais comum
        if depth >= self.depth_limit or len(y) <= 1:
            most_common = Counter(y).most_common(1)[0][0]
            return DecisionNode(class_label=most_common)

        # Procura a melhor pergunta (atributo e corte) para este nível
        idx, thresh = self._best_split(X, y)
        
        # Se não houver forma de melhorar a separação, paramos aqui
        if idx is None:
            return DecisionNode(class_label=Counter(y).most_common(1)[0][0])

        # Cria os caminhos da esquerda e da direita repetindo o processo (Recursividade)
        left_mask = X[:, idx] <= thresh
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return DecisionNode(left, right, idx, thresh)

    def predict(self, X):
        """Recebe novos estados do tabuleiro e tenta adivinhar a jogada."""
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        """Caminha pela árvore até chegar a uma folha (uma decisão)."""
        if node.is_leaf():
            return node.class_label
        
        # Segue o caminho baseado no valor do atributo
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def show(self, node=None, depth=0, feature_names=None):
        """Desenha a árvore no terminal para podermos ver as regras que ela criou."""
        if node is None: node = self.root
        
        indent = "  " * depth
        if node.is_leaf():
            print(f"{indent}JOGADA FINAL: {node.class_label}")
            return

        feat = feature_names[node.feature_idx] if feature_names else f"Coluna {node.feature_idx}"
        print(f"{indent}[Se {feat} <= {node.threshold:.2f}]")
        self.show(node.left, depth + 1, feature_names)
        
        print(f"{indent}[Se {feat} > {node.threshold:.2f}]")
        self.show(node.right, depth + 1, feature_names)
