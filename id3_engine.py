import numpy as np
from collections import Counter
import math

class DecisionNode:
    """
    Represents a single node or leaf in the decision tree.
    """
    def __init__(self, left=None, right=None, feature_idx=None, threshold=None, class_label=None):
        """
        Initializes a DecisionNode.

        Args:
            left (DecisionNode): The left branch (child node). Defaults to None.
            right (DecisionNode): The right branch (child node). Defaults to None.
            feature_idx (int): The index of the feature (board cell) used for splitting. Defaults to None.
            threshold (float): The value used to split the data. Defaults to None.
            class_label (int): The final move prediction (only for leaves). Defaults to None.
        """
        self.left = left            
        self.right = right          
        self.feature_idx = feature_idx  
        self.threshold = threshold      
        self.class_label = class_label  

    def is_leaf(self):
        """
        Checks if the current node is a leaf node (endpoint).

        Returns:
            bool: True if the node has a final class prediction, False otherwise.
        """
        return self.class_label is not None

class ID3DecisionTree:
    """
    Implementation of the ID3 Decision Tree algorithm with numerical discretization support.
    
    Attributes:
        depth_limit (int/float): The maximum allowed depth for the tree to prevent overfitting.
        root (DecisionNode): The root node of the trained tree.
    """
    def __init__(self, depth_limit=float('inf')):
        """
        Initializes the ID3DecisionTree instance.

        Args:
            depth_limit (int or float): Maximum depth limit. Defaults to infinity.
        """
        self.root = None
        self.depth_limit = depth_limit 

    def _entropy(self, y):
        """
        Calculates the entropy of a given target array. 
        Higher entropy means the target classes are very mixed.

        Args:
            y (numpy.ndarray): Array of target labels.

        Returns:
            float: The calculated entropy value.
        """
        if len(y) == 0: return 0
        counts = Counter(y)
        probs = [counts[c] / len(y) for c in counts]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _information_gain(self, y, y_left, y_right):
        """
        Calculates how much information is gained by a specific split.

        Args:
            y (numpy.ndarray): Original array of labels before the split.
            y_left (numpy.ndarray): Array of labels for the left split.
            y_right (numpy.ndarray): Array of labels for the right split.

        Returns:
            float: The calculated information gain.
        """
        parent_ent = self._entropy(y)
        n = len(y)
        n_l, n_r = len(y_left), len(y_right)
        if n_l == 0 or n_r == 0: return 0
        
        child_ent = (n_l / n) * self._entropy(y_left) + (n_r / n) * self._entropy(y_right)
        return parent_ent - child_ent

    def _best_split(self, X, y):
        """
        Iterates through all features to find the optimal split point (highest information gain).

        Args:
            X (numpy.ndarray): The feature matrix.
            y (numpy.ndarray): The target labels array.

        Returns:
            tuple: A tuple containing (best_feature_index, best_threshold). Returns (None, None) if no valid split.
        """
        best_gain = -1
        split_idx, split_thresh = None, None

        for idx in range(X.shape[1]):
            column_values = X[:, idx]
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
        """
        Starts the tree building process using the provided training data.

        Args:
            X (numpy.ndarray): The training feature matrix.
            y (numpy.ndarray): The training target labels array.
        """
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """
        Recursively builds the decision tree level by level.

        Args:
            X (numpy.ndarray): The feature matrix for the current node.
            y (numpy.ndarray): The target labels array for the current node.
            depth (int): The current depth in the tree. Defaults to 0.

        Returns:
            DecisionNode: The root node of the constructed (sub)tree.
        """
        if len(set(y)) == 1:
            return DecisionNode(class_label=list(y)[0])
        
        if depth >= self.depth_limit or len(y) <= 1:
            most_common = Counter(y).most_common(1)[0][0]
            return DecisionNode(class_label=most_common)

        idx, thresh = self._best_split(X, y)
        
        if idx is None:
            return DecisionNode(class_label=Counter(y).most_common(1)[0][0])

        left_mask = X[:, idx] <= thresh
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return DecisionNode(left, right, idx, thresh)

    def predict(self, X):
        """
        Predicts the best move for a given set of board states.

        Args:
            X (numpy.ndarray): The feature matrix to predict.

        Returns:
            numpy.ndarray: An array containing the predicted move for each state.
        """
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        """
        Walks down the tree based on feature values until it reaches a leaf.

        Args:
            x (numpy.ndarray): A single state (feature vector).
            node (DecisionNode): The current node in the traversal.

        Returns:
            int: The predicted class label (move).
        """
        if node.is_leaf():
            return node.class_label
        
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def show(self, node=None, spacing="", feature_names=None):
        """
        Displays the visual structure of the tree in the terminal.

        Args:
            node (DecisionNode): The starting node. Defaults to the root.
            spacing (str): The string used for indentation.
            feature_names (list): List of feature names for readability. Defaults to None.
        """
        if node is None:
            node = self.root

        if node.is_leaf():
            print(spacing + "  Prediction:", node.class_label)
            return

        feat = feature_names[node.feature_idx] if feature_names else f"Column {node.feature_idx}"
        
        print(f"{spacing}[{feat} <= {node.threshold:.2f}]")

        print(spacing + " ├── Yes:")
        self.show(node.left, spacing + " │   ", feature_names)

        print(spacing + " └── No:")
        self.show(node.right, spacing + "     ", feature_names)