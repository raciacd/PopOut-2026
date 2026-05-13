import numpy as np
from collections import Counter
import math

class DecisionNode:
    """This class represents a single 'piece' of our tree (either a node or a leaf)."""
    def __init__(self, left=None, right=None, feature_idx=None, threshold=None, class_label=None):
        self.left = left            # Left branch
        self.right = right          # Right branch
        self.feature_idx = feature_idx  # Which board cell we are checking
        self.threshold = threshold      # The value used to split the data
        self.class_label = class_label  # The final move prediction (only for leaves)

    def is_leaf(self):
        # If it has a class_label, we reached the end of a branch
        return self.class_label is not None

class ID3DecisionTree:
    def __init__(self, depth_limit=float('inf')):
        self.root = None
        self.depth_limit = depth_limit # Prevents the tree from growing forever

    def _entropy(self, y):
        """Measures the 'disorder' of the data. Higher entropy means moves are very mixed."""
        if len(y) == 0: return 0
        counts = Counter(y)
        probs = [counts[c] / len(y) for c in counts]
        # Entropy formula: helps determine if a group of moves is already 'pure'
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _information_gain(self, y, y_left, y_right):
        """Calculates how much we 'gain' by splitting the data this way."""
        parent_ent = self._entropy(y)
        n = len(y)
        n_l, n_r = len(y_left), len(y_right)
        if n_l == 0 or n_r == 0: return 0
        
        # Check if this split actually reduced the 'disorder'
        child_ent = (n_l / n) * self._entropy(y_left) + (n_r / n) * self._entropy(y_right)
        return parent_ent - child_ent

    def _best_split(self, X, y):
        """Iterates through all board positions to find the best 'question' to ask."""
        best_gain = -1
        split_idx, split_thresh = None, None

        # Check every column/feature in the dataset
        for idx in range(X.shape[1]):
            column_values = X[:, idx]
            unique_values = np.unique(column_values)
            # Create split points to decide where to separate the data
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for t in thresholds:
                left_mask = column_values <= t
                y_l, y_r = y[left_mask], y[~left_mask]
                
                # Evaluate if this specific split gives us good information gain
                gain = self._information_gain(y, y_l, y_r)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = idx
                    split_thresh = t
        
        return split_idx, split_thresh

    def fit(self, X, y):
        """Starts building the tree using the training data."""
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        """Main recursive function that builds the tree level by level."""
        # Base case: If all moves are the same, create a leaf with that move
        if len(set(y)) == 1:
            return DecisionNode(class_label=list(y)[0])
        
        # Stop if we hit the depth limit or run out of data
        if depth >= self.depth_limit or len(y) <= 1:
            most_common = Counter(y).most_common(1)[0][0]
            return DecisionNode(class_label=most_common)

        # Find the best split (attribute and threshold) for this level
        idx, thresh = self._best_split(X, y)
        
        # If no further improvement is possible, stop here
        if idx is None:
            return DecisionNode(class_label=Counter(y).most_common(1)[0][0])

        # Create the left and right branches by repeating the process (Recursion)
        left_mask = X[:, idx] <= thresh
        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return DecisionNode(left, right, idx, thresh)

    def predict(self, X):
        """Takes new board states and tries to predict the best move."""
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        """Walks down the tree until reaching a leaf (a decision)."""
        if node.is_leaf():
            return node.class_label
        
        # Go left or right based on the feature value
        if x[node.feature_idx] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
