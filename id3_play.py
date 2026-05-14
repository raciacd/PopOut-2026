import pandas as pd
import numpy as np
import random
from id3_engine import ID3DecisionTree

# --- ID3 AGENT: THE BOT THAT LEARNS FROM MCTS ---

class ID3Play:
    def __init__(self, name="ID3-Bot", csv_path='mcts_2min_vs_random.csv'):
        self.name = name
        # Initialize the tree with a depth limit to prevent overfitting
        self.tree = ID3DecisionTree(depth_limit=20)
        self.csv_path = csv_path
        # As soon as the bot is created, it 'studies' the CSV file
        self._prepare_model()

    def _prepare_model(self):
        """Loads data and runs validation tests to ensure the bot learned correctly."""
        print(f"\n[!] {self.name} is studying MCTS gameplay data...")
        try:
            # Load the 8,300 moves collected previously
            df = pd.read_csv(self.csv_path)
            X = df.iloc[:, 0:43].values  # Board state + player turn
            y = df.iloc[:, 43].values     # The move chosen by MCTS (target)
            
            # --- TEST 1: 70/30 TRAIN-TEST SPLIT ---
            # Shuffle indices so the bot doesn't get biased by data order
            indices = np.random.permutation(len(X))
            split_point = int(0.7 * len(X))
            
            # Use 70% to learn and 30% to test if it can predict correctly
            X_train, X_test = X[indices[:split_point]], X[indices[split_point:]]
            y_train, y_test = y[indices[:split_point]], y[indices[split_point:]]
            
            self.tree.fit(X_train, y_train)
            
            # Check accuracy on the 30% unseen data
            preds = self.tree.predict(X_test)
            accuracy = np.mean(preds == y_test) * 100
            print(f"-> 70/30 Holdout Accuracy: {accuracy:.2f}%")

            # --- TEST 2: 5-FOLD CROSS-VALIDATION ---
            # To ensure the result wasn't just luck, we test on 5 different data slices
            print("-> Running 5-Fold Cross-Validation for robustness...")
            k = 5
            fold_size = len(X) // k
            cv_scores = []
            
            for i in range(k):
                start, end = i * fold_size, (i + 1) * fold_size
                
                # Separate one slice for validation and the other 4 for training
                X_val, y_val = X[start:end], y[start:end]
                X_train_cv = np.concatenate([X[:start], X[end:]])
                y_train_cv = np.concatenate([y[:start], y[end:]])
                
                # Create a temporary tree for this specific fold test
                test_tree = ID3DecisionTree(depth_limit=15)
                test_tree.fit(X_train_cv, y_train_cv)
                
                # Save the score for this round
                score = np.mean(test_tree.predict(X_val) == y_val)
                cv_scores.append(score)
            
            # Display the final average of the 5 tests
            avg_cv = np.mean(cv_scores) * 100
            print(f"-> Final Cross-Validation Mean: {avg_cv:.2f}%")
            
        except Exception as e:
            print(f"Error reading CSV or training model: {e}")

    def get_move(self, state):
        """This function is called during the bot's turn to pick a move."""
        
        # 1. TRANSLATE BOARD: The game uses bits, but the tree needs simple numbers (-1, 0, 1)
        # Create a flat array of 42 cells (7x6 Connect4 board)
        simple_board = np.full(42, -1)
        
        # Extract bitboards for Player 0 and Player 1
        p0 = state.position if state.turn == 0 else (state.position ^ state.mask)
        p1 = state.position if state.turn == 1 else (state.position ^ state.mask)
        
        for i in range(42):
            # Map linear index 'i' back to the bitboard coordinate (column * 7 + row)
            bit_mask = 1 << ((i % 7) * 7 + (i // 7))
            if p0 & bit_mask: simple_board[i] = 0
            elif p1 & bit_mask: simple_board[i] = 1
        
        # Combine board state with 'whose turn it is' (turn)
        input_data = np.append(simple_board, state.turn)
        
        # 2. DECISION: Ask the tree to predict what move MCTS would make here
        predicted_move = int(self.tree.predict(np.array([input_data]))[0])
        
        # 3. SAFETY CHECK: If the tree suggests an illegal move (e.g. full column), pick a random one
        legal_moves = state.legal_moves()
        if predicted_move in legal_moves:
            return predicted_move
        else:
            # Backup strategy if the tree encounters a completely new situation
            return random.choice(legal_moves)
