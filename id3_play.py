import pandas as pd
import numpy as np
import random
from id3_engine import ID3DecisionTree
from id3_train_test import custom_train_test_split 

class ID3Play:
    """
    Wrapper class that utilizes the id3 engine to return a move based on the training data.
    
        - loads a dataset of MCTS games
        - filters for winning moves
        - trains a decision tree and uses it to predict optimal moves

    Attributes:
        name (str): The display name of the agent.
        csv_path (str): Path to the dataset used for training.
        tree (ID3DecisionTree): The decision tree model.
    """
    def __init__(self, name="ID3-Bot", csv_path='mcts_2sec_vs_random_10k.csv'):
        """
        Initializes the instance and starts the learning phase.

        Args:
            name (str): The name of the agent, ID3-Bot.
            csv_path (str): The dataset filename. Defaults to 'mcts_2sec_vs_random_10k.csv'.
        """
        self.name = name
        # Initialize the tree with a depth limit to prevent overfitting
        self.tree = ID3DecisionTree(depth_limit=20)
        self.csv_path = csv_path
        self._prepare_model()

    def _prepare_model(self):
        """
        Loads data, filters for optimal plays, and runs validation tests.
        """
        print(f"\n[!] {self.name} learning MCTS gameplay data...")
        try:
            df = pd.read_csv(self.csv_path)
            
            filtro_vitorias = (
                ((df['player_turn'] == 0) & (df['game_result'] == 1)) |
                ((df['player_turn'] == 1) & (df['game_result'] == -1))
            )
            df_limpo = df[filtro_vitorias]
            
            X = df_limpo.iloc[:, 0:43].values  # Board state + player turn
            y = df_limpo.iloc[:, 43].values    # The move chosen by MCTS (target)
            
            X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.3)
            
            self.tree.fit(X_train, y_train)
            
            # Check accuracy on unseen data
            preds = self.tree.predict(X_test)
            accuracy = np.mean(preds == y_test) * 100
            print(f"70/30 Holdout Accuracy: {accuracy:.2f}%")

            # 5-FOLD CROSS-VALIDATION
            print("Running 5-Fold Cross-Validation for robustness...")
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
            print(f"Final Cross-Validation Mean: {avg_cv:.2f}%")
            
        except Exception as e:
            print(f"Error reading CSV or training model: {e}")

    def get_move(self, state):
        """
        Translates the bitboard state and asks the tree for a prediction.

        Args:
            state (Position): The current state of the game board.

        Returns:
            int: The index of the predicted move.
        """
        # The game uses bits, but the tree needs simple numbers (-1, 0, 1)
        simple_board = np.full(42, -1)
        
        p0 = state.position if state.turn == 0 else (state.position ^ state.mask)
        p1 = state.position if state.turn == 1 else (state.position ^ state.mask)
        
        for i in range(42):
            bit_mask = 1 << ((i % 7) * 7 + (i // 7))
            if p0 & bit_mask: simple_board[i] = 0
            elif p1 & bit_mask: simple_board[i] = 1
        
        input_data = np.append(simple_board, state.turn)
        
        predicted_move = int(self.tree.predict(np.array([input_data]))[0])
        
        legal_moves = state.legal_moves()
        if predicted_move in legal_moves:
            return predicted_move
        else:
            return random.choice(legal_moves)