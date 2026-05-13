from id3_engine import ID3DecisionTree
import numpy as np
import pandas as pd
import random

# --- NOVA CLASSE AGENTE ID3 COM VALIDAÇÃO CRUZADA ---

class ID3Play:
    def __init__(self, name="ID3-Bot", csv_path='mcts_2min_vs_random.csv'):
        self.name = name
        self.tree = ID3DecisionTree(depth_limit=20)
        self.csv_path = csv_path
        self._prepare_model()

    def _prepare_model(self):
        print(f"\n[!] A treinar e validar o {self.name}...")
        try:
            df = pd.read_csv(self.csv_path)
            X = df.iloc[:, 0:43].values
            y = df.iloc[:, 43].values
            
            # --- MELHORIA 1: 70% TREINO / 30% TESTE ---
            indices = np.random.permutation(len(X))
            train_idx = int(0.7 * len(X))
            X_train, X_test = X[indices[:train_idx]], X[indices[train_idx:]]
            y_train, y_test = y[indices[:train_idx]], y[indices[train_idx:]]
            
            self.tree.fit(X_train, y_train)
            preds = self.tree.predict(X_test)
            acc_holdout = np.mean(preds == y_test) * 100
            print(f"-> Sucesso: Accuracy (Holdout 70/30): {acc_holdout:.2f}%")

            # --- MELHORIA 2: CROSS-VALIDATION (K=5) ---
            print("-> A executar 5-Fold Cross-Validation...")
            k = 5
            fold_size = len(X) // k
            cv_scores = []
            for i in range(k):
                s, e = i * fold_size, (i + 1) * fold_size
                X_val_f, y_val_f = X[s:e], y[s:e]
                X_train_f = np.concatenate([X[:s], X[e:]])
                y_train_f = np.concatenate([y[:s], y[e:]])
                
                temp_tree = ID3DecisionTree(depth_limit=15)
                temp_tree.fit(X_train_f, y_train_f)
                cv_scores.append(np.mean(temp_tree.predict(X_val_f) == y_val_f))
            
            print(f"-> Média Cross-Validation: {np.mean(cv_scores)*100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)")
            
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")

    def get_move(self, state):
        # Converter bits do motor para formato do CSV (-1, 0, 1)
        board_array = np.full(42, -1)
        p0 = state.position if state.turn == 0 else (state.position ^ state.mask)
        p1 = state.position if state.turn == 1 else (state.position ^ state.mask)
        for i in range(42):
            bit = 1 << ((i % 7) * 7 + (i // 7))
            if p0 & bit: board_array[i] = 0
            elif p1 & bit: board_array[i] = 1
        
        input_v = np.append(board_array, state.turn)
        move = int(self.tree.predict(np.array([input_v]))[0])
        
        # Validação de segurança
        legal = state.legal_moves()
        return move if move in legal else random.choice(legal)