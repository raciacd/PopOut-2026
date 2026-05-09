import pandas as pd
import numpy as np
import sys
from id3_engine import ID3DecisionTree
from mcts_play import MCTSPlay

# --- TRUQUE PARA EVITAR O ERRO DO RANDOM_PLAY ---
# Criamos um módulo falso "random_play" na memória para o main.py não crashar
from types import ModuleType
fake_rp = ModuleType('random_play')
fake_rp.RandomPlay = lambda x: None 
sys.modules['random_play'] = fake_rp

try:
    from main import Position
    print("Motor de jogo carregado com sucesso!")
except ImportError as e:
    print(f"Erro crítico ao carregar as regras: {e}")
    sys.exit()

def iniciar_confronto():
    # 1. Treino da Árvore
    print("--- A treinar ID3-Bot com 8300 exemplos ---")
    nome_csv = 'mcts_2min_vs_random.csv'
    
    try:
        df = pd.read_csv(nome_csv)
        X = df.iloc[:, 0:43].values
        y = df.iloc[:, 43].values
        tree_model = ID3DecisionTree(depth_limit=20)
        tree_model.fit(X, y)
    except Exception as e:
        print(f"Erro no CSV: {e}")
        return

    # 2. Configurar Agentes
    player_mcts = MCTSPlay(name="MCTS-Master", time_limit=1.0)
    
    def id3_strategy(state):
        # Conversão Bitwise para o formato do CSV (-1, 0, 1)
        board_array = np.full(42, -1)
        # No teu motor, Player 0 e Player 1 são calculados assim:
        p0_bits = state.position if state.turn == 0 else (state.position ^ state.mask)
        p1_bits = state.position if state.turn == 1 else (state.position ^ state.mask)
        
        for i in range(42):
            # Mapeamento do bit para a célula 0-41
            col = i % 7
            row = i // 7
            bit = 1 << (col * 7 + row)
            if p0_bits & bit: board_array[i] = 0
            elif p1_bits & bit: board_array[i] = 1
            
        input_vector = np.append(board_array, state.turn)
        prediction = tree_model.predict(np.array([input_vector]))[0]
        return int(prediction)

    # 3. Loop do Jogo
    state = Position(turn=0) 
    print("\n--- JOGO INICIADO: MCTS vs ID3 ---")
    
    while not state.terminal:
        state.print_board(legal_moves=state.legal_moves())
        
        if state.turn == 0: 
            move = player_mcts.get_move(state)
        else:
            print("\n[ID3-Bot] A pensar...")
            move = id3_strategy(state)
            if move not in state.legal_moves():
                import random
                move = random.choice(state.legal_moves())
            print(f"ID3 escolheu: {move}")
        
        state = state.move(move)

    state.print_board()
    vencedor = "Empate" if state.result == 0 else ("MCTS" if state.result == 1 else "ID3")
    print(f"\nFim de jogo! Vencedor: {vencedor}")

if __name__ == "__main__":
    iniciar_confronto()