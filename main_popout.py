import numpy as np
import pandas as pd
from id3_engine import ID3DecisionTree

def preparar_dados_e_treinar(nome_ficheiro):
    print(f"--- A carregar dataset: {nome_ficheiro} ---")
    try:
        df = pd.read_csv(nome_ficheiro)
        
        # O teu CSV tem 45 colunas. 
        # Vamos usar as primeiras 43 (tabuleiro + turno) como atributos (X)
        X = df.iloc[:, 0:43].values
        
        # A coluna 'chosen_move' (índice 43) é o que queremos prever (y)
        y = df.iloc[:, 43].values
        
        print(f"A treinar ID3 com {len(X)} exemplos de jogadas...")
        # Aumentamos a profundidade para 20 para captar melhor a estratégia do jogo
        tree = ID3DecisionTree(depth_limit=20) 
        tree.fit(X, y)
        
        # Teste de Accuracy rápido
        previsoes = tree.predict(X)
        acc = np.mean(previsoes == y) * 100
        print(f"Treino concluído! Accuracy no set de treino: {acc:.2f}%")
        
        return tree
    except Exception as e:
        print(f"Erro ao carregar ou treinar: {e}")
        return None

if __name__ == "__main__":
    # NOME EXATO DO TEU FICHEIRO
    nome_do_csv = 'mcts_2min_vs_random.csv' 
    
    arvore = preparar_dados_e_treinar(nome_do_csv)
    
    if arvore:
        print("\n[SUCESSO] A árvore está pronta para ser usada no play_game.py!")
    else:
        print("\n[ERRO] Verifica se o ficheiro mcts_2min_vs_random.csv está na mesma pasta.")