from id3_engine import ID3DecisionTree
import numpy as np
import pandas as pd
import random

# --- AGENTE ID3: O BOT QUE APRENDE COM O MCTS ---

class ID3Play:
    def __init__(self, name="ID3-Bot", csv_path='mcts_2min_vs_random.csv'):
        self.name = name
        # Criamos a árvore. Usei profundidade 20 para ela não ficar demasiado simples
        self.tree = ID3DecisionTree(depth_limit=20)
        self.csv_path = csv_path
        # Mal o jogo começa, o bot "estuda" o ficheiro CSV
        self._prepare_model()

    def _prepare_model(self):
        """Aqui o bot lê o dataset e fazemos os testes para ver se ele aprendeu bem."""
        print(f"\n[!] O {self.name} está a estudar as jogadas do MCTS...")
        try:
            # Abrir o ficheiro com as 8300 jogadas que guardámos antes
            df = pd.read_csv(self.csv_path)
            X = df.iloc[:, 0:43].values  # O estado do tabuleiro e de quem é a vez
            y = df.iloc[:, 43].values     # A jogada que o MCTS sugere fazer
            
            # --- TESTE 1: DIVISÃO 70/30 (TREINO E TESTE) ---
            # Baralhamos tudo para o bot não viciar na ordem dos dados
            indices = np.random.permutation(len(X))
            ponto_de_corte = int(0.7 * len(X))
            
            # 70% dos dados para a árvore aprender, 30% para testarmos se ela acerta
            X_train, X_test = X[indices[:ponto_de_corte]], X[indices[ponto_de_corte:]]
            y_train, y_test = y[indices[:ponto_de_corte]], y[indices[ponto_de_corte:]]
            
            # Treinamos a nossa árvore ID3
            self.tree.fit(X_train, y_train)
            
            # Verificamos a eficácia nos 30% que ficaram de fora
            previsoes = self.tree.predict(X_test)
            accuracy = np.mean(previsoes == y_test) * 100
            print(f"-> No teste de 70/30, o bot acertou {accuracy:.2f}% das jogadas.")

            # --- TESTE 2: CROSS-VALIDATION (K=5) ---
            # Para ter a certeza que o resultado acima não foi sorte, testamos em 5 fatias diferentes
            print("-> A verificar a robustez com 5-Fold Cross-Validation...")
            k = 5
            tamanho_fatia = len(X) // k
            resultados_cv = []
            
            for i in range(k):
                inicio, fim = i * tamanho_fatia, (i + 1) * tamanho_fatia
                
                # Separamos uma fatia para validar e as outras 4 para treinar
                X_val, y_val = X[inicio:fim], y[inicio:fim]
                X_treino_cv = np.concatenate([X[:inicio], X[fim:]])
                y_treino_cv = np.concatenate([y[:inicio], y[fim:]])
                
                # Criamos uma árvore rápida só para este teste
                arvore_teste = ID3DecisionTree(depth_limit=15)
                arvore_teste.fit(X_treino_cv, y_treino_cv)
                
                # Guardamos o resultado desta rodada
                score = np.mean(arvore_teste.predict(X_val) == y_val)
                resultados_cv.append(score)
            
            # Mostramos a média final dos 5 testes
            media_final = np.mean(resultados_cv) * 100
            print(f"-> Média final do Cross-Validation: {media_final:.2f}%")
            
        except Exception as e:
            print(f"Houve um erro ao ler o CSV ou a treinar: {e}")

    def get_move(self, state):
        """Esta função é chamada na vez do bot para ele escolher uma coluna."""
        
        # 1. TRADUZIR O TABULEIRO: O motor usa bits, mas a árvore precisa de números simples
        # Criamos uma lista vazia (-1) para representar as 42 casas do Connect4
        tabuleiro_simples = np.full(42, -1)
        
        # Vamos buscar as peças do Jogador 0 e Jogador 1 através das máscaras de bits
        p0 = state.position if state.turn == 0 else (state.position ^ state.mask)
        p1 = state.position if state.turn == 1 else (state.position ^ state.mask)
        
        for i in range(42):
            # Transformamos a posição linear 'i' no bit correspondente no tabuleiro
            bit_da_casa = 1 << ((i % 7) * 7 + (i // 7))
            if p0 & bit_da_casa: tabuleiro_simples[i] = 0
            elif p1 & bit_da_casa: tabuleiro_simples[i] = 1
        
        # Juntamos o tabuleiro com a informação de quem joga agora (o 'turn')
        dados_entrada = np.append(tabuleiro_simples, state.turn)
        
        # 2. DECISÃO: Pedimos à árvore para prever qual seria a jogada do MCTS
        jogada_prevista = int(self.tree.predict(np.array([dados_entrada]))[0])
        
        # 3. VERIFICAÇÃO: Se a árvore sugerir uma coluna cheia, jogamos ao calhas para não crashar
        jogadas_possiveis = state.legal_moves()
        if jogada_prevista in jogadas_possiveis:
            return jogada_prevista
        else:
            # Backup caso a árvore se engane (situação nova que não estava no CSV)
            return random.choice(jogadas_possiveis)
