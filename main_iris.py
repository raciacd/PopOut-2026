import numpy as np
import pandas as pd
from id3_engine import ID3DecisionTree # Importa a tua classe

def correr_projeto_iris():
    print("--- A carregar dados do Iris ---")
    
    # 1. Carregar o CSV
    try:
        df = pd.read_csv('iris.csv')
    except FileNotFoundError:
        print("Erro: O ficheiro iris.csv não foi encontrado na pasta!")
        return

    # 2. Pré-processamento (Obrigatório para este dataset)
    # O CSV que enviaste tem: [ID, sepallength, sepalwidth, petallength, petalwidth, class]
    # Precisamos de remover a coluna 'ID' porque ela baralha a árvore (o ID não ajuda a prever a flor)
    
    # Atributos (X): colunas da 1 à 4 (sepallength até petalwidth)
    X = df.iloc[:, 1:5].values
    
    # Etiquetas/Classes (y): última coluna (class)
    y = df.iloc[:, 5].values
    
    # Nomes dos atributos para a visualização
    feature_names = df.columns[1:5].tolist()

    # 3. Treinar a Árvore
    # Criamos a árvore com um limite de profundidade (ex: 4) para evitar sobreajuste
    tree = ID3DecisionTree(depth_limit=4)
    tree.fit(X, y)

    # 4. Mostrar a Árvore Visualmente (Requisito do enunciado)
    print("\n--- Estrutura da Árvore Gerada ---")
    tree.show(feature_names=feature_names)

    # 5. Testar com um exemplo novo (Aceitar test examples)
    print("\n--- Teste de Classificação ---")
    # Vamos inventar uma flor: SepalL=5.1, SepalW=3.5, PetalL=1.4, PetalW=0.2
    exemplo = np.array([[5.1, 3.5, 1.4, 0.2]])
    resultado = tree.predict(exemplo)
    print(f"Dados inseridos: {exemplo}")
    print(f"Resultado da Árvore: {resultado[0]}")

if __name__ == "__main__":
    correr_projeto_iris()