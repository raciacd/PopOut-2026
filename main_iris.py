import numpy as np
import pandas as pd
from id3_engine import ID3DecisionTree 

def correr_projeto_iris():
    print("--- Loading Iris Dataset ---")
    
    # 1. Load the CSV
    try:
        df = pd.read_csv('iris.csv')
    except FileNotFoundError:
        print("Error: iris.csv not found!")
        return

    # 2. Pre-processing
    # Features (X): sepallength to petalwidth / Labels (y): class
    X = df.iloc[:, 1:5].values
    y = df.iloc[:, 5].values
    feature_names = df.columns[1:5].tolist()

    # 3. Train the Tree
    tree = ID3DecisionTree(depth_limit=4)
    tree.fit(X, y)

    # --- NEW: ACCURACY CALCULATION ---
    # We ask the tree to predict the classes for the data it just studied
    predictions = tree.predict(X)
    
    # Compare predictions with the real 'y' values
    # np.mean(predictions == y) gives the percentage of correct answers
    accuracy = np.mean(predictions == y) * 100
    print(f"-> Training Accuracy: {accuracy:.2f}%")
    # ---------------------------------

    # 4. Show the Tree structure (Requirement)
    print("\n--- Generated Tree Structure ---")
    tree.show(feature_names=feature_names)

    # 5. Test with a new example
    print("\n--- Manual Test Case ---")
    example = np.array([[5.1, 3.5, 1.4, 0.2]])
    result = tree.predict(example)
    print(f"Input data: {example}")
    print(f"Tree Prediction: {result[0]}")

if __name__ == "__main__":
    correr_projeto_iris()

if __name__ == "__main__":
    correr_projeto_iris()
