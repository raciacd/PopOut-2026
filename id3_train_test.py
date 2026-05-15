import numpy as np
import pandas as pd
from id3_engine import ID3DecisionTree

def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Args:
        X (numpy.ndarray): The feature matrix of the dataset.
        y (numpy.ndarray): The target labels array.
        test_size (float): The proportion of the dataset used for testing.
        random_state (int): Ensures reproducibility. Sets a fixed number for the seed.

    Returns:
        tuple: A tuple containing four arrays: (X_train, X_test, y_train, y_test).
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    X_shuffled, y_shuffled = X[indices], y[indices]
    
    cutoff = int((1 - test_size) * len(X))
    return X_shuffled[:cutoff], X_shuffled[cutoff:], y_shuffled[:cutoff], y_shuffled[cutoff:]

def iris_train():
    """
    Pipeline to train and evaluate the ID3 Decision Tree using the fixed Iris dataset.

    This function loads the 'iris.csv' data, splits it into training and testing sets,
    trains the ID3 model with a depth limit of 4, evaluates the accuracy on both sets,
    visually displays the generated tree, and runs a test case.
    """
    print("\n" + "="*40)
    print("        Training: Iris Dataset")
    print("="*40)
    try:
        df = pd.read_csv('iris.csv')
        X = df.iloc[:, 1:5].values
        y = df.iloc[:, 5].values
        feature_names = df.columns[1:5].tolist()

        # Train/test divison
        X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2)

        print(f"\nTraining tree (Depth: 4) with {len(X_train)} examples...")
        tree = ID3DecisionTree(depth_limit=4)
        
        tree.fit(X_train, y_train)

        acc_train = np.mean(tree.predict(X_train) == y_train) * 100
        acc_test = np.mean(tree.predict(X_test) == y_test) * 100

        print("\n--- Results ---")
        print(f"Accuracy on train set: {acc_train:.2f}%")
        print(f"Accuracy no test set: {acc_test:.2f}%\n")

        print("--- Tree Structure ---")
        tree.show(feature_names=feature_names)

        print("\n--- Single test ---")
        example = np.array([[5.1, 3.5, 1.4, 0.2]])
        print(f"Entry data: {example[0]}")
        print(f"Tree prediction: {tree.predict(example)[0]}")
        
    except FileNotFoundError:
        print("Error: iris.csv not found!")

def popout_train():
    """
    Pipeline to train and evaluate the ID3 Decision Tree using any PopOut dataset.

    Prompts the user for the dataset filename, filters the data to keep
    only the winning moves, splits it into training and testing sets, trains a deeper 
    tree (depth 20) to capture game logic, and evaluates its accuracy without printing 
    the massive tree structure to the terminal.
    """
    print("\n" + "="*40)
    print("       Training: PopOut Dataset")
    print("="*40)
    nome_csv = input("Enter the PopOut csv name (e.g. mcts_2sec_vs_random_10k.csv): ").strip()
    
    try:
        df = pd.read_csv(nome_csv)
        total_original = len(df)
        
        # Filter only the moves from the winner
        filtro_vitorias = (
            ((df['player_turn'] == 0) & (df['game_result'] == 1)) |
            ((df['player_turn'] == 1) & (df['game_result'] == -1))
        )
        df_limpo = df[filtro_vitorias]
        
        print(f"\n[INFO] Total moves on CSV: {total_original}")
        print(f"[INFO] Filtered winner moves: {len(df_limpo)}.")
        
        X = df_limpo.iloc[:, 0:43].values
        y = df_limpo.iloc[:, 43].values
        
        X_train, X_test, y_train, y_test = custom_train_test_split(X, y, test_size=0.2)
        
        print(f"\nTraining tree (Depth: 20) with {len(X_train)} examples...")
        tree = ID3DecisionTree(depth_limit=20)
        tree.fit(X_train, y_train)
        
        acc_train = np.mean(tree.predict(X_train) == y_train) * 100
        acc_test = np.mean(tree.predict(X_test) == y_test) * 100
        
        print("\n--- Results ---")
        print(f"Accuracy on train set: {acc_train:.2f}%")
        print(f"Accuracy on test set: {acc_test:.2f}%")
        print("The tree is too big to be printed on the terminal")
        
    except FileNotFoundError:
        print(f"Error: {nome_csv} not found!")

if __name__ == "__main__":
    while True:
        print("\n" + "="*40)
        print("    Training Menu")
        print("="*40)
        print("1 - Train and visualize Iris")
        print("2 - Train and evaluate PopOut (MCTS)")
        print("0 - Exit")
        
        escolha = input("Escolha uma opção: ").strip()
        
        if escolha == '1':
            iris_train()
        elif escolha == '2':
            popout_train()
        elif escolha == '0':
            print("Exiting...")
            break
        else:
            print("Invalid option!")