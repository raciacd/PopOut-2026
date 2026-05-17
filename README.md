# PopOut game implementation, alongside Monte Carlo Tree Search, ID3 Decision Tree and Data Generator

This project is part of the **Artificail Inteligence** course at the **Faculty of Sciences, University of Porto**,
developed to explore and better understand tree traversal algorithms and machine learning.

## Setup and Installation

### 1. Create and Activate a Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Execution

### Main game

Run the main.py file and choose according to the terminal choices. The game offers a small variety of setups.

```python
python main.py
```

### ID3 Train and test statistics

Run the id3_train_test.py file and choose according to the terminal choices. You can choose either Decision Tree benchmarks, the Iris or a custom PopOut one. For the PopOut train and test, be sure to write the file name correcl

Run file:

```bash
python id3_train_test.py
```

### Data generation

To generate PopOut datasets, follow the steps. At execution, you'll be prompted with choices of number of games, what agent to play and the number of cores to use (worker parallelism for the mcts tree building).

Run the file:

```python
python dataset_generator.py
```
