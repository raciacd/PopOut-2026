import random
import time

class RandomPlay:
    """
    Wrapper class that manages an agent that plays random moves.

    Attributes:
        name (str): The display name of the random agent.
    """
    def __init__(self, name="Random"):
        """
        Initializes the instance.

        Args:
            name (str): The name of the agent, "Random".
        """
        self.name = name

    def get_move(self, position):
        """
        Evaluates the board state and returns a random legal move.

        Args:
            position (Position): The current state of the game board.

        Returns:
            int: The index of the randomly selected valid move.
        """
        legal_moves = position.legal_moves()
        
        time.sleep(0.5) # Doesn't let the game play too fast 
        
        move = random.choice(legal_moves)
        print(f"{self.name} played: {move}")
        return move