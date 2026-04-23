import random
import time

class RandomPlay:
    def __init__(self, name="Random"):
        self.name = name

    def get_move(self, position):
        legal_moves = position.legal_moves()
        
        time.sleep(0.5) # Doesn't let the game play too fast 
        
        move = random.choice(legal_moves)
        print(f"{self.name} played: {move}")
        return move