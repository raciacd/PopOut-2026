class Connect4:
    def __init__(self):
        self.turn = 0
        self.result = None
        self.terminal = False
    def get_initial_position(self):
        return Position(self.turn)
                
class Position:
    def __init__(self, turn, mask = 0, position = 0, num_turns = 0):
        self.turn = turn
        self.result = None
        self.terminal = False
        self.num_turns = num_turns
        self.mask = mask
        self.position = position
        self._compute_hash()
                
    # returns new position
    def move(self, loc):
        new_position = self.position ^ self.mask
        new_mask = self.mask | (self.mask + (1 << (loc*7)))

        new_pos = Position(int(not self.turn), new_mask, new_position, self.num_turns + 1)
        new_pos.game_over()
        return new_pos
    
    # return list of legal moves
    def legal_moves(self):
        bit_moves = []
        for i in range(7):
            col_mask = 0b111111 << 7 * i
            if col_mask != self.mask & col_mask:
                bit_moves.append(i)
        return bit_moves
    

    def game_over(self):
        # sets result to -1, 0, or 1 if game is over (otherwise self.result is None)
        connected_4 = self.connected_four_fast()
        
        if connected_4:
            self.terminal = True
            # The player who JUST moved is the winner.
            # Since turn was already flipped in move(), we check the previous turn.
            self.result = 1 if self.turn == 0 else -1
        else:
            self.terminal = False
            self.result = None
            
        # mask when all spaces are full
        if self.mask == 279258638311359:
            self.terminal = True
            self.result = 0
            
    def connected_four_fast(self):
        other_position = self.position ^ self.mask
        
        # Horizontal check
        m = other_position & (other_position >> 7)
        if m & (m >> 14):
            return True
        # Diagonal \
        m = other_position & (other_position >> 6)
        if m & (m >> 12):
            return True
        # Diagonal /
        m = other_position & (other_position >> 8)
        if m & (m >> 16):
            return True
        # Vertical
        m = other_position & (other_position >> 1)
        if m & (m >> 2):
            return True
        # Nothing found
        return False
    
    
    def _compute_hash(self):
        position_1 = self.position if self.turn == 0 else self.position ^ self.mask
        self.hash = 2 * hash((position_1, self.mask)) + self.turn
    
    def __hash__(self):
        return self.hash
    def __eq__(self, other):
        return isinstance(other, Position) and self.turn == other.turn and self.mask == other.mask and self.position == other.position

    # --- ADDED FOR VISUALIZATION ---
    def print_board(self):
        # We determine Player 0 and Player 1 bits based on current turn
        p0_bits = self.position if self.turn == 0 else (self.position ^ self.mask)
        p1_bits = self.position if self.turn == 1 else (self.position ^ self.mask)
        
        display = "\n 0 1 2 3 4 5 6\n---------------"
        for row in range(5, -1, -1):
            line = "|"
            for col in range(7):
                bit = 1 << (col * 7 + row)
                if p0_bits & bit:
                    line += "X " # Player 0
                elif p1_bits & bit:
                    line += "O " # Player 1
                else:
                    line += ". " # Empty
            display += "\n" + line + "|"
        print(display + "\n---------------")

# --- RUDIMENTARY GAME LOOP ---
if __name__ == "__main__":
    game = Connect4()
    pos = game.get_initial_position()
    
    while not pos.terminal:
        pos.print_board()
        legal = pos.legal_moves()
        print(f"Player {pos.turn}'s turn. Legal moves: {legal}")
        
        try:
            move = int(input("Choose a column: "))
            if move not in legal:
                print("Invalid move! Try again.")
                continue
            pos = pos.move(move)
        except ValueError:
            print("Please enter a number.")

    pos.print_board()
    if pos.result == 0:
        print("It's a Draw!")
    else:
        winner = 0 if pos.result == 1 else 1
        print(f"Player {winner} wins!")