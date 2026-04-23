from random_play import RandomPlay

class Connect4:
    def __init__(self):
        self.turn = 0
        self.result = None
        self.terminal = False
    def get_initial_position(self):
        return Position(self.turn)
                
class Position:
    def __init__(self, turn, mask = 0, position = 0, num_turns = 0, history = None):
        self.turn = turn
        self.result = None
        self.terminal = False
        self.num_turns = num_turns
        self.mask = mask
        self.position = position
        self._compute_hash()
        
        # Repetition rule
        if history is None:
            self.history = {self.hash: 1}
        else:
            self.history = history
            self.history[self.hash] = self.history.get(self.hash, 0) + 1
                
    # returns new position
    def move(self, loc):
        # Rules 2 and 3, draw
        if loc == -1:
            new_pos = Position(int(not self.turn), self.mask, self.position ^ self.mask, self.num_turns + 1, self.history.copy())
            new_pos.terminal = True
            new_pos.result = 0
            return new_pos

        is_pop = False
        
        if 0 <= loc <= 6:
            # Normal move (0-6)
            new_position = self.position ^ self.mask
            new_mask = self.mask | (self.mask + (1 << (loc*7)))
        else:
            # Pop move (7-13)
            c = loc - 7
            col_mask = 0b111111 << (c * 7)
            
            curr_col = self.position & col_mask
            opp_col = (self.position ^ self.mask) & col_mask
            
            # shift move
            new_curr_col = (curr_col >> 1) & col_mask
            new_opp_col = (opp_col >> 1) & col_mask
            
            new_curr = (self.position & ~col_mask) | new_curr_col
            new_opp = ((self.position ^ self.mask) & ~col_mask) | new_opp_col
            
            new_position = new_opp
            new_mask = new_curr | new_opp
            is_pop = True

        new_pos = Position(int(not self.turn), new_mask, new_position, self.num_turns + 1, self.history.copy())
        new_pos.game_over(is_pop)
        return new_pos
    
    # return list of legal moves
    def legal_moves(self):
        bit_moves = []
        # regular moves
        for i in range(7):
            col_mask = 0b111111 << 7 * i
            if col_mask != self.mask & col_mask:
                bit_moves.append(i)
                
        # pop move
        # checks if the piece belongs to the current player
        for i in range(7):
            bottom_bit = 1 << (i * 7)
            if self.position & bottom_bit:
                bit_moves.append(i + 7)

        # draw checkm, full board
        is_full = (self.mask == 279258638311359)
        # second draw check, repetition
        is_repetition = self.history.get(self.hash, 0) >= 3
        
        # player declares draw
        if is_full or is_repetition:
            bit_moves.append(-1)
            
        return bit_moves

    def game_over(self, is_pop=False):
        # sets result to -1, 0, or 1 if game is over (otherwise self.result is None)
        
        just_moved_won = self.connected_four_fast()
        about_to_move_won = False
        
        if is_pop:
            about_to_move_won = self.connected_four_fast(self.position)
            
        if just_moved_won or about_to_move_won:
            self.terminal = True
            
            if just_moved_won:
                self.result = 1 if self.turn == 1 else -1 
            else:
                self.result = 1 if self.turn == 0 else -1
        else:
            self.terminal = False
            self.result = None
            
        if not self.terminal and not self.legal_moves():
            self.terminal = True
            self.result = 0
            
    def connected_four_fast(self, board_position=None):
        if board_position is None:
            board_position = self.position ^ self.mask
            
        # Horizontal check
        m = board_position & (board_position >> 7)
        if m & (m >> 14):
            return True
        # Diagonal \
        m = board_position & (board_position >> 6)
        if m & (m >> 12):
            return True
        # Diagonal /
        m = board_position & (board_position >> 8)
        if m & (m >> 16):
            return True
        # Vertical
        m = board_position & (board_position >> 1)
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

    # Visual interface
    def print_board(self, legal_moves=None):
        p0_bits = self.position if self.turn == 0 else (self.position ^ self.mask)
        p1_bits = self.position if self.turn == 1 else (self.position ^ self.mask)
        
        display = "\n  7  8  9 10 11 12 13 : Pop"
        display += "\n  0  1  2  3  4  5  6 : Drop\n-----------------------"
        for row in range(5, -1, -1):
            line = "|"
            for col in range(7):
                bit = 1 << (col * 7 + row)
                if p0_bits & bit:
                    line += " O " # Player 0
                elif p1_bits & bit:
                    line += " X " # Player 1
                else:
                    line += " . " # Empty
            display += "\n" + line + "|"
        display += "\n-----------------------"

        if legal_moves is not None:
                    drops = [m for m in legal_moves if 0 <= m <= 6]
                    pops = [m for m in legal_moves if 7 <= m <= 13]
                    draw = [-1] if -1 in legal_moves else []
                    
                    display += "\nLegal Moves:"
                    if drops: display += f"\n  Drops: {drops}"
                    if pops:  display += f"\n  Pops:  {pops}"
                    if draw:  display += f"\n  Draw:  [-1]"
                    display += "\n"

        print(display)

class HumanPlay:
    def __init__(self, name="Human"):
        self.name = name

    def get_move(self, position):
        legal = position.legal_moves()
        while True:
            try:
                move = int(input(f"[{self.name}] Choose a move : "))
                if move in legal:
                    return move
                print("Invalid move! Try again.")
            except ValueError:
                print("Please enter a number.")

# Game loop
if __name__ == "__main__":
    
    def choose_player(num_player):
        while True:
            print(f"\nChoose a player for {num_player}")
            print("1 - Human")
            print("2 - Random Play")
            player_str = input("Enter 1 or 2: ").strip()
            
            if player_str == '1':
                return HumanPlay(f"Player {num_player} (Human)")
            elif player_str == '2':
                return RandomPlay(f"Player {num_player} (Random)")
            else:
                print("Invalid player! Please enter 1 or 2.")

    agents = {
        0: choose_player(0),
        1: choose_player(1)
    }

    game = Connect4()
    pos = game.get_initial_position()
    
    print(f"\n--- Game start: {agents[0].name} VS {agents[1].name} ---")

    # Loop Principal
    while not pos.terminal:
        legal = pos.legal_moves()
        pos.print_board(legal_moves=legal)
        
        current_agent = agents[pos.turn]
        print(f"\n{current_agent.name}'s turn.")
        
        move = current_agent.get_move(pos)
        pos = pos.move(move)

    # Fim de jogo
    pos.print_board()
    if pos.result == 0:
        print("It's a Draw!")
    else:
        winner = 0 if pos.result == 1 else 1
        print(f"Player {winner} wins!")