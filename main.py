from random_play import RandomPlay
from mcts_play import MCTSPlay
from id3_play import ID3Play

class Connect4:
    """
    Main class to manage the game initialization
    """
    def __init__(self):
        """
        Initializes the Connect4 game.
        """
        self.turn = 0
        self.result = None
        self.terminal = False
        
    def get_initial_position(self):
        """
        Gets the starting position of the game.

        Returns:
            Position: The initial board state with turn 0.
        """
        return Position(self.turn)
                
class Position:
    """
    Represents a state of the game board using bitboards.
    """
    # Removed 'history' parameter to avoid memory overload during rollouts
    def __init__(self, turn, mask = 0, position = 0, num_turns = 0):
        """
        Initializes a game state.

        Args:
            turn (int): The current player's turn (0 or 1).
            mask (int): Bitboard representing all pieces on the board. Defaults to 0.
            position (int): Bitboard representing the current player's pieces. Defaults to 0.
            num_turns (int): Counter for the number of turns played. Defaults to 0.
        """
        self.turn = turn
        self.result = None
        self.terminal = False
        self.num_turns = num_turns
        self.mask = mask
        self.position = position
        self._compute_hash()
                
    def move(self, loc):
        """
        Applies a move and returns a new board state.

        Args:
            loc (int): The chosen move. 0-6 for dropping a piece, 7-13 for popping a piece from the bottom, or -1 for a draw.

        Returns:
            Position: A new Position instance of the board after the move.
        """
        # Rules 2 and 3, draw
        if loc == -1:
            # Creation without history copy (pure math and bitwise operations)
            new_pos = Position(int(not self.turn), self.mask, self.position ^ self.mask, self.num_turns + 1)
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

        # mathematical instance creation (tries to fix garbage collection bottleneck)
        new_pos = Position(int(not self.turn), new_mask, new_position, self.num_turns + 1)
        new_pos.game_over(is_pop)
        return new_pos
    
    # return list of legal moves
    def legal_moves(self):
        """
        Determines all valid moves for the current board state.

        Returns:
            list: A list of integers representing valid moves (0-6 for drops, 7-13 for pops, -1 for draw).
        """
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

        # draw check, full board
        is_full = (self.mask == 279258638311359)
        
        if is_full:
            bit_moves.append(-1)
            
        return bit_moves

    def game_over(self, is_pop=False):
        """
        Checks if the game has ended and updates the terminal status and result.

        Args:
            is_pop (bool, optional): Indicates if the last move was a PopOut move. Defaults to False.
        """
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
        """
        Detection of four connected pieces using bitwise shifts.

        Args:
            board_position (int): The board to evaluate. If None, evaluates the current player's board.

        Returns:
            bool: True if there is a 4-in-a-row, False otherwise.
        """
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
        """
        Computes a unique hash for the current board state.
        """
        position_1 = self.position if self.turn == 0 else self.position ^ self.mask
        self.hash = 2 * hash((position_1, self.mask)) + self.turn
    
    def __hash__(self):
        return self.hash
        
    def __eq__(self, other):
        return isinstance(other, Position) and self.turn == other.turn and self.mask == other.mask and self.position == other.position

    # Visual interface
    def print_board(self, legal_moves=None):
        """
        Displays the board to the console.

        Args:
            legal_moves (list): List of legal moves to be displayed for the player.
        """
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
    """
    Wrapper class that manages the human player input.

    Attributes:
        name (str): The display name of the human player.
    """
    def __init__(self, name="Human"):
        """
        Initializes the instance
        """
        self.name = name

    def get_move(self, position):
        """
        Prompts the user for a move and validates it against the legal moves.

        Args:
            position : The current state of the game board.

        Returns:
            int: The index of the selected valid move.
        """
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
        """
        Displays a menu for the user to select the type of agent for a specific player.

        Args:
            num_player (int): The player ID (0 or 1).

        Returns:
            object: An instance of the selected player class (HumanPlay, RandomPlay, MCTSPlay, or ID3Play).
        """
        while True:
            print(f"\nChoose a player for {num_player}")
            print("1 - Human")
            print("2 - Random Play")
            print("3 - MCTS")
            print("4 - ID3 (MCTS)")
            player_str = input("Enter 1 or 2 or 3 or 4: ").strip()
            
            if player_str == '1':
                return HumanPlay(f"Player {num_player} (Human)")
            elif player_str == '2':
                return RandomPlay(f"Player {num_player} (Random)")
            elif player_str == '3':
                return MCTSPlay(f"Player {num_player} (MCTS)", time_limit=2.0)
            elif player_str == '4':
                return ID3Play(f"Player {num_player} (ID3)")
            else:
                print("Invalid player! Please enter 1, 2 or 3.")

    agents = {
        0: choose_player(0),
        1: choose_player(1)
    }

    game = Connect4()
    pos = game.get_initial_position()
    
    print(f"\n--- Game start: {agents[0].name} VS {agents[1].name} ---")

    # memory stored to check for draws
    global_history = {}

    # Main Loop
    while not pos.terminal:
        # updates the current board state occurences
        global_history[pos.hash] = global_history.get(pos.hash, 0) + 1
        
        legal = pos.legal_moves()
        
        # Injects the draw option (-1) if threefold repetition is met
        if global_history[pos.hash] >= 3 and -1 not in legal:
            legal.append(-1)
            print("\n[!] Attention: This state has repeated 3 times. Draw (-1) is available.")

        pos.print_board(legal_moves=legal)
        
        current_agent = agents[pos.turn]
        print(f"\n{current_agent.name}'s turn.")
        
        move = current_agent.get_move(pos)
        pos = pos.move(move)

    # End of game
    pos.print_board()
    if pos.result == 0:
        print("It's a Draw!")
    else:
        winner = 0 if pos.result == 1 else 1
        print(f"Player {winner} wins!")