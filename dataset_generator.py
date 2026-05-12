import csv
import time
from main import Connect4, Position
from mcts_play import MCTSPlay
from random_play import RandomPlay

def bitboard_to_flat(pos):
    """Translates the ultra-fast Bitboards into a 42-element list (Flat 6x7 Matrix).

    Legend for the ID3 algorithm:
         0 = Player 0's piece (O)
         1 = Player 1's piece (X)
        -1 = Empty space

    Args:
        pos (Position): The current game state.

    Returns:
        list: A list of 42 integers representing the board state from top to bottom, 
            left to right.
    """
    p0_bits = pos.position if pos.turn == 0 else (pos.position ^ pos.mask)
    p1_bits = pos.position if pos.turn == 1 else (pos.position ^ pos.mask)
    
    flat_board = []
    
    # Reads the board from top to bottom (row 5 to 0), left to right (col 0 to 6)
    for row in range(5, -1, -1):
        for col in range(7):
            bit = 1 << (col * 7 + row)
            if p0_bits & bit:
                flat_board.append(0)  # Player 0's piece
            elif p1_bits & bit:
                flat_board.append(1)  # Player 1's piece
            else:
                flat_board.append(-1) # Empty space
                
    return flat_board

def generate_dataset(agent_0, agent_1, num_games=10, filename="connect4_dataset.csv"):
    """Simulates full games between two chosen agents and exports the history to a CSV.

    Args:
        agent_0 (Object): The agent instance playing as Player 0.
        agent_1 (Object): The agent instance playing as Player 1.
        num_games (int, optional): The number of complete games to simulate. Defaults to 10.
        filename (str, optional): The output CSV filename. Defaults to "connect4_dataset.csv".
    """
    print(f"\nStarting dataset generation: {num_games} games.")
    print(f"Matchup: {agent_0.name} VS {agent_1.name}\n")
    
    agents = {0: agent_0, 1: agent_1}

    wins_0 = 0
    wins_1 = 0
    draws = 0

    # Prepare the CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Create the header (Features: cell_0 to cell_41 | Targets: turn, move, result)
        header = [f"cell_{i}" for i in range(42)] + ["player_turn", "chosen_move", "game_result"]
        writer.writerow(header)

        for i in range(num_games):
            print(f"--- Simulating Game {i+1} of {num_games} ---")
            game = Connect4()
            pos = game.get_initial_position()
            global_history = {}
            
            # Temporary memory for this specific game
            game_records = [] 

            while not pos.terminal:
                # Referee logic (Draw detection via Threefold Repetition)
                global_history[pos.hash] = global_history.get(pos.hash, 0) + 1
                legal = pos.legal_moves()
                if global_history[pos.hash] >= 3 and -1 not in legal:
                    legal.append(-1)

                current_agent = agents[pos.turn]
                
                # 1. Capture the Current State (Translation for ID3)
                flat_b = bitboard_to_flat(pos)
                turn = pos.turn
                
                # 2. Agent chooses the move
                move = current_agent.get_move(pos)
                
                # 3. Store the record in temporary memory
                game_records.append((flat_b, turn, move))
                
                # 4. Apply the move to the real board
                pos = pos.move(move)

            # When the game ends, evaluate the final result
            # 1.0 -> Player 0 won | -1.0 -> Player 1 won | 0.0 -> Draw
            if pos.result == 1:
                final_result = 1
                wins_0 += 1
                winner_str = f"Player 0 ({agent_0.name}) Won"
            elif pos.result == -1:
                final_result = -1
                wins_1 += 1
                winner_str = f"Player 1 ({agent_1.name}) Won"
            else:
                final_result = 0
                draws += 1
                winner_str = "Draw"
            
            # Backpropagate the final result to save in the CSV
            for record in game_records:
                flat_b, turn, move = record
                # Write row: [42 cells] + Turn + Chosen Move + Game Result
                writer.writerow(flat_b + [turn, move, final_result])
            
            print(f"Game {i+1} finished. Result: {winner_str}\n")
            
    print(f"Dataset successfully generated! Saved to: {filename}\n")
    print("=" * 40)
    print("      FINAL GENERATION SUMMARY")
    print("=" * 40)
    print(f"Total Games Played: {num_games}")
    print(f"Player 0 ({agent_0.name}) Wins: {wins_0} ({ (wins_0/num_games)*100 :.1f}%)")
    print(f"Player 1 ({agent_1.name}) Wins: {wins_1} ({ (wins_1/num_games)*100 :.1f}%)")
    print(f"Draws: {draws} ({ (draws/num_games)*100 :.1f}%)")
    print("=" * 40)
    print(f"Dataset saved to: {filename}\n")

# ==========================================
# INTERACTIVE MENU FOR CONFIGURATION
# ==========================================
if __name__ == "__main__":
    print("=== Connect 4 Dataset Generator ===")
    
    try:
        num_games = int(input("How many games do you want to simulate? (e.g. 10, 100): "))
        
        selected_agents = {}
        for player_id in range(2):
            print(f"\n--- Configure Player {player_id} ---")
            print("1 - MCTS AI")
            print("2 - Random AI")
            choice = input("Select the agent type (1 or 2): ").strip()
            
            if choice == '1':
                time_limit = float(input(f"Thinking time for MCTS {player_id} (seconds, e.g. 0.5, 2.0): "))
                cores = int(input(f"Number of CPU cores for MCTS {player_id} (e.g. 4): "))
                selected_agents[player_id] = MCTSPlay(name=f"MCTS_{player_id}", time_limit=time_limit, num_workers=cores)
            else:
                selected_agents[player_id] = RandomPlay(name=f"Random_{player_id}")
                
        filename = input("\nEnter the output CSV filename (e.g. dataset.csv): ").strip()
        if not filename.endswith(".csv"):
            filename += ".csv"
            
        generate_dataset(selected_agents[0], selected_agents[1], num_games=num_games, filename=filename)
        
    except ValueError:
        print("Invalid input! Please enter numbers only where requested.")