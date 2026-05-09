import random
import math
import time
import os
from multiprocessing import Pool

def mcts_worker(args):
    """
    Worker function to build an MCTS tree in a separate CPU core.

    Args: tuple containing root_state and time_limit.

    Return: tuple containing children_stats (dict), iterations (int) and total rollouts (int).
    """
    root_state, time_limit = args
    
    # Ensures each CPU core generates completely different random games
    random.seed(os.urandom(4))

    nodes, iterations, total_rollouts = build_mcts_tree(root_state, time_limit)
    
    children_stats = {}
    for loc in root_state.legal_moves():
        next_state = root_state.move(loc)
        if next_state in nodes:
            w, n, _ = nodes[next_state]
            children_stats[loc] = (w, n)
            
    return children_stats, iterations, total_rollouts


def build_mcts_tree(root_state, time_limit):
    """
    Builds the MCTS statistical tree given a time limit.

    Args: tuple containgin root_state and time_limit.

    Return: tuple containing nodes (dict), iterations (int) and total_rollouts (int).
    """
    # Structure: {state: (total_reward, visits, {parent_state: parent_visits})}
    nodes = {} 
    nodes[root_state] = (0.0, 0.0, {root_state: 0})
    
    start_time = time.time()
    iterations = 0
    total_rollouts = 0 
    
    while time.time() - start_time < time_limit:   
        iterations += 1

        # ==========================================
        # SELECTION
        # ==========================================
        selection_path = selection_phase(nodes, root_state)
        selected_node = selection_path[-1]
        
        _, visits, _ = nodes[selected_node]
        
        # ==========================================
        # EXPANSION
        # ==========================================
        if visits > 0 and not selected_node.terminal:
            legal_moves = selected_node.legal_moves()
            for loc in legal_moves:
                new_state = selected_node.move(loc)
                if new_state not in nodes:
                    nodes[new_state] = (0.0, 0.0, {selected_node: 0})
                    
            loc = random.choice(legal_moves)
            child_state = selected_node.move(loc)
            
            # ==========================================
            # SIMULATION / ROLLOUT
            # ==========================================
            reward = 0
            num_runs = 10
            for _ in range(num_runs):
                reward += simulate_rollout(child_state)
            total_rollouts += num_runs
            
            w, n, parent_n_dict = nodes[child_state]
            if selected_node not in parent_n_dict:
                parent_n_dict[selected_node] = 0
            parent_n_dict[selected_node] += 1
            nodes[child_state] = (w + reward, n + num_runs, parent_n_dict)

        else:
            # ==========================================
            # SIMULATION / ROLLOUT (Direct Simulation)
            # ==========================================
            reward = 0
            num_runs = 10
            for _ in range(num_runs):
                reward += simulate_rollout(selected_node)   
            total_rollouts += num_runs
        
        # ==========================================
        # BACKPROPAGATION
        # ==========================================
        parent = root_state
        for state_in_path in selection_path:
            w, n, parent_n_dict = nodes[state_in_path]
            parent_n_dict[parent] += num_runs
            nodes[state_in_path] = (w + reward, n + num_runs, parent_n_dict)
            parent = state_in_path
            
    return nodes, iterations, total_rollouts


def mcts_agent(time_limit, num_workers=4):
    """
    Wraps the MCTS strategy, returning a move function.

    Args: tuple containing time_limit and num_workers.

    Return: function that takes current_state and returns the optimal move.
    """
    def strat(current_state):
        # Prepare arguments for multiprocessing
        args = [(current_state, time_limit) for _ in range(num_workers)]
        
        # Run MCTS in parallel across available CPU cores
        with Pool(processes=num_workers) as pool:
            results = pool.map(mcts_worker, args)
            
        total_iters = 0
        total_rollouts = 0
        merged_stats = {}
        
        # Merge stats from all workers
        for children_stats, iters, rollouts in results:
            total_iters += iters
            total_rollouts += rollouts
            
            for loc, (w, n) in children_stats.items():
                if loc not in merged_stats:
                    merged_stats[loc] = [0.0, 0.0]
                merged_stats[loc][0] += w
                merged_stats[loc][1] += n

        iter_per_sec = total_iters / time_limit
        rollouts_per_sec = total_rollouts / time_limit
        
        print(f"Number of MCTS iterations: {total_iters} ({iter_per_sec:.0f} iters/sec) [Cores: {num_workers}]")
        print(f"Number of MCTS rollouts:   {total_rollouts} ({rollouts_per_sec:.0f} rollouts/sec) [Cores: {num_workers}]")
        
        player = current_state.turn
        best_score = float('-inf') if player == 0 else float('inf')
        next_best_move = None
   
        for loc in current_state.legal_moves():
            if loc not in merged_stats:
                score = 0.0
            else:
                w, n = merged_stats[loc]
                score = 0.0 if n == 0 else w / n
            
            if score < best_score and player == 1: # Player 1 minimizes
                best_score = score
                next_best_move = loc
            elif score > best_score and player == 0: # Player 0 maximizes
                best_score = score
                next_best_move = loc
                
        # Safety fallback
        if next_best_move is None:
            return random.choice(current_state.legal_moves())
            
        return next_best_move
    return strat
        

def simulate_rollout(state, max_depth=50):
    """
    Executes random rollout from given state until the game ends or reaches the number of plays determined by max_depoth.

    Args: tuple containing state and max_depth.

    Return: float for standardized reward: 1.0 for player 0 win, 0.0 for player 1 win, and 0.5 for a draw.
    """
    current_state = state
    depth = 0
    
    while not current_state.terminal and depth < max_depth:
        moves = current_state.legal_moves()
        loc = random.choice(moves)
        current_state = current_state.move(loc) 
        depth += 1
        
    if not current_state.terminal:
        return 0.5 

    if current_state.result == 1: return 1.0
    elif current_state.result == -1: return 0.0
    return 0.5
        

def selection_phase(nodes, root_state):
    """
    Navigates the tree using the UCB1 policy.

    Args: tuple containing nodes and root_state.

    Return: a list containing the path of traversed nodes from the root to the selected leaf.
    """
    current_node = root_state
    path = []
    visited = set() # Cycle detector for PopOut infinite loops
    
    while True:
        if current_node in visited:
            return path
        
        visited.add(current_node)
        
        _, visits, _ = nodes[current_node]
        path.append(current_node)
        
        if visits == 0:
            return path
        
        legal_moves = current_node.legal_moves()
        next_player = current_node.turn
        
        best_score = float('-inf') if next_player == 0 else float('inf')
        next_best_node = None
        
        for loc in legal_moves:
            result_state = current_node.move(loc)
            
            if result_state not in nodes:
                return path
                
            temp_w, temp_visits, temp_parent_n_count = nodes[result_state]
            if current_node not in temp_parent_n_count:
                temp_parent_n_count[current_node] = 0
                
            if temp_parent_n_count[current_node] == 0:
                path.append(result_state)
                return path
                        
            score = calculate_ucb_score(nodes[current_node][1], temp_parent_n_count[current_node], temp_w / temp_visits, next_player)
            
            if score < best_score and next_player == 1:
                best_score = score
                next_best_node = result_state
            elif score > best_score and next_player == 0:
                best_score = score
                next_best_node = result_state
                
        current_node = next_best_node
        if current_node is None:
            return path
            
            
def calculate_ucb_score(parent_visits, node_visits, win_rate, player, exploration_const=2.0):
    """
    Upper Confidence Bound (UCB1) formula.

    Args: tuple containg parent_visits, node_visits, win_rate, player and exploration_const

    Return: a float of the calculated UCB1 value
    """
    exploration_term = math.sqrt(exploration_const * math.log(parent_visits) / node_visits)
    if player == 0: 
        return win_rate + exploration_term
    return win_rate - exploration_term


class MCTSPlay:
    """
    Wrapper class that manages the MCTS player.

    Atributtes:
        name (str): the display name of the agent.
        time_limit (float): maximum calculation time per move.
        num_workers (int): number of parallel processes to use.
        strategy (function): the strategic move evaluation function.
    """
    def __init__(self, name="MCTS", time_limit=2.0, num_workers=4):
        """Initializes the MCTSPlay instance with custom settings.

        Args: tuple containing name, time_limit and num_workers.
        """
        self.name = name
        self.time_limit = time_limit
        self.num_workers = num_workers
        self.strategy = mcts_agent(self.time_limit, self.num_workers)

    def get_move(self, position):
        """
        Evaluates the board state and returns the optimal move.
        
        Args: position.

        Return: the index of the selected play (int)
        """
        print(f"[{self.name}] Thinking for {self.time_limit} seconds using {self.num_workers} cores...")
        move = self.strategy(position)
        print(f"{self.name} chose to play: {move}")
        return move