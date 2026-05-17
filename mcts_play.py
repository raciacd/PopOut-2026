import random
import math
import time
import os
from dataclasses import dataclass
from typing import Callable
from multiprocessing import Pool


# ============================================================
# ROLLOUT POLICIES
# ============================================================

def random_rollout_policy(state, moves):
    """
    Baseline rollout policy that selects a move uniformly at random.

    Args: tuple containing state and moves.

    Return: an int representing the chosen move.
    """
    return random.choice(moves)


def center_biased_rollout_policy(state, moves):
    """
    Rollout policy that favors central columns.

    Args: tuple containing state and moves.

    Return: an int representing the chosen move.
    """
    column_weights = [1, 2, 3, 4, 3, 2, 1]
    weights = []
    for m in moves:
        if 0 <= m <= 6:
            weights.append(column_weights[m])
        elif 7 <= m <= 13:
            weights.append(column_weights[m - 7] * 0.5) # pops weighted lower
        else: 
            weights.append(0.1)
    return random.choices(moves, weights=weights, k=1)[0]


def _is_winning_for(state_after_move, mover_turn):
    """
    Helper that checks if the player who just moved won in the resulting state.

    Args: tuple containing state_after_move and mover_turn.

    Return: a bool, True if the mover just won, False otherwise.
    """
    if not state_after_move.terminal or state_after_move.result in (None, 0):
        return False
    # result == 1 means player 0 won, result == -1 means player 1 won
    return (mover_turn == 0 and state_after_move.result == 1) or \
           (mover_turn == 1 and state_after_move.result == -1)


def win_aware_rollout_policy(state, moves):
    """
    Rollout policy that takes an immediate winning move when available, otherwise plays randomly.

    Args: tuple containing state and moves.

    Return: an int representing the chosen move.
    """
    for m in moves:
        if m == -1:
            continue
        if _is_winning_for(state.move(m), state.turn):
            return m
    return random.choice(moves)


def win_block_rollout_policy(state, moves):
    """
    Rollout policy that takes wins, avoids moves that allow the opponent to win next turn, otherwise plays randomly.
    Costs O(n^2) per step but improves sample quality.

    Args: tuple containing state and moves.

    Return: an int representing the chosen move.
    """
    # Immediate win check
    for m in moves:
        if m == -1:
            continue
        if _is_winning_for(state.move(m), state.turn):
            return m

    # Filter out moves that give the opponent an immediate win
    safe = []
    for m in moves:
        if m == -1:
            continue
        nxt = state.move(m)
        if nxt.terminal:
            safe.append(m)
            continue
        opp_wins = False
        for om in nxt.legal_moves():
            if om == -1:
                continue
            if _is_winning_for(nxt.move(om), nxt.turn):
                opp_wins = True
                break
        if not opp_wins:
            safe.append(m)

    return random.choice(safe) if safe else random.choice(moves)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class MCTSConfig:
    """
    All MCTS hyperparameters and the rollout heuristic in a single object.

    Attributes:
        name (str): Display name of this configuration.
        time_limit (float): Maximum calculation time per move, in seconds.
        num_workers (int): Number of parallel processes to use.
        exploration_const (float): UCB1 exploration constant.
        rollouts_per_expansion (int): Number of rollouts run per expansion step.
        max_rollout_depth (int): Maximum number of plays allowed in a rollout before cutoff.
        rollout_policy (Callable): Function (state, moves) -> move used during simulation.
    """
    name: str = "Default"
    time_limit: float = 2.0
    num_workers: int = 4
    exploration_const: float = 2.0
    rollouts_per_expansion: int = 10
    max_rollout_depth: int = 50
    rollout_policy: Callable = random_rollout_policy


# ============================================================
# WORKER + TREE BUILDING
# ============================================================

def mcts_worker(args):
    """
    Worker function to build an MCTS tree in a separate CPU core.

    Args: tuple containing root_state and config.

    Return: tuple containing children_stats (dict), iterations (int) and total rollouts (int).
    """
    root_state, config = args
    
    # Ensures each CPU core generates completely different random games
    random.seed(os.urandom(4))

    nodes, iterations, total_rollouts = build_mcts_tree(root_state, config)
    
    children_stats = {}
    for loc in root_state.legal_moves():
        next_state = root_state.move(loc)
        if next_state in nodes:
            w, n, _ = nodes[next_state]
            children_stats[loc] = (w, n)
            
    return children_stats, iterations, total_rollouts


def build_mcts_tree(root_state, config):
    """
    Builds the MCTS statistical tree using the hyperparameters defined in config.

    Args: tuple containing root_state and config.

    Return: tuple containing nodes (dict), iterations (int) and total_rollouts (int).
    """
    # Structure: {state: (total_reward, visits, {parent_state: parent_visits})}
    nodes = {} 
    nodes[root_state] = (0.0, 0.0, {root_state: 0})
    
    start_time = time.time()
    iterations = 0
    total_rollouts = 0 
    
    while time.time() - start_time < config.time_limit:   
        iterations += 1

        # ==========================================
        # SELECTION
        # ==========================================
        selection_path = selection_phase(nodes, root_state, config)
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
            num_runs = config.rollouts_per_expansion
            for _ in range(num_runs):
                reward += simulate_rollout(child_state, config)
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
            num_runs = config.rollouts_per_expansion
            for _ in range(num_runs):
                reward += simulate_rollout(selected_node, config)   
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


def mcts_agent(config):
    """
    Wraps the MCTS strategy, returning a move function parameterized by config.

    Args: a single MCTSConfig.

    Return: function that takes current_state and returns the optimal move.
    """
    def strat(current_state):
        """
        Runs MCTS in parallel across workers and selects the best move from the aggregated stats.

        Args: current_state.

        Return: the index of the chosen move (int).
        """
        # Prepare arguments for multiprocessing
        args = [(current_state, config) for _ in range(config.num_workers)]
        
        # Run MCTS in parallel across available CPU cores
        with Pool(processes=config.num_workers) as pool:
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

        iter_per_sec = total_iters / config.time_limit
        rollouts_per_sec = total_rollouts / config.time_limit
        
        print(f"[{config.name}] MCTS iterations: {total_iters} ({iter_per_sec:.0f} iters/sec) [Cores: {config.num_workers}]")
        print(f"[{config.name}] MCTS rollouts:   {total_rollouts} ({rollouts_per_sec:.0f} rollouts/sec) [Cores: {config.num_workers}]")
        
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

        # Expose last-move stats so the tournament runner can collect telemetry
        strat.last_stats = {
            'iterations': total_iters,
            'rollouts': total_rollouts,
            'merged_stats': merged_stats,
            'chosen_move': next_best_move,
        }
        return next_best_move

    strat.last_stats = None
    return strat
        

def simulate_rollout(state, config):
    """
    Executes a rollout from the given state using the policy in config, until the game ends or max_rollout_depth is reached.

    Args: tuple containing state and config.

    Return: float for standardized reward: 1.0 for player 0 win, 0.0 for player 1 win, and 0.5 for a draw or cutoff.
    """
    current_state = state
    depth = 0
    
    while not current_state.terminal and depth < config.max_rollout_depth:
        moves = current_state.legal_moves()
        loc = config.rollout_policy(current_state, moves)
        current_state = current_state.move(loc) 
        depth += 1
        
    if not current_state.terminal:
        return 0.5 

    if current_state.result == 1: return 1.0
    elif current_state.result == -1: return 0.0
    return 0.5
        

def selection_phase(nodes, root_state, config):
    """
    Navigates the tree using the UCB1 policy with the exploration constant defined in config.

    Args: tuple containing nodes, root_state and config.

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
                        
            score = calculate_ucb_score(
                nodes[current_node][1],
                temp_parent_n_count[current_node],
                temp_w / temp_visits,
                next_player,
                config.exploration_const,
            )
            
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

    Args: tuple containing parent_visits, node_visits, win_rate, player and exploration_const.

    Return: a float of the calculated UCB1 value.
    """
    exploration_term = math.sqrt(exploration_const * math.log(parent_visits) / node_visits)
    if player == 0: 
        return win_rate + exploration_term
    return win_rate - exploration_term


class MCTSPlay:
    """
    Wrapper class that manages the MCTS player.

    Atributtes:
        config (MCTSConfig): the full configuration object for the agent.
        name (str): the display name of the agent.
        time_limit (float): maximum calculation time per move.
        num_workers (int): number of parallel processes to use.
        strategy (function): the strategic move evaluation function.
    """
    def __init__(self, name="MCTS", time_limit=2.0, num_workers=4, config=None):
        """
        Initializes the MCTSPlay instance. If a config is provided it takes precedence; otherwise one is built from the legacy kwargs to preserve backwards compatibility.

        Args: tuple containing name, time_limit, num_workers and config.
        """
        if config is None:
            config = MCTSConfig(name=name, time_limit=time_limit, num_workers=num_workers)
        self.config = config
        self.name = config.name
        self.time_limit = config.time_limit
        self.num_workers = config.num_workers
        self.strategy = mcts_agent(config)

    def get_move(self, position):
        """
        Evaluates the board state and returns the optimal move.

        Args: position.

        Return: the index of the selected play (int).
        """
        print(f"[{self.name}] Thinking for {self.config.time_limit} seconds using {self.config.num_workers} cores...")
        move = self.strategy(position)
        print(f"{self.name} chose to play: {move}")
        return move