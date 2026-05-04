import random
import math
import time

def build_mcts_tree(root_state, time_limit):
    """Builds the MCTS statistical tree given a time limit."""
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
        # Traverses the tree using the UCB1 formula until a leaf node is found
        selection_path = selection_phase(nodes, root_state)
        selected_node = selection_path[-1]
        
        # Extracts stats from the selected node
        _, visits, _ = nodes[selected_node]
        
        # ==========================================
        # EXPANSION
        # ==========================================
        # If the node has been visited and the game is not over, expand its children
        if visits > 0 and not selected_node.terminal:
            legal_moves = selected_node.legal_moves()
            for loc in legal_moves:
                new_state = selected_node.move(loc)
                
                # If the state does not exist in the tree yet, initialize it
                if new_state not in nodes:
                    nodes[new_state] = (0.0, 0.0, {selected_node: 0})
                    
            # Randomly chooses a newly expanded child to investigate
            loc = random.choice(legal_moves)
            child_state = selected_node.move(loc)
            
            # ==========================================
            # SIMULATION / ROLLOUT
            # ==========================================
            # Plays random games to the end from the new child
            reward = 0
            num_runs = 10
            for _ in range(num_runs):
                reward += simulate_rollout(child_state)
            total_rollouts += num_runs
            
            # Updates the parent dictionaries of the new child
            w, n, parent_n_dict = nodes[child_state]
            if selected_node not in parent_n_dict:
                parent_n_dict[selected_node] = 0
            parent_n_dict[selected_node] += 1
            nodes[child_state] = (w + reward, n + num_runs, parent_n_dict)

        else:
            # If the leaf node has never been visited or the game is over
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
        # Propagates up the traversed path updating wins and visits of all nodes
        parent = root_state
        for state_in_path in selection_path:
            w, n, parent_n_dict = nodes[state_in_path]
            parent_n_dict[parent] += num_runs
            nodes[state_in_path] = (w + reward, n + num_runs, parent_n_dict)
            parent = state_in_path
            
    return nodes, iterations, total_rollouts


def mcts_agent(time_limit):
    """Wraps the MCTS strategy, returning a move function."""
    def strat(current_state):
        nodes, iterations, total_rollouts = build_mcts_tree(current_state, time_limit)

        iter_per_sec = iterations / time_limit
        rollouts_per_sec = total_rollouts / time_limit
        
        print(f"Number of MCTS iterations: {iterations} ({iter_per_sec:.0f} iters/sec)")
        print(f"Number of MCTS rollouts:   {total_rollouts} ({rollouts_per_sec:.0f} rollouts/sec)")
        
        player = current_state.turn
        best_score = float('-inf') if player == 0 else float('inf')
        next_best_move = None
   
        for loc in current_state.legal_moves():
            next_state = current_state.move(loc)
            
            if next_state not in nodes:
                score = 0.0
            else:
                w, n, _ = nodes[next_state]
                score = 0.0 if n == 0 else w / n
            
            if score < best_score and player == 1: # Player 1 minimizes
                best_score = score
                next_best_move = loc
            elif score > best_score and player == 0: # Player 0 maximizes
                best_score = score
                next_best_move = loc
                
        return next_best_move
    return strat
        

def simulate_rollout(state, max_depth=50):
    """Phase 3: Plays random moves until the end or depth limit."""
    current_state = state
    depth = 0
    
    # Prevents infinite loops in PopOut games
    while not current_state.terminal and depth < max_depth:
        moves = current_state.legal_moves()
        loc = random.choice(moves)
        current_state = current_state.move(loc) 
        depth += 1
        
    # Statistical draw if depth limit is reached
    if not current_state.terminal:
        return 0.5 

    # Reward standardization for UCB1: (1.0 = Win P0) | (0.0 = Win P1) | (0.5 = Draw)
    if current_state.result == 1: return 1.0
    elif current_state.result == -1: return 0.0
    return 0.5
        

def selection_phase(nodes, root_state):
    """Phase 1: Navigates the tree using the UCB1 policy."""
    current_node = root_state
    path = []
    visited = set() # NEW: Cycle detector for PopOut infinite loops
    
    while True:
        # If we step on the same state during this specific descent, we found a cycle!
        # Break the loop immediately to simulate and avoid freezing.
        if current_node in visited:
            return path
        
        visited.add(current_node)
        
        _, visits, _ = nodes[current_node]
        path.append(current_node)
        
        # If a leaf node (never visited) is found, stop the search
        if visits == 0:
            return path
        
        legal_moves = current_node.legal_moves()
        next_player = current_node.turn
        
        best_score = float('-inf') if next_player == 0 else float('inf')
        next_best_node = None
        
        for loc in legal_moves:
            result_state = current_node.move(loc)
            
            # If the child hasn't been created in the tree yet, this branch is a leaf
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
    """Upper Confidence Bound (UCB1) formula that balances exploration and exploitation."""
    exploration_term = math.sqrt(exploration_const * math.log(parent_visits) / node_visits)
    if player == 0: 
        return win_rate + exploration_term
    return win_rate - exploration_term


class MCTSPlay:
    def __init__(self, name="MCTS", time_limit=2.0):
        self.name = name
        self.time_limit = time_limit
        self.strategy = mcts_agent(self.time_limit)

    def get_move(self, position):
        print(f"[{self.name}] Thinking for {self.time_limit} seconds...")
        move = self.strategy(position)
        print(f"{self.name} chose to play: {move}")
        return move