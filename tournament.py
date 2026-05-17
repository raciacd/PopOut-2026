import time
import csv
from statistics import mean
from main import Connect4
from mcts_play import (
    MCTSPlay, MCTSConfig,
    random_rollout_policy,
    center_biased_rollout_policy,
    win_aware_rollout_policy,
    win_block_rollout_policy,
)


def play_game(config_a, config_b, max_moves=200, verbose=False):
    """
    Plays a single game between two MCTS configs. config_a's agent is player 0 and config_b's agent is player 1.
    A hard cap on the move count prevents infinite loops since the agents don't see the threefold-repetition draw option.

    Args: tuple containing config_a, config_b, max_moves and verbose.

    Return: a dict with the result (int), move count (int), per-agent time totals, per-agent iteration totals, per-agent rollout totals, the full move log (list) and a timed_out flag (bool).
    """
    agent_0 = MCTSPlay(config=config_a)
    agent_1 = MCTSPlay(config=config_b)
    agents = {0: agent_0, 1: agent_1}

    game = Connect4()
    pos = game.get_initial_position()

    # memory stored to mirror the human-game repetition counter
    history = {}
    times = {0: 0.0, 1: 0.0}
    iters = {0: 0, 1: 0}
    rollouts = {0: 0, 1: 0}
    move_count = 0
    move_log = []

    # Main loop
    while not pos.terminal and move_count < max_moves:
        history[pos.hash] = history.get(pos.hash, 0) + 1

        if verbose:
            pos.print_board(pos.legal_moves())

        current = pos.turn
        t0 = time.time()
        move = agents[current].get_move(pos)
        times[current] += time.time() - t0

        # Pull telemetry exposed by mcts_agent.strat
        last = agents[current].strategy.last_stats
        if last is not None:
            iters[current] += last['iterations']
            rollouts[current] += last['rollouts']

        move_log.append({'turn': current, 'move': move})
        pos = pos.move(move)
        move_count += 1

    # End of game
    if pos.terminal:
        result = pos.result
    else:
        result = 0 # timeout, score as a draw

    return {
        'result': result,
        'moves': move_count,
        'time_p0': times[0],
        'time_p1': times[1],
        'iters_p0': iters[0],
        'iters_p1': iters[1],
        'rollouts_p0': rollouts[0],
        'rollouts_p1': rollouts[1],
        'move_log': move_log,
        'timed_out': not pos.terminal,
    }


def run_match(config_a, config_b, num_games=4, swap_sides=True, verbose=False):
    """
    Plays a round of games between two configs, alternating who starts when swap_sides is True. Outcomes are reported from config_a's perspective.

    Args: tuple containing config_a, config_b, num_games, swap_sides and verbose.

    Return: a dict containing win/loss/draw counts, per-agent average time and iteration counts per move, and a per-game breakdown.
    """
    summary = {
        'a_name': config_a.name, 'b_name': config_b.name,
        'a_wins': 0, 'b_wins': 0, 'draws': 0, 'timeouts': 0,
        'avg_moves': 0.0,
        'a_avg_time_per_move': 0.0, 'b_avg_time_per_move': 0.0,
        'a_avg_iters_per_move': 0.0, 'b_avg_iters_per_move': 0.0,
        'games': [],
    }
    move_counts = []
    a_time_pm, b_time_pm = [], []
    a_iter_pm, b_iter_pm = [], []

    for g in range(num_games):
        a_is_p0 = (not swap_sides) or (g % 2 == 0)

        if a_is_p0:
            res = play_game(config_a, config_b, verbose=verbose)
            a_time, b_time = res['time_p0'], res['time_p1']
            a_iters, b_iters = res['iters_p0'], res['iters_p1']
            a_moves = sum(1 for m in res['move_log'] if m['turn'] == 0)
            b_moves = sum(1 for m in res['move_log'] if m['turn'] == 1)
            if res['result'] == 1: outcome = 'a'
            elif res['result'] == -1: outcome = 'b'
            else: outcome = 'draw'
        else:
            res = play_game(config_b, config_a, verbose=verbose)
            a_time, b_time = res['time_p1'], res['time_p0']
            a_iters, b_iters = res['iters_p1'], res['iters_p0']
            a_moves = sum(1 for m in res['move_log'] if m['turn'] == 1)
            b_moves = sum(1 for m in res['move_log'] if m['turn'] == 0)
            if res['result'] == -1: outcome = 'a'
            elif res['result'] == 1: outcome = 'b'
            else: outcome = 'draw'

        # Tally
        if outcome == 'a': summary['a_wins'] += 1
        elif outcome == 'b': summary['b_wins'] += 1
        else: summary['draws'] += 1
        if res['timed_out']: summary['timeouts'] += 1

        move_counts.append(res['moves'])
        if a_moves:
            a_time_pm.append(a_time / a_moves)
            a_iter_pm.append(a_iters / a_moves)
        if b_moves:
            b_time_pm.append(b_time / b_moves)
            b_iter_pm.append(b_iters / b_moves)

        summary['games'].append({
            'game': g, 'a_started': a_is_p0, 'outcome': outcome,
            'moves': res['moves'], 'timed_out': res['timed_out'],
        })
        print(f"  [{g+1}/{num_games}] {config_a.name} vs {config_b.name}: "
              f"{outcome} in {res['moves']} moves"
              + (" (TIMEOUT)" if res['timed_out'] else ""))

    summary['avg_moves'] = mean(move_counts) if move_counts else 0
    summary['a_avg_time_per_move'] = mean(a_time_pm) if a_time_pm else 0
    summary['b_avg_time_per_move'] = mean(b_time_pm) if b_time_pm else 0
    summary['a_avg_iters_per_move'] = mean(a_iter_pm) if a_iter_pm else 0
    summary['b_avg_iters_per_move'] = mean(b_iter_pm) if b_iter_pm else 0
    return summary


def round_robin(configs, num_games_per_match=4, output_csv='tournament_results.csv'):
    """
    Runs a round-robin tournament where every config plays every other config, then writes the aggregated results to a CSV file and prints a summary table.

    Args: tuple containing configs, num_games_per_match and output_csv.

    Return: a list of dicts, one per matchup, with the per-match summary stats.
    """
    rows = []
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            ca, cb = configs[i], configs[j]
            print(f"\n=== {ca.name}  vs  {cb.name} ===")
            s = run_match(ca, cb, num_games=num_games_per_match)
            rows.append({
                'a': s['a_name'], 'b': s['b_name'],
                'a_wins': s['a_wins'], 'b_wins': s['b_wins'],
                'draws': s['draws'], 'timeouts': s['timeouts'],
                'avg_moves': round(s['avg_moves'], 1),
                'a_time/move': round(s['a_avg_time_per_move'], 3),
                'b_time/move': round(s['b_avg_time_per_move'], 3),
                'a_iters/move': round(s['a_avg_iters_per_move'], 0),
                'b_iters/move': round(s['b_avg_iters_per_move'], 0),
            })

    # CSV output
    if rows:
        with open(output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\nResults written to {output_csv}")

    # Console summary
    print("\n=== SUMMARY ===")
    for r in rows:
        print(f"{r['a']:25s} vs {r['b']:25s}  "
              f"{r['a_wins']}-{r['b_wins']}-{r['draws']}  "
              f"(avg {r['avg_moves']} moves)")
    return rows


# Tournament entry point
if __name__ == "__main__":
    # Keep time_limit small (e.g. 0.5-1.0s) for tournament runs since games add up.
    T = 2.0
    W = 8

    configs = [
        MCTSConfig(name="baseline",      time_limit=T, num_workers=W,
                   exploration_const=2.0, rollout_policy=random_rollout_policy),
        MCTSConfig(name="low_explore",   time_limit=T, num_workers=W,
                   exploration_const=0.5, rollout_policy=random_rollout_policy),
        MCTSConfig(name="high_explore",  time_limit=T, num_workers=W,
                   exploration_const=5.0, rollout_policy=random_rollout_policy),
        MCTSConfig(name="center_bias",   time_limit=T, num_workers=W,
                   exploration_const=2.0, rollout_policy=center_biased_rollout_policy),
        MCTSConfig(name="win_aware",     time_limit=T, num_workers=W,
                   exploration_const=2.0, rollout_policy=win_aware_rollout_policy),
        MCTSConfig(name="win_block",     time_limit=T, num_workers=W,
                   exploration_const=2.0, rollout_policy=win_block_rollout_policy),
    ]

    round_robin(configs, num_games_per_match=40)