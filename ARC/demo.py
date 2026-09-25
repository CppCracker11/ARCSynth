"""
Usage:
  python demo.py                     # Default demo (A* on hard_01)
  python demo.py --task med_01       # Run on medium task
  python demo.py --algo bfs          # Run with BFS search
  python demo.py --algo weighted     # Run with Weighted A* search
  python demo.py --algo rl           # Run with Tabular Q-Learning
  python demo.py --list              # List all 24 benchmark tasks
"""

import argparse
import sys
import time
import numpy as np

from grids import load_tasks
from search_astar import solve_astar
from search_bfs import solve_bfs, replay_program
from search_weighted import solve_weighted_astar
from rl_agent import QLearningAgent


def render_grid_lines(grid, raw=False):
    """Convert a 2D numpy grid into formatted text lines."""
    lines = []
    for row in grid:
        if raw:
            line_str = " ".join(str(val) for val in row)
        else:
            line_str = " ".join(str(val) if val != 0 else "." for val in row)
        lines.append(line_str)
    return lines


def format_side_by_side(g1, label1, g2, label2, raw=False, indent="  "):
    """Format two grids side-by-side with aligned labels and divider."""
    lines1 = render_grid_lines(g1, raw=raw)
    lines2 = render_grid_lines(g2, raw=raw)
    w1 = max(len(label1), max((len(l) for l in lines1), default=0))
    w2 = max(len(label2), max((len(l) for l in lines2), default=0))

    header_lbl = f"{indent}{label1:<{w1}}     -->     {label2:<{w2}}"
    header_div = f"{indent}{'-' * w1}             {'-' * w2}"
    out = [header_lbl, header_div]

    max_h = max(len(lines1), len(lines2))
    for i in range(max_h):
        l1 = lines1[i] if i < len(lines1) else ""
        l2 = lines2[i] if i < len(lines2) else ""
        out.append(f"{indent}{l1:<{w1}}             {l2:<{w2}}")

    return "\n".join(out)


def print_banner():
    """Print the project title and architectural summary."""
    print("=" * 80)
    print("           ARCSynth: Search-Based Program Synthesis Engine")
    print("              ARC-AGI Grid Puzzle Solving via Search & DSL")
    print("=" * 80)
    print("Architecture Overview:")
    print("  * Paradigm   : State-space search over Domain-Specific Language (DSL) primitives")
    print("  * Core Stack : Pure Python 3 + NumPy (Zero neural nets, zero GPU dependencies)")
    print("  * Guarantee  : Multi-pair parallel verification with held-out generalization")
    print("=" * 80)


def list_benchmark_tasks(tasks):
    """Print all available benchmark tasks grouped by difficulty tier."""
    print("\nAvailable Benchmark Tasks in 'benchmark_tasks.json':")
    print("-" * 80)
    print(f"{'Task ID':<12} {'Tier':<10} {'Prims':<8} {'Ground Truth Program'}")
    print("-" * 80)
    for tid, tinfo in tasks.items():
        tier = tinfo.get("tier", "unknown")
        prog = tinfo.get("program", [])
        prog_str = " -> ".join(prog) if prog else "N/A"
        print(f"{tid:<12} {tier:<10} {len(prog):<8} {prog_str}")
    print("-" * 80)
    print(f"Total: {len(tasks)} tasks available.\n")


def run_demo(task_id="hard_01", algo_name="astar", tasks_path="benchmark_tasks.json", raw=False):
    """Run the complete faculty demo for the selected task and algorithm."""
    # 1. Title Banner
    print_banner()

    # Load tasks
    try:
        tasks = load_tasks(tasks_path)
    except Exception as e:
        print(f"\n[ERROR] Failed to load '{tasks_path}': {e}")
        sys.exit(1)

    if task_id not in tasks:
        print(f"\n[ERROR] Task '{task_id}' not found in '{tasks_path}'.")
        print(f"Available tasks: {list(tasks.keys())}")
        sys.exit(1)

    task = tasks[task_id]
    tier = task.get("tier", "N/A").upper()
    train_pairs = task["train"]
    test_pair = task["test"][0]

    # 2. Benchmark Task Header
    print(f"\n[1] DEMONSTRATION TASK: {task_id} (Tier: {tier})")
    print("-" * 80)
    if "program" in task:
        gt_prog = " -> ".join(task["program"])
        print(f"  Synthetic Ground Truth : {gt_prog} ({len(task['program'])} primitive operations)")
    print(f"  Training Set Size      : {len(train_pairs)} input/output grid pairs")
    print(f"  Held-out Test Set Size : 1 unseen input/output grid pair")
    print("  Grid Legend            : '.' = 0 (black background), digits 1-9 = ARC color palette")

    # 3. Training Pairs Display
    print(f"\n[2] TRAINING PAIRS (State Space Specification)")
    print("-" * 80)
    for idx, pair in enumerate(train_pairs, 1):
        in_shape = f"{pair['input'].shape[0]}x{pair['input'].shape[1]}"
        out_shape = f"{pair['output'].shape[0]}x{pair['output'].shape[1]}"
        lbl_in = f"Train {idx} Input ({in_shape})"
        lbl_out = f"Train {idx} Target Output ({out_shape})"
        print(f"\n  Pair {idx}:")
        print(format_side_by_side(pair["input"], lbl_in, pair["output"], lbl_out, raw=raw, indent="    "))

    # 4. Search Algorithm Selection
    algo_key = algo_name.lower().strip()
    print(f"\n[3] SEARCH ALGORITHM CONFIGURATION")
    print("-" * 80)

    if algo_key in ("astar", "a*"):
        algo_label = "A* Search (Heuristic-Guided Priority Queue)"
        print(f"  Algorithm : {algo_label}")
        print(f"  Heuristic : f(n) = depth(n) + sum(Hamming pixel distance + shape penalty)")
        print(f"  Strategy  : Finds shortest provably correct program with optimal expansions")
        res = solve_astar(train_pairs, max_depth=3)

    elif algo_key == "bfs":
        algo_label = "Breadth-First Search (BFS)"
        print(f"  Algorithm : {algo_label}")
        print(f"  Strategy  : Exhaustive level-by-level exploration; guarantees minimal depth")
        res = solve_bfs(train_pairs, max_depth=3)

    elif algo_key in ("weighted", "weighted_astar", "wastar"):
        algo_label = "Weighted A* Search (w=2.0)"
        print(f"  Algorithm : {algo_label}")
        print(f"  Heuristic : f(n) = depth(n) + 2.0 * heuristic(n)")
        print(f"  Strategy  : Accelerated greedy goal approach, trading optimality for speed")
        res = solve_weighted_astar(train_pairs, weight=2.0, max_depth=3)

    elif algo_key in ("rl", "qlearning", "q_learning"):
        algo_label = "Tabular Q-Learning (Reinforcement Learning)"
        print(f"  Algorithm : {algo_label}")
        print(f"  Strategy  : Feature-based Q(s,a) lookup, synthesizing solutions in single rollout")
        print(f"  Training  : Pre-training agent on easy/medium tasks...")
        agent = QLearningAgent(alpha=0.4, gamma=0.8)
        train_tasks = {k: v for k, v in tasks.items() if v.get("tier") in ["easy", "medium"]}
        agent.train(train_tasks, episodes=400)
        res = agent.solve(train_pairs, max_steps=3)

    else:
        print(f"[ERROR] Unknown algorithm '{algo_name}'. Choose from: astar, bfs, weighted, rl")
        sys.exit(1)

    elapsed_ms = res["time_taken"] * 1000.0
    solved = res["solved"]
    program = res.get("program") or []
    depth = len(program) if solved else 0
    nodes = res["nodes_expanded"]

    # 5. Program Synthesis Result
    print(f"\n[4] SYNTHESIS RESULTS & DISCOVERED PROGRAM")
    print("-" * 80)
    if solved:
        print(f"  Status           : SUCCESS (Solution found!)")
        print(f"  Discovered DSL   : {' -> '.join(str(a) for a in program)}")
        print(f"  Program Length   : {depth} primitive operation(s)")
        print(f"  Search Depth     : {depth}")
        print(f"  Nodes Expanded   : {nodes}")
        print(f"  Synthesis Time   : {elapsed_ms:.2f} ms ({res['time_taken']:.5f} seconds)")
        print("\n  Step-by-Step Primitive Breakdown:")
        for s_idx, act in enumerate(program, 1):
            print(f"    Step {s_idx}: {act.name}")
    else:
        print(f"  Status           : FAILED (No program found within depth/node limits)")
        print(f"  Nodes Expanded   : {nodes}")
        print(f"  Synthesis Time   : {elapsed_ms:.2f} ms")
        return

    # 6. Test Input and Step-by-Step Replay
    print(f"\n[5] GENERALIZATION ON UNSEEN HELD-OUT TEST INPUT")
    print("-" * 80)
    test_in = test_pair["input"]
    test_out = test_pair["output"]
    t_in_shape = f"{test_in.shape[0]}x{test_in.shape[1]}"
    t_out_shape = f"{test_out.shape[0]}x{test_out.shape[1]}"

    print(f"  Held-out Test Input Grid ({t_in_shape}):")
    for row_str in render_grid_lines(test_in, raw=raw):
        print(f"    {row_str}")

    print(f"\n  Executing Synthesized Program Step-by-Step on Test Input:")
    cur_grid = test_in.copy()
    for s_idx, act in enumerate(program, 1):
        prev_shape = f"{cur_grid.shape[0]}x{cur_grid.shape[1]}"
        cur_grid = act(cur_grid)
        curr_shape = f"{cur_grid.shape[0]}x{cur_grid.shape[1]}"
        print(f"\n  --> Apply Step {s_idx}: {act.name}  (shape: {prev_shape} -> {curr_shape})")
        for row_str in render_grid_lines(cur_grid, raw=raw):
            print(f"      {row_str}")

    predicted_out = cur_grid

    # 7. Verification: Predicted Output vs Ground Truth
    print(f"\n[6] TEST VERIFICATION & GENERALIZATION RESULT")
    print("-" * 80)
    pred_shape = f"{predicted_out.shape[0]}x{predicted_out.shape[1]}"
    lbl_pred = f"Predicted Output ({pred_shape})"
    lbl_exp = f"Expected Ground Truth ({t_out_shape})"
    print(format_side_by_side(predicted_out, lbl_pred, test_out, lbl_exp, raw=raw, indent="  "))

    shape_matches = predicted_out.shape == test_out.shape
    exact_match = shape_matches and bool(np.array_equal(predicted_out, test_out))

    if shape_matches:
        pixel_errors = int(np.count_nonzero(predicted_out != test_out))
    else:
        pixel_errors = -1

    print("\n  Validation Assessment:")
    if exact_match:
        print(f"    * Test Result   : PASS (100% Exact Match)")
        print(f"    * Pixel Errors  : 0 / {test_out.size}")
        print(f"    * Generalization: VERIFIED (Discovered program perfectly generalizes to unseen test pair)")
    else:
        print(f"    * Test Result   : FAIL")
        print(f"    * Pixel Errors  : {pixel_errors}")

    # 8. Performance Summary Table
    print(f"\n[7] PERFORMANCE SUMMARY METRICS")
    print("=" * 80)
    print(f"  {'Metric':<25} | {'Value'}")
    print("  " + "-" * 25 + "-+-" + "-" * 45)
    print(f"  {'Benchmark Task':<25} | {task_id} ({tier})")
    print(f"  {'Algorithm':<25} | {algo_label}")
    print(f"  {'Training Status':<25} | {'SOLVED' if solved else 'FAILED'}")
    print(f"  {'Test Generalization':<25} | {'PASSED (Exact Match)' if exact_match else 'FAILED'}")
    print(f"  {'Synthesized Program':<25} | {' -> '.join(str(a) for a in program)}")
    print(f"  {'Search Depth':<25} | {depth}")
    print(f"  {'Nodes Expanded':<25} | {nodes}")
    print(f"  {'Execution Time':<25} | {elapsed_ms:.2f} ms ({res['time_taken']:.5f} s)")
    print("=" * 80)
    print("\nDemo completed successfully.\n")


def main():
    parser = argparse.ArgumentParser(
        description="ARCSynth Faculty Demonstration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python demo.py                     # Run default demo (A* on hard_01)
  python demo.py --task med_01       # Run on medium benchmark task
  python demo.py --task easy_06      # Run on cropping task
  python demo.py --algo bfs          # Solve using Breadth-First Search
  python demo.py --algo weighted     # Solve using Weighted A*
  python demo.py --algo rl           # Solve using Tabular Q-Learning
  python demo.py --list              # List all 24 benchmark tasks
        """,
    )
    parser.add_argument(
        "--task",
        "-t",
        type=str,
        default="hard_01",
        help="Task ID to demonstrate (default: 'hard_01')",
    )
    parser.add_argument(
        "--algo",
        "-a",
        type=str,
        default="astar",
        choices=["astar", "bfs", "weighted", "rl"],
        help="Search algorithm to use: 'astar' (default), 'bfs', 'weighted', 'rl'",
    )
    parser.add_argument(
        "--tasks-path",
        type=str,
        default="benchmark_tasks.json",
        help="Path to benchmark tasks JSON file (default: 'benchmark_tasks.json')",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Display raw integers (0) instead of '.' for background pixels",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all benchmark tasks and exit",
    )

    args = parser.parse_args()

    if args.list:
        tasks = load_tasks(args.tasks_path)
        list_benchmark_tasks(tasks)
        return

    run_demo(
        task_id=args.task,
        algo_name=args.algo,
        tasks_path=args.tasks_path,
        raw=args.raw,
    )


if __name__ == "__main__":
    main()
