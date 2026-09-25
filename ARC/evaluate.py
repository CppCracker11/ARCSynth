import csv
import time
import numpy as np
from grids import load_tasks
from search_bfs import solve_bfs, replay_program
from search_astar import solve_astar
from search_weighted import solve_weighted_astar
from rl_agent import QLearningAgent


def evaluate_benchmark(tasks_path="benchmark_tasks.json", csv_out="results.csv"):
    tasks = load_tasks(tasks_path)
    train_tasks = {k: v for k, v in tasks.items() if v["tier"] in ["easy", "medium"]}

    # Train RL agent
    print("Training Q-Learning agent on easy & medium tasks...")
    rl_agent = QLearningAgent(alpha=0.4, gamma=0.8)
    rl_agent.train(train_tasks, episodes=600)
    print("Training complete.\n")

    methods = [
        ("BFS", lambda tp: solve_bfs(tp, max_depth=3)),
        ("A*", lambda tp: solve_astar(tp, max_depth=3)),
        ("Weighted A* (w=2.0)", lambda tp: solve_weighted_astar(tp, weight=2.0, max_depth=3)),
        ("Q-Learning RL", lambda tp: rl_agent.solve(tp, max_steps=3)),
    ]

    er = []

    print(f"{'Task':<10} {'Tier':<8} {'Method':<20} {'Train':<7} {'Test':<7} {'Nodes':<8} {'Time (ms)':<10} {'Program'}")
    print("-" * 90)

    for task_id, task in tasks.items():
        tier = task["tier"]
        train_pairs = task["train"]
        test_in = task["test"][0]["input"]
        test_out = task["test"][0]["output"]

        for m_name, m_fn in methods:
            res = m_fn(train_pairs)
            solved_train = res["solved"]
            prog = res["program"]
            prog_str = " -> ".join(str(a) for a in prog) if prog else "None"
            prog_len = len(prog) if prog else 0

            solved_test = False
            if solved_train and prog is not None:
                try:
                    pred = replay_program(test_in, prog)
                    solved_test = bool(np.array_equal(pred, test_out))
                except Exception:
                    solved_test = False

            t_ms = res["time_taken"] * 1000.0
            nodes = res["nodes_expanded"]

            er.append({
                "task_id": task_id,
                "tier": tier,
                "method": m_name,
                "train_solved": solved_train,
                "test_solved": solved_test,
                "program_length": prog_len,
                "nodes_expanded": nodes,
                "time_ms": round(t_ms, 3),
                "program": prog_str,
            })

            train_mark = "OK" if solved_train else "FAIL"
            test_mark = "OK" if solved_test else "FAIL"
            print(f"{task_id:<10} {tier:<8} {m_name:<20} {train_mark:<7} {test_mark:<7} {nodes:<8} {t_ms:<10.2f} {prog_str}")

    # Write to CSV
    fields = [
        "task_id",
        "tier",
        "method",
        "train_solved",
        "test_solved",
        "program_length",
        "nodes_expanded",
        "time_ms",
        "program",
    ]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(er)

    print(f"\nSaved detailed evaluation results to {csv_out}")

    # Summarize per tier and method
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY BY TIER AND SEARCH METHOD")
    print("=" * 80)
    print(f"{'Tier':<10} {'Method':<20} {'Train Solve':<13} {'Test Solve':<12} {'Avg Nodes':<12} {'Avg Time (ms)'}")
    print("-" * 80)

    tiers = ["easy", "medium", "hard", "all"]
    method_names = [m[0] for m in methods]

    for tier in tiers:
        for m_name in method_names:
            if tier == "all":
                subset = [r for r in er if r["method"] == m_name]
            else:
                subset = [r for r in er if r["method"] == m_name and r["tier"] == tier]

            if not subset:
                continue

            n_tasks = len(subset)
            train_ok = sum(1 for r in subset if r["train_solved"])
            test_ok = sum(1 for r in subset if r["test_solved"])
            avg_nodes = np.mean([r["nodes_expanded"] for r in subset])
            avg_time = np.mean([r["time_ms"] for r in subset])

            train_pct = f"{train_ok}/{n_tasks} ({train_ok/n_tasks*100:.1f}%)"
            test_pct = f"{test_ok}/{n_tasks} ({test_ok/n_tasks*100:.1f}%)"

            print(f"{tier:<10} {m_name:<20} {train_pct:<13} {test_pct:<12} {avg_nodes:<12.1f} {avg_time:.2f}")
        print("-" * 80)

    return er
    """
    er = evaluation result records
    """


if __name__ == "__main__":
    evaluate_benchmark()
