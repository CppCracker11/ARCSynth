from pathlib import Path
import csv
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from grids import load_tasks
from search_bfs import solve_bfs, replay_program
from search_astar import solve_astar
from search_weighted import solve_weighted_astar
from rl_agent import QLearningAgent


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
TASKS_PATH = ROOT / "benchmark_tasks.json"
RESULTS_PATH = ROOT / "results.csv"


# ============================================================
# Application
# ============================================================

app = FastAPI(
    title="ARCSynth API",
    description="API for the ARCSynth search-based program synthesis engine.",
    version="1.0.0",
)


# Allow the Vite development server to communicate with Python.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Data models
# ============================================================

class SynthesisRequest(BaseModel):
    task_id: str
    algorithm: str = Field(
        default="astar",
        pattern="^(astar|bfs|weighted|rl)$",
    )
    max_depth: int = Field(default=3, ge=1, le=8)


# ============================================================
# Helpers
# ============================================================

def json_grid(grid):
    """Convert a NumPy grid into JSON-safe nested lists."""
    return np.asarray(grid, dtype=np.int32).tolist()


def json_pair(pair):
    return {
        "input": json_grid(pair["input"]),
        "output": json_grid(pair["output"]),
    }


def action_name(action):
    """Return the readable name of a DSL action."""
    return getattr(action, "name", str(action))


def build_execution_steps(test_input, program):
    """
    Execute the discovered DSL program on the test input and
    record every actual intermediate grid.

    IMPORTANT:
    These states come directly from the existing Python DSL.
    The frontend never recreates the transformations itself.
    """
    current = np.asarray(test_input, dtype=np.int32).copy()

    steps = [
        {
            "operation": "Original test input",
            "grid": json_grid(current),
            "note": "Step 0 · initial state",
        }
    ]

    for index, action in enumerate(program, start=1):
        current = action(current)

        steps.append(
            {
                "operation": action_name(action),
                "grid": json_grid(current),
                "note": f"Step {index} · actual DSL transformation",
            }
        )

    return steps, current


def solve_task(task, algorithm, max_depth):
    """
    Run one of the existing ARCSynth search/learning methods.

    The underlying implementation is unchanged.
    """
    train_pairs = task["train"]

    if algorithm == "astar":
        return solve_astar(
            train_pairs,
            max_depth=max_depth,
        )

    if algorithm == "bfs":
        return solve_bfs(
            train_pairs,
            max_depth=max_depth,
        )

    if algorithm == "weighted":
        return solve_weighted_astar(
            train_pairs,
            weight=2.0,
            max_depth=max_depth,
        )

    if algorithm == "rl":
        # Train a fresh RL agent for this API request.
        # This keeps the implementation faithful to the existing
        # RL experiment, although A*/BFS are faster for interactive use.
        all_tasks = load_tasks(str(TASKS_PATH))

        train_tasks = {
            key: value
            for key, value in all_tasks.items()
            if value["tier"] in ["easy", "medium"]
        }

        agent = QLearningAgent(alpha=0.4, gamma=0.8)
        agent.train(
            train_tasks,
            episodes=600,
            seed=42,
        )

        return agent.solve(
            train_pairs,
            max_steps=max_depth,
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


# ============================================================
# Routes
# ============================================================

@app.get("/api/health")
def health():
    """Simple API health check."""
    return {
        "status": "ok",
        "service": "ARCSynth API",
        "version": "1.0.0",
    }


@app.get("/api/tasks")
def get_tasks():
    """
    Return all benchmark tasks in frontend-friendly JSON format.
    """
    if not TASKS_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="benchmark_tasks.json was not found.",
        )

    tasks = load_tasks(str(TASKS_PATH))

    output = []

    for task_id, task in tasks.items():
        output.append(
            {
                "id": task_id,
                "tier": task.get("tier", "unknown"),
                "train": [
                    json_pair(pair)
                    for pair in task["train"]
                ],
                "test": [
                    json_pair(pair)
                    for pair in task["test"]
                ],
            }
        )

    return output


@app.post("/api/synthesize")
def synthesize(request: SynthesisRequest):
    """
    Run the selected ARCSynth algorithm on a benchmark task.

    Returns:
    - discovered program
    - actual test prediction
    - every intermediate grid state
    - search statistics
    """
    tasks = load_tasks(str(TASKS_PATH))

    if request.task_id not in tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{request.task_id}' was not found.",
        )

    task = tasks[request.task_id]

    started = time.perf_counter()

    try:
        result = solve_task(
            task,
            request.algorithm,
            request.max_depth,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {exc}",
        ) from exc

    server_time_ms = (time.perf_counter() - started) * 1000.0

    program = result.get("program")

    if not result.get("solved") or program is None:
        return {
            "success": False,
            "taskId": request.task_id,
            "algorithm": request.algorithm,
            "program": [],
            "steps": [],
            "testInput": json_grid(task["test"][0]["input"]),
            "predictedOutput": [],
            "expectedOutput": json_grid(task["test"][0]["output"]),
            "nodesExpanded": int(result.get("nodes_expanded", 0)),
            "timeMs": round(
                result.get("time_taken", server_time_ms / 1000.0) * 1000.0,
                3,
            ),
            "depth": 0,
            "searchTrace": [],
            "mode": "live",
            "error": "No solution found within the configured search limit.",
        }

    # --------------------------------------------------------
    # Execute the discovered program on the unseen test input.
    # --------------------------------------------------------

    test_input = task["test"][0]["input"]
    expected_output = task["test"][0]["output"]

    steps, predicted = build_execution_steps(
        test_input,
        program,
    )

    solved_test = bool(
        np.array_equal(predicted, expected_output)
    )

    program_names = [
        action_name(action)
        for action in program
    ]

    # A lightweight solution-path trace.
    #
    # This is intentionally NOT presented as the complete search
    # tree because the current search implementations do not expose
    # every expanded node.
    search_trace = []

    previous_id = None

    for index, operation in enumerate(program_names):
        node_id = f"step-{index}"

        search_trace.append(
            {
                "id": node_id,
                "label": operation,
                "depth": index + 1,
                "status": (
                    "goal"
                    if index == len(program_names) - 1
                    else "expanded"
                ),
                "parentId": previous_id,
            }
        )

        previous_id = node_id

    return {
        "success": True,
        "taskId": request.task_id,
        "algorithm": request.algorithm,
        "program": program_names,
        "steps": steps,
        "testInput": json_grid(test_input),
        "predictedOutput": json_grid(predicted),
        "expectedOutput": json_grid(expected_output),
        "nodesExpanded": int(result.get("nodes_expanded", 0)),
        "timeMs": round(
            result.get("time_taken", server_time_ms / 1000.0) * 1000.0,
            3,
        ),
        "depth": len(program_names),
        "searchTrace": search_trace,
        "mode": "live",
        "testSolved": solved_test,
    }


@app.get("/api/benchmark")
def get_benchmark():
    """
    Aggregate the existing results.csv into the format expected
    by the frontend.
    """
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "results.csv was not found. "
                "Run evaluate.py first."
            ),
        )

    rows = []

    with RESULTS_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(row)

    method_map = {
        "BFS": "BFS",
        "A*": "A*",
        "Weighted A* (w=2.0)": "Weighted A*",
        "Q-Learning RL": "Q-Learning",
    }

    output = []

    for source_name, display_name in method_map.items():
        subset = [
            row
            for row in rows
            if row["method"] == source_name
        ]

        if not subset:
            continue

        total = len(subset)

        train_solved = sum(
            row["train_solved"].lower() == "true"
            for row in subset
        )

        test_solved = sum(
            row["test_solved"].lower() == "true"
            for row in subset
        )

        avg_nodes = (
            sum(float(row["nodes_expanded"]) for row in subset)
            / total
        )

        avg_time = (
            sum(float(row["time_ms"]) for row in subset)
            / total
        )

        output.append(
            {
                "algorithm": display_name,
                "solvedTrain": train_solved,
                "totalTrain": total,
                "solvedTest": test_solved,
                "totalTest": total,
                "avgNodes": round(avg_nodes, 2),
                "avgTimeMs": round(avg_time, 3),
            }
        )

    return output