import type { BenchmarkRow, SynthesisResult, Task } from "../types";

const easy01Input = [
  [0, 0, 2, 0],
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [0, 0, 2, 3],
];

const easy01Output = [
  [0, 0, 0, 0],
  [0, 0, 0, 0],
  [2, 0, 0, 2],
  [3, 0, 0, 0],
];

export const mockTasks: Task[] = [
  {
    id: "easy_01",
    tier: "easy",
    train: [
      {
        input: [
          [0,0,2,0],
          [0,0,0,0],
          [0,1,0,1],
          [0,0,0,0],
        ],
        output: [
          [0,0,0,0],
          [0,1,0,0],
          [0,0,0,2],
          [0,1,0,0],
        ],
      },
      {
        input: [
          [0,0,0,2],
          [0,0,0,0],
          [0,0,0,3],
          [0,2,0,0],
        ],
        output: [
          [0,0,0,0],
          [2,0,0,0],
          [0,0,0,0],
          [0,3,0,2],
        ],
      },
    ],
    test: [{ input: easy01Input, output: easy01Output }],
  },
];

export const mockResult: SynthesisResult = {
  success: true,
  taskId: "easy_01",
  algorithm: "astar",
  program: ["rotate_90"],
  steps: [
    { operation: "Original test input", grid: easy01Input, note: "Step 0 · initial state" },
    { operation: "rotate_90", grid: easy01Output, note: "Step 1 · synthesized operation" },
  ],
  testInput: easy01Input,
  predictedOutput: easy01Output,
  expectedOutput: easy01Output,
  nodesExpanded: 5,
  timeMs: 1.4,
  depth: 1,
  searchTrace: [
    { id: "root", label: "initial", depth: 0, heuristic: 2, status: "expanded" },
    { id: "rotate", label: "rotate_90", depth: 1, heuristic: 0, status: "goal", parentId: "root" },
    { id: "flip", label: "flip_h", depth: 1, heuristic: 4, status: "candidate", parentId: "root" },
    { id: "transpose", label: "transpose", depth: 1, heuristic: 5, status: "candidate", parentId: "root" },
  ],
  mode: "demo",
};

export const mockBenchmark: BenchmarkRow[] = [
  { algorithm: "BFS", solvedTrain: 24, totalTrain: 24, solvedTest: 24, totalTest: 24, avgNodes: 9.4, avgTimeMs: 3.69 },
  { algorithm: "A*", solvedTrain: 24, totalTrain: 24, solvedTest: 24, totalTest: 24, avgNodes: 5.5, avgTimeMs: 1.45 },
  { algorithm: "Weighted A*", solvedTrain: 24, totalTrain: 24, solvedTest: 24, totalTest: 24, avgNodes: 7.4, avgTimeMs: 1.35 },
  { algorithm: "Q-Learning", solvedTrain: 18, totalTrain: 24, solvedTest: 16, totalTest: 24, avgNodes: 2.1, avgTimeMs: 1.58 },
];