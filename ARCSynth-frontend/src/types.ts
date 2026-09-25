export type Grid = number[][];

export type Algorithm = "astar" | "bfs" | "weighted" | "rl";

export interface Pair {
  input: Grid;
  output: Grid;
}

export interface Task {
  id: string;
  tier: "easy" | "medium" | "hard";
  train: Pair[];
  test: Pair[];
}

export interface Step {
  operation: string;
  grid: Grid;
  note?: string;
}

export interface SynthesisResult {
  success: boolean;
  taskId: string;
  algorithm: Algorithm;
  program: string[];
  steps: Step[];
  testInput: Grid;
  predictedOutput: Grid;
  expectedOutput?: Grid;
  nodesExpanded: number;
  timeMs: number;
  depth: number;
  searchTrace?: SearchNode[];
  mode: "live" | "demo";
  error?: string;
}

export interface SearchNode {
  id: string;
  label: string;
  depth: number;
  heuristic?: number;
  status: "expanded" | "candidate" | "goal";
  parentId?: string;
}

export interface BenchmarkRow {
  algorithm: string;
  solvedTrain: number;
  totalTrain: number;
  solvedTest: number;
  totalTest: number;
  avgNodes: number;
  avgTimeMs: number;
}