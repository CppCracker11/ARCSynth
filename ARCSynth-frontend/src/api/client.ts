import { mockBenchmark, mockResult, mockTasks } from "../data/mock";
import type { Algorithm, BenchmarkRow, SynthesisResult, Task } from "../types";

export const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000/api";

export interface ClientOptions {
  demoMode: boolean;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `API request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export async function getTasks(options: ClientOptions): Promise<Task[]> {
  if (options.demoMode) return mockTasks;
  return request<Task[]>("/tasks");
}

export async function synthesize(
  taskId: string,
  algorithm: Algorithm,
  maxDepth: number,
  options: ClientOptions,
): Promise<SynthesisResult> {
  if (options.demoMode) {
    return {
      ...mockResult,
      taskId,
      algorithm,
      mode: "demo",
    };
  }

  return request<SynthesisResult>("/synthesize", {
    method: "POST",
    body: JSON.stringify({ task_id: taskId, algorithm, max_depth: maxDepth }),
  });
}

export async function getBenchmark(options: ClientOptions): Promise<BenchmarkRow[]> {
  if (options.demoMode) return mockBenchmark;
  return request<BenchmarkRow[]>("/benchmark");
}