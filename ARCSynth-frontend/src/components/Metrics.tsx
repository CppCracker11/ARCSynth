import type { SynthesisResult } from "../types";

export function Metrics({ result }: { result: SynthesisResult }) {
  const items = [
    ["Algorithm", result.algorithm === "astar" ? "A*" : result.algorithm === "weighted" ? "Weighted A*" : result.algorithm.toUpperCase()],
    ["Depth", String(result.depth)],
    ["Nodes", String(result.nodesExpanded)],
    ["Time", `${result.timeMs.toFixed(2)} ms`],
  ];
  return (
    <div className="metric-grid">
      {items.map(([label, value]) => (
        <div className="metric" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
      <div className="metric success">
        <span>Status</span>
        <strong>{result.success ? "✓ Solved" : "✕ Failed"}</strong>
      </div>
    </div>
  );
}