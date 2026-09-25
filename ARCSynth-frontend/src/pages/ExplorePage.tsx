import type { SynthesisResult } from "../types";
import { SearchViewer } from "../components/SearchViewer";
import { GridView } from "../components/GridView";

export function ExplorePage({ result }: { result: SynthesisResult | null }) {
  return (
    <div className="page">
      <div className="hero compact-hero">
        <div>
          <div className="eyebrow">SEARCH EXPLORER</div>
          <h1>See how the solution was found.</h1>
          <p>Inspect the states exposed by the backend and the heuristic-guided path to the goal.</p>
        </div>
      </div>
      {!result ? (
        <div className="empty-state large">Run synthesis first, then return here to explore the search.</div>
      ) : (
        <div className="explore-layout">
          <div className="panel">
            <div className="panel-title">Search trace · {result.algorithm === "astar" ? "A*" : result.algorithm}</div>
            <SearchViewer nodes={result.searchTrace ?? []} />
          </div>
          <div className="panel">
            <div className="panel-title">Goal state</div>
            <GridView grid={result.predictedOutput} label="Predicted output" />
            <div className="goal-summary">
              <strong>{result.success ? "✓ Exact solution found" : "✕ No solution"}</strong>
              <span>{result.nodesExpanded} nodes expanded · depth {result.depth}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}