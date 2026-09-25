import type { SearchNode } from "../types";

export function SearchViewer({ nodes }: { nodes: SearchNode[] }) {
  if (!nodes.length) return <div className="empty-state">Search trace is not available for this run.</div>;
  return (
    <div className="search-trace">
      {nodes.map((node, index) => (
        <div className={`search-node ${node.status}`} key={node.id}>
          <div className="node-index">{index + 1}</div>
          <div className="node-body">
            <strong>{node.label}</strong>
            <span>depth {node.depth}{node.heuristic !== undefined ? ` · h = ${node.heuristic}` : ""}</span>
          </div>
          <span className="node-status">{node.status}</span>
        </div>
      ))}
    </div>
  );
}