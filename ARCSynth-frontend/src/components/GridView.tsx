import type { Grid } from "../types";

const ARC_COLORS = [
  "#0b0d10", "#1d4ed8", "#dc2626", "#16a34a", "#facc15",
  "#a855f7", "#f8fafc", "#64748b", "#f97316", "#22d3ee",
];

interface Props {
  grid: Grid;
  label?: string;
  compact?: boolean;
  highlighted?: Set<string>;
}

export function GridView({ grid, label, compact = false, highlighted }: Props) {
  const cols = grid[0]?.length ?? 0;
  return (
    <div className={`grid-card ${compact ? "compact" : ""}`}>
      {label && <div className="grid-label">{label}</div>}
      <div
        className="arc-grid"
        style={{ gridTemplateColumns: `repeat(${Math.max(cols, 1)}, minmax(0, 1fr))` }}
        aria-label={label ?? "ARC grid"}
      >
        {grid.flatMap((row, r) =>
          row.map((value, c) => {
            const active = highlighted?.has(`${r}:${c}`);
            return (
              <div
                key={`${r}-${c}`}
                className={`arc-cell ${active ? "highlighted" : ""}`}
                style={{ background: ARC_COLORS[value] ?? ARC_COLORS[0] }}
                title={`row ${r}, col ${c}: ${value}`}
              >
                {value !== 0 && <span>{value}</span>}
              </div>
            );
          }),
        )}
      </div>
      <div className="grid-meta">{grid.length} × {cols}</div>
    </div>
  );
}