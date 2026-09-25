import { useEffect, useState } from "react";
import type { BenchmarkRow } from "../types";
import { getBenchmark } from "../api/client";

export function BenchmarkPage({ demoMode }: { demoMode: boolean }) {
  const [rows, setRows] = useState<BenchmarkRow[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    getBenchmark({ demoMode }).then(setRows).catch((e: Error) => setError(e.message));
  }, [demoMode]);

  const maxNodes = Math.max(...rows.map((r) => r.avgNodes), 1);
  return (
    <div className="page">
      <div className="hero compact-hero">
        <div>
          <div className="eyebrow">EXPERIMENTAL EVALUATION</div>
          <h1>Benchmark the search strategies.</h1>
          <p>Synthetic DSL benchmark · 24 tasks · easy, medium and hard tiers.</p>
        </div>
        <div className="snapshot-badge">24 tasks</div>
      </div>
      {error && <div className="error-banner">{error}</div>}
      <section className="section">
        <div className="panel table-panel">
          <div className="panel-title">Algorithm comparison</div>
          <table>
            <thead><tr><th>Algorithm</th><th>Train</th><th>Test</th><th>Avg nodes</th><th>Avg time</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.algorithm}>
                  <td><strong>{r.algorithm}</strong></td>
                  <td>{r.solvedTrain}/{r.totalTrain}</td>
                  <td>{r.solvedTest}/{r.totalTest}</td>
                  <td>{r.avgNodes.toFixed(1)}</td>
                  <td>{r.avgTimeMs.toFixed(2)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <section className="section">
        <div className="panel">
          <div className="panel-title">Average node expansions</div>
          <div className="bars">
            {rows.map((r) => (
              <div className="bar-row" key={r.algorithm}>
                <div className="bar-label">{r.algorithm}</div>
                <div className="bar-track"><div className="bar" style={{ width: `${(r.avgNodes / maxNodes) * 100}%` }} /></div>
                <strong>{r.avgNodes.toFixed(1)}</strong>
              </div>
            ))}
          </div>
        </div>
      </section>
      <div className="footnote">Benchmark data is labelled synthetic and is intended to compare the implemented methods, not to claim official ARC-AGI performance.</div>
    </div>
  );
}