import { useEffect, useState } from "react";
import type { Algorithm, SynthesisResult, Task } from "../types";
import { getTasks, synthesize } from "../api/client";
import { GridView } from "../components/GridView";
import { Metrics } from "../components/Metrics";
import { ProgramViewer } from "../components/ProgramViewer";
import { TransformationPlayer } from "../components/TransformationPlayer";

interface Props {
  demoMode: boolean;
  onResult: (result: SynthesisResult | null) => void;
}

export function SynthesizePage({ demoMode, onResult }: Props) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [taskId, setTaskId] = useState("easy_01");
  const [algorithm, setAlgorithm] = useState<Algorithm>("astar");
  const [maxDepth, setMaxDepth] = useState(3);
  const [result, setResult] = useState<SynthesisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getTasks({ demoMode })
      .then((items) => {
        setTasks(items);
        if (items[0]) setTaskId(items[0].id);
      })
      .catch((e: Error) => setError(e.message));
  }, [demoMode]);

  const task = tasks.find((t) => t.id === taskId);

  async function run() {
    setLoading(true);
    setError("");
    try {
      const next = await synthesize(taskId, algorithm, maxDepth, { demoMode });
      setResult(next);
      onResult(next);
    } catch (e) {
      const message = e instanceof Error ? e.message : "Synthesis failed.";
      setError(message);
      setResult(null);
      onResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="hero">
        <div>
          <div className="eyebrow">SYMBOLIC PROGRAM SYNTHESIS</div>
          <h1>Teach the machine a transformation.</h1>
          <p>Give ARCSynth examples. Search the DSL. Watch the discovered program execute on the unseen grid.</p>
        </div>
        <div className={`mode-badge ${demoMode ? "demo" : "live"}`}>
          <span /> {demoMode ? "Frontend demo mode" : "Live Python API"}
        </div>
      </div>

      <div className="control-bar">
        <label>Task
          <select value={taskId} onChange={(e) => { setTaskId(e.target.value); setResult(null); onResult(null); }}>
            {tasks.map((t) => <option value={t.id} key={t.id}>{t.id} · {t.tier}</option>)}
          </select>
        </label>
        <label>Algorithm
          <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value as Algorithm)}>
            <option value="astar">A*</option>
            <option value="bfs">BFS</option>
            <option value="weighted">Weighted A*</option>
            <option value="rl">Q-Learning</option>
          </select>
        </label>
        <label>Max depth
          <input type="number" min={1} max={8} value={maxDepth} onChange={(e) => setMaxDepth(Number(e.target.value))} />
        </label>
        <button className="primary" onClick={run} disabled={loading || !task}>
          {loading ? "Searching…" : "▶ Synthesize"}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {!task ? (
        <div className="empty-state">No task available.</div>
      ) : (
        <>
          <section className="section">
            <div className="section-heading"><div><span className="eyebrow">01 · LEARN FROM EXAMPLES</span><h2>Training pairs</h2></div><span>{task.train.length} examples</span></div>
            <div className="training-grid">
              {task.train.map((pair, i) => (
                <div className="pair-card" key={i}>
                  <GridView grid={pair.input} label={`Input ${i + 1}`} />
                  <div className="pair-arrow">→</div>
                  <GridView grid={pair.output} label={`Target ${i + 1}`} />
                </div>
              ))}
            </div>
          </section>

          <section className="section test-section">
            <div className="section-heading"><div><span className="eyebrow">02 · HOLD OUT</span><h2>Unseen test input</h2></div><span>Prediction is hidden until synthesis</span></div>
            <div className="test-layout">
              <GridView grid={task.test[0].input} label="Test input" />
              <div className="big-arrow">→</div>
              {result ? <GridView grid={result.predictedOutput} label="Predicted output" /> : <div className="prediction-placeholder">Run synthesis<br /><span>to generate a prediction</span></div>}
            </div>
          </section>

          {result && (
            <>
              <section className="section">
                <div className="section-heading"><div><span className="eyebrow">03 · SYNTHESIS RESULT</span><h2>Discovered program</h2></div></div>
                <div className="result-layout">
                  <div className="panel"><ProgramViewer program={result.program} currentStep={result.steps.length > 1 ? 0 : 1} /></div>
                  <div className="panel"><Metrics result={result} /></div>
                </div>
              </section>

              <section className="section">
                <div className="section-heading"><div><span className="eyebrow">04 · EXECUTE THE PROGRAM</span><h2>Watch the grid transform</h2></div><span>Actual intermediate states</span></div>
                <TransformationPlayer steps={result.steps} />
              </section>
            </>
          )}
        </>
      )}
    </div>
  );
}