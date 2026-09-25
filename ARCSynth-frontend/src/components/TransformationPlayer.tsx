import { useEffect, useMemo, useState } from "react";
import type { Step } from "../types";
import { GridView } from "./GridView";

interface Props {
  steps: Step[];
  autoPlay?: boolean;
}

export function TransformationPlayer({ steps, autoPlay = false }: Props) {
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(autoPlay);
  const [speed, setSpeed] = useState(900);

  useEffect(() => {
    setIndex(0);
    setPlaying(autoPlay);
  }, [steps, autoPlay]);

  useEffect(() => {
    if (!playing || steps.length <= 1) return;
    const id = window.setInterval(() => {
      setIndex((current) => {
        if (current >= steps.length - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, speed);
    return () => window.clearInterval(id);
  }, [playing, speed, steps.length]);

  const step = steps[index];
  const progress = useMemo(
    () => steps.length > 1 ? (index / (steps.length - 1)) * 100 : 100,
    [index, steps.length],
  );

  if (!step) return <div className="empty-state">Run synthesis to see the transformation.</div>;

  return (
    <section className="player">
      <div className="player-topline">
        <div>
          <div className="eyebrow">LIVE TRANSFORMATION</div>
          <h3>{step.operation}</h3>
        </div>
        <span className="step-pill">STEP {index} / {Math.max(steps.length - 1, 0)}</span>
      </div>

      <div className="progress-track"><div className="progress-fill" style={{ width: `${progress}%` }} /></div>

      <div className="player-grid-stage">
        {index === 0 ? (
          <GridView grid={step.grid} label="Initial state" />
        ) : (
          <div className="transform-pair">
            <GridView grid={steps[index - 1].grid} label="Before" />
            <div className="transform-arrow">→<span>{step.operation}</span></div>
            <div className="animated-after"><GridView grid={step.grid} label="After" /></div>
          </div>
        )}
      </div>

      <div className="player-controls">
        <button onClick={() => { setIndex(0); setPlaying(false); }}>↺</button>
        <button onClick={() => setIndex((i) => Math.max(0, i - 1))}>←</button>
        <button className="play-button" onClick={() => setPlaying((p) => !p)}>
          {playing ? "Ⅱ Pause" : "▶ Play"}
        </button>
        <button onClick={() => setIndex((i) => Math.min(steps.length - 1, i + 1))}>→</button>
        <label>
          Speed
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            <option value={1400}>Slow</option>
            <option value={900}>Normal</option>
            <option value={500}>Fast</option>
          </select>
        </label>
      </div>

      {step.note && <p className="player-note">{step.note}</p>}
    </section>
  );
}