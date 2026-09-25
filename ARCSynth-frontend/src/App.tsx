import { useState } from "react";
import type { SynthesisResult } from "./types";
import { SynthesizePage } from "./pages/SynthesizePage";
import { ExplorePage } from "./pages/ExplorePage";
import { BenchmarkPage } from "./pages/BenchmarkPage";
import "./styles.css";

type Page = "synthesize" | "explore" | "benchmark";

export default function App() {
  const [page, setPage] = useState<Page>("synthesize");
  const [demoMode, setDemoMode] = useState(true);
  const [result, setResult] = useState<SynthesisResult | null>(null);

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => setPage("synthesize")}>
          <span className="brand-mark">A</span>
          <span><strong>ARCSynth</strong><small>Studio</small></span>
        </button>
        <nav>
          <button className={page === "synthesize" ? "active" : ""} onClick={() => setPage("synthesize")}>Synthesize</button>
          <button className={page === "explore" ? "active" : ""} onClick={() => setPage("explore")}>Explore</button>
          <button className={page === "benchmark" ? "active" : ""} onClick={() => setPage("benchmark")}>Benchmark</button>
        </nav>
        <label className="mode-switch">
          <span>Demo</span>
          <button className={`toggle ${demoMode ? "on" : ""}`} onClick={() => { setDemoMode((v) => !v); setResult(null); }} aria-label="Toggle demo mode">
            <span />
          </button>
          <span>Live API</span>
        </label>
      </header>

      <main>
        {page === "synthesize" && <SynthesizePage demoMode={demoMode} onResult={setResult} />}
        {page === "explore" && <ExplorePage result={result} />}
        {page === "benchmark" && <BenchmarkPage demoMode={demoMode} />}
      </main>

      <footer>
        <span>ARCSynth · Search-Based Program Synthesis</span>
        <span>{demoMode ? "Frontend preview data" : "Connected to Python API"}</span>
      </footer>
    </div>
  );
}