# ARCSynth Studio — Frontend

Interactive frontend for the existing **ARCSynth: Search-Based Program Synthesis Engine**.

The frontend is intentionally a presentation layer. The Python ARCSynth engine remains the source of truth for:
- the DSL
- BFS / A* / Weighted A* / RL
- benchmark generation
- evaluation
- actual intermediate grid states

## Included

- Synthesize page
- ARC 10-color grid renderer
- Program visualization
- Step-by-step transformation player with play/pause/next/previous
- Search trace explorer
- Benchmark comparison page
- Demo mode with precomputed `easy_01` data
- Live API mode for the Python backend

## Run the frontend

Requirements:
- Node.js 20+ recommended

```powershell
npm install
npm run dev
```

Then open the Vite URL shown in the terminal, normally:

`http://localhost:5173`

The frontend starts in **Demo Mode**, so it can be inspected without a Python API.

## Live API mode

Start the Python FastAPI backend separately, then switch the **Demo / Live API** toggle in the top-right.

The frontend expects:

```text
GET  /api/tasks
POST /api/synthesize
GET  /api/benchmark
```

### Synthesis request

```json
{
  "task_id": "easy_01",
  "algorithm": "astar",
  "max_depth": 3
}
```

### Synthesis response

The important part is that `steps` contains the ACTUAL grids produced by the Python DSL:

```json
{
  "success": true,
  "taskId": "easy_01",
  "algorithm": "astar",
  "program": ["rotate_90"],
  "steps": [
    {
      "operation": "Original test input",
      "grid": [[0,0,2,0]]
    },
    {
      "operation": "rotate_90",
      "grid": [[0,0,0,0]]
    }
  ],
  "testInput": [],
  "predictedOutput": [],
  "expectedOutput": [],
  "nodesExpanded": 5,
  "timeMs": 1.4,
  "depth": 1,
  "searchTrace": []
}
```

See `src/types.ts` and `src/api/client.ts` for the exact TypeScript contract.

## Important design rule

Do **not** reimplement the Python DSL in the frontend.

The frontend should animate the grids returned by the backend. This guarantees that what is shown on screen is the actual execution of the ARCSynth engine.

## Build

```powershell
npm run build
```

## Current demo fixture

Demo mode uses a small precomputed `easy_01` example and the current benchmark snapshot so the UI can be reviewed independently of the backend.

It is clearly labelled as frontend demo/preview data.
