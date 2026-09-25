interface Props {
  program: string[];
  currentStep: number;
}

function pretty(op: string) {
  return op.replaceAll("_", " ");
}

export function ProgramViewer({ program, currentStep }: Props) {
  if (!program.length) {
    return <div className="empty-state">No synthesized program yet.</div>;
  }

  return (
    <div className="program-list">
      {program.map((op, index) => {
        const step = index + 1;
        const state = currentStep === step ? "current" : currentStep > step ? "done" : "pending";
        return (
          <div className="program-row-wrap" key={`${op}-${index}`}>
            <div className={`program-row ${state}`}>
              <span className="step-number">{step}</span>
              <span className="program-name">{pretty(op)}()</span>
              <span className="program-status">{state === "done" ? "✓" : state === "current" ? "●" : "○"}</span>
            </div>
            {index < program.length - 1 && <div className="program-arrow">↓</div>}
          </div>
        );
      })}
    </div>
  );
}