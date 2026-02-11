import { cn } from "@/lib/utils";

const STEPS = [
  { id: "upload", label: "Upload" },
  { id: "preprocess", label: "Preprocess" },
  { id: "model", label: "Model" },
  { id: "results", label: "Results" },
] as const;

type StepId = (typeof STEPS)[number]["id"];

type PipelineProps = {
  currentStep?: StepId | null;
};

export function Pipeline({ currentStep = null }: PipelineProps) {
  return (
    <section className="border border-border bg-card px-6 py-8 max-w-lg">
      <p className="text-[11px] font-sans font-bold uppercase tracking-widest text-muted-foreground mb-2">
        Pipeline
      </p>
      <h2 className="heading-sm text-foreground mb-3">
        How it works
      </h2>
      <p className="prose-copy text-foreground/80 mb-8">
        After you upload an image, we preprocess it, run our damage classifier, and return a damage ratio.
      </p>

      <ol className="space-y-4 sm:space-y-5">
        {STEPS.map((step, i) => {
          const isActive = currentStep === step.id;
          const isPast = currentStep && STEPS.findIndex((s) => s.id === currentStep) > i;
          return (
            <li key={step.id} className="flex items-baseline gap-4">
              <span
                className={cn(
                  "text-sm font-sans tabular-nums",
                  isActive ? "font-bold text-foreground" : "text-foreground/50"
                )}
              >
                {String(i + 1).padStart(2, "0")}
              </span>
              <span
                className={cn(
                  "font-sans text-lg sm:text-xl",
                  isActive
                    ? "font-bold text-foreground"
                    : isPast
                      ? "font-medium text-foreground/80"
                      : "text-foreground/60"
                )}
              >
                {step.label}
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
