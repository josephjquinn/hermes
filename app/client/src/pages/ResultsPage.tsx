import { useNavigate, useLocation } from "react-router-dom";
import Button from "../components/Button";
import type { PredictResponse } from "../api";

export default function ResultsPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { result } = location.state as { result: PredictResponse };

  return (
    <div className="flex-1 px-5 sm:px-10 md:px-20 max-w-[1300px] mx-auto w-full pt-[30px] pb-[60px]">
      <section className="text-center mb-10">
        <h1 className="heading-lg text-foreground mb-6">
          Results
        </h1>
        <div className="inline-block border border-border bg-card px-6 py-5 mb-8">
          <p className="text-xs font-sans uppercase tracking-[0.15em] text-muted-foreground mb-1">Overall Damage Score</p>
          <p className="text-5xl md:text-6xl font-sans font-extrabold text-primary tabular-nums">{result.damage_score}</p>
          <p className="text-sm text-muted-foreground mt-1">out of 100</p>
        </div>
      </section>

      <section className="border border-border bg-card px-6 py-8 mb-10">
          <h2 className="heading-sm mb-6 text-foreground">Damage assessment</h2>
          <div className="flex flex-col gap-8 sm:flex-row sm:items-start">
            <div className="shrink-0">
              <div className="border border-border overflow-hidden">
                <img
                  src={`data:image/png;base64,${result.mask_image_base64}`}
                  alt="Damage segmentation mask"
                  className="max-h-[320px] w-auto block"
                />
              </div>
              <p className="mt-2 text-xs text-muted-foreground font-sans">
                Mask shape: {result.shape.join(" × ")}
              </p>
            </div>
            <div className="min-w-0">
              <h3 className="text-sm font-sans font-semibold text-foreground mb-3">Damage breakdown</h3>
              <ul className="space-y-2 text-sm font-serif">
                {Object.entries(result.stats).map(([name, { pixels, percent }]) => (
                  <li key={name} className="flex justify-between gap-4">
                    <span className="text-foreground/80">{name}</span>
                    <span className="tabular-nums text-foreground font-sans">
                      {pixels.toLocaleString()} px ({percent}%)
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </section>

      <div className="text-center">
        <Button variant="filled" showArrow={false} onClick={() => navigate("/")}>
          Analyze another image
        </Button>
      </div>
    </div>
  );
}
