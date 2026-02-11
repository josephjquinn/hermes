import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { predict } from "../api";

export default function ProcessingPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { preFile, postFile } = location.state as { preFile: File; postFile: File };
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const runPrediction = async () => {
      try {
        const data = await predict(preFile, postFile);
        navigate("/results", { state: { result: data } });
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setTimeout(() => navigate("/"), 3000);
      }
    };

    runPrediction();
  }, [preFile, postFile, navigate]);

  return (
    <div className="flex-1 flex flex-col items-center justify-center">
      {error ? (
        <>
          <div className="mb-6 border border-border bg-card px-4 py-3 text-sm text-foreground/90">
            {error}
          </div>
          <p className="text-muted-foreground font-serif">Redirecting to upload page...</p>
        </>
      ) : (
        <div
          className="size-12 rounded-full border-2 border-border border-t-primary animate-spin"
          aria-label="Loading"
        />
      )}
    </div>
  );
}
