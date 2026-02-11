import { useState } from "react";
import { UploadZone } from "../components/UploadZone";
import { DamageMap, type DamagePoint } from "../components/DamageMap";
import Button from "../components/Button";
import { predict, seedColorado, seedJapan, computeRoute, generateSummary, type RoutingAlgorithm } from "../api";

type RoutingResultsState = {
  points: DamagePoint[];
  hub: { lat: number; lng: number } | null;
  routeOrder: number[] | null;
  total_distance_km: number | null;
  total_cost_km: number | null;
  algorithm?: string;
} | null;

type RoutingEntry = {
  id: string;
  preFile: File | null;
  postFile: File | null;
  lat: string;
  lng: string;
};

export default function RoutingPage() {
  const [routingEntries, setRoutingEntries] = useState<RoutingEntry[]>([
    { id: crypto.randomUUID(), preFile: null, postFile: null, lat: "", lng: "" },
  ]);
  const [hubLat, setHubLat] = useState("");
  const [hubLng, setHubLng] = useState("");
  const [damagePriority, setDamagePriority] = useState(100);
  const [routingAlgorithm, setRoutingAlgorithm] = useState<RoutingAlgorithm>("greedy");
  const [routingStatus, setRoutingStatus] = useState<"idle" | "processing" | "done">("idle");
  const [routingError, setRoutingError] = useState<string | null>(null);
  const [routingResults, setRoutingResults] = useState<RoutingResultsState>(null);
  const [summary, setSummary] = useState<string | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [recomputeRouteLoading, setRecomputeRouteLoading] = useState(false);
  const [seedError, setSeedError] = useState<string | null>(null);

  const canRunRouting =
    routingEntries.length > 0 &&
    routingEntries.every(
      (e) =>
        e.preFile &&
        e.postFile &&
        e.lat.trim() !== "" &&
        e.lng.trim() !== ""
    ) &&
    hubLat.trim() !== "" &&
    hubLng.trim() !== "";

  const fileToDataUrl = (file: File): Promise<string> =>
    new Promise((resolve, reject) => {
      const r = new FileReader();
      r.onload = () => resolve((r.result as string) ?? "");
      r.onerror = () => reject(r.error);
      r.readAsDataURL(file);
    });

  const handleRunRouting = async () => {
    if (!canRunRouting || routingStatus === "processing") return;
    const entries = routingEntries
      .filter(
        (e) =>
          e.preFile &&
          e.postFile &&
          e.lat.trim() !== "" &&
          e.lng.trim() !== ""
      )
      .map((e) => ({
        preFile: e.preFile!,
        postFile: e.postFile!,
        lat: parseFloat(e.lat.trim()),
        lng: parseFloat(e.lng.trim()),
      }));
    const hub =
      hubLat.trim() !== "" && hubLng.trim() !== ""
        ? { lat: parseFloat(hubLat.trim()), lng: parseFloat(hubLng.trim()) }
        : null;
    setRoutingError(null);
    setRoutingStatus("processing");
    setRoutingResults(null);
    setSummary(null);
    const points: DamagePoint[] = [];
    try {
      for (let i = 0; i < entries.length; i++) {
        const { preFile, postFile, lat, lng } = entries[i];
        const [result, pre_image_base64, post_image_base64] = await Promise.all([
          predict(preFile, postFile),
          fileToDataUrl(preFile),
          fileToDataUrl(postFile),
        ]);
        points.push({
          lat,
          lng,
          damage_score: result.damage_score,
          label: `Location ${i + 1}`,
          mask_image_base64: result.mask_image_base64,
          stats: result.stats,
          pre_image_base64,
          post_image_base64,
        });
      }
      let routeOrder: number[] | null = null;
      let total_distance_km: number | null = null;
      let total_cost_km: number | null = null;
      let routeAlgorithm: string | undefined;
      if (hub && points.length > 0) {
        try {
          const routeRes = await computeRoute(hub, points, damagePriority / 100, routingAlgorithm);
          routeOrder = routeRes.order;
          total_distance_km = routeRes.total_distance_km;
          total_cost_km = routeRes.total_cost_km;
          routeAlgorithm = routeRes.algorithm;
        } catch {
          routeOrder = null;
        }
      }
      setRoutingResults({
        points,
        hub,
        routeOrder,
        total_distance_km,
        total_cost_km,
        algorithm: routeAlgorithm,
      });
      setRoutingStatus("done");
    } catch (e) {
      setRoutingError(e instanceof Error ? e.message : String(e));
      setRoutingStatus("idle");
    }
  };

  const handleNewRoutingAssessment = () => {
    setRoutingResults(null);
    setRoutingStatus("idle");
    setRoutingError(null);
    setSummary(null);
    setSummaryError(null);
    setRecomputeRouteLoading(false);
  };

  const handleRecomputeRoute = async () => {
    if (!routingResults?.hub || !routingResults.points.length) return;
    setRecomputeRouteLoading(true);
    try {
      const routeRes = await computeRoute(routingResults.hub, routingResults.points, damagePriority / 100, routingAlgorithm);
      setRoutingError(null);
      setRoutingResults({
        ...routingResults,
        routeOrder: routeRes.order,
        total_distance_km: routeRes.total_distance_km,
        total_cost_km: routeRes.total_cost_km,
        algorithm: routeRes.algorithm,
      });
    } catch (e) {
      setRoutingError(e instanceof Error ? e.message : String(e));
    } finally {
      setRecomputeRouteLoading(false);
    }
  };

  const handleGenerateSummary = async () => {
    if (!routingResults) return;
    setSummaryError(null);
    setSummaryLoading(true);
    try {
      const res = await generateSummary({
        hub: routingResults.hub,
        sites: routingResults.points.map((p, i) => ({
          label: p.label ?? `Location ${i + 1}`,
          lat: p.lat,
          lng: p.lng,
          damage_score: p.damage_score,
          stats: p.stats ?? {},
        })),
        route_order: routingResults.routeOrder,
        total_distance_km: routingResults.total_distance_km,
        total_cost_km: routingResults.total_cost_km,
      });
      setSummary(res.summary);
    } catch (e) {
      setSummaryError(e instanceof Error ? e.message : String(e));
    } finally {
      setSummaryLoading(false);
    }
  };

  const addRoutingEntry = () => {
    setRoutingEntries((prev) => [
      ...prev,
      { id: crypto.randomUUID(), preFile: null, postFile: null, lat: "", lng: "" },
    ]);
  };

  const removeRoutingEntry = (id: string) => {
    setRoutingEntries((prev) => (prev.length > 1 ? prev.filter((e) => e.id !== id) : prev));
  };

  const updateRoutingEntry = (id: string, patch: Partial<Omit<RoutingEntry, "id">>) => {
    setRoutingEntries((prev) =>
      prev.map((e) => (e.id === id ? { ...e, ...patch } : e))
    );
  };

  const base64ToFile = (b64: string, filename: string): File => {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new File([bytes], filename, { type: "image/png" });
  };

  const handleSeedColorado = async () => {
    setSeedError(null);
    try {
      const data = await seedColorado();
      const entriesFromApi = data.entries ?? [];
      const newEntries: RoutingEntry[] = entriesFromApi.map((e, i) => ({
        id: crypto.randomUUID(),
        preFile: base64ToFile(e.pre_base64, `seed_${i + 1}_pre.png`),
        postFile: base64ToFile(e.post_base64, `seed_${i + 1}_post.png`),
        lat: String(e.lat),
        lng: String(e.lng),
      }));
      setRoutingEntries(newEntries);
      if (data.hub) {
        setHubLat(String(data.hub.lat));
        setHubLng(String(data.hub.lng));
      }
    } catch (e) {
      setSeedError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleSeedJapan = async () => {
    setSeedError(null);
    try {
      const data = await seedJapan();
      const entriesFromApi = data.entries ?? [];
      const newEntries: RoutingEntry[] = entriesFromApi.map((e, i) => ({
        id: crypto.randomUUID(),
        preFile: base64ToFile(e.pre_base64, `seed_japan_${i + 1}_pre.png`),
        postFile: base64ToFile(e.post_base64, `seed_japan_${i + 1}_post.png`),
        lat: String(e.lat),
        lng: String(e.lng),
      }));
      setRoutingEntries(newEntries);
      if (data.hub) {
        setHubLat(String(data.hub.lat));
        setHubLng(String(data.hub.lng));
      }
    } catch (e) {
      setSeedError(e instanceof Error ? e.message : String(e));
    }
  };

  const shuffleArray = <T,>(arr: T[]): T[] => {
    const out = [...arr];
    for (let i = out.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [out[i], out[j]] = [out[j], out[i]];
    }
    return out;
  };

  const handleRandomSeed = () => {
    const completeIndices = routingEntries
      .map((_, i) => i)
      .filter(
        (i) =>
          routingEntries[i].preFile &&
          routingEntries[i].postFile &&
          routingEntries[i].lat.trim() !== "" &&
          routingEntries[i].lng.trim() !== ""
      );
    if (completeIndices.length < 2) return;
    const coords = completeIndices.map((i) => ({
      lat: routingEntries[i].lat,
      lng: routingEntries[i].lng,
    }));
    const files = completeIndices.map((i) => ({
      preFile: routingEntries[i].preFile!,
      postFile: routingEntries[i].postFile!,
    }));
    const shuffledCoords = shuffleArray(coords);
    const shuffledFiles = shuffleArray(files);
    setRoutingEntries((prev) => {
      const next = [...prev];
      completeIndices.forEach((entryIndex, k) => {
        next[entryIndex] = {
          ...next[entryIndex],
          lat: shuffledCoords[k].lat,
          lng: shuffledCoords[k].lng,
          preFile: shuffledFiles[k].preFile,
          postFile: shuffledFiles[k].postFile,
        };
      });
      return next;
    });
  };

  const canRandomSeed =
    routingEntries.filter(
      (e) =>
        e.preFile &&
        e.postFile &&
        e.lat.trim() !== "" &&
        e.lng.trim() !== ""
    ).length >= 2;

  return (
    <div className="flex-1 px-5 sm:px-10 md:px-20 max-w-[1300px] mx-auto w-full pt-[30px] pb-[30px] sm:pt-[60px] sm:pb-[60px]">
      {routingStatus !== "done" && (
        <section className="mb-10">
          <h1 className="font-sans font-extrabold uppercase text-foreground mb-4 text-3xl md:text-4xl tracking-tight">
            Routing
          </h1>
          <p className="font-serif text-foreground/80 max-w-xl text-base leading-relaxed">
            For each location, upload pre- and post-disaster image pairs and enter coordinates. We’ll assess damage and compute an optimal visit order from your hub.
          </p>
        </section>
      )}

      {routingStatus === "done" && routingResults && (
        <>
          <section className="mb-6">
            <h2 className="heading-lg text-foreground mb-2">Routing results</h2>
            <p className="prose-copy text-foreground/80">
              Damage scores by location. Click a marker for details.
              {routingResults.routeOrder != null && routingResults.hub && (
                <> Route: {routingResults.algorithm === "tsp" ? "TSP (OR-Tools optimal)" : "Greedy (damage vs distance)"}.</>
              )}
            </p>
            {routingResults.routeOrder != null &&
              (routingResults.total_distance_km != null || routingResults.total_cost_km != null) && (
                <p className="text-sm text-foreground/60 font-sans mt-1">
                  Total distance: {routingResults.total_distance_km?.toFixed(2) ?? "—"} km
                  {routingResults.total_cost_km != null && (
                    <> · Damage-weighted cost: {routingResults.total_cost_km.toFixed(2)} km</>
                  )}
                </p>
              )}
          </section>

          {routingResults.routeOrder != null && routingResults.routeOrder.length > 0 && (
            <section className="mb-6 border border-border bg-card p-4 sm:p-5">
              <h3 className="heading-sm text-foreground mb-3">Visit order</h3>
              <div className="flex flex-wrap items-center gap-1 sm:gap-2">
                {routingResults.routeOrder.map((oneBasedIndex, visitIndex) => {
                  const isLast = visitIndex === routingResults.routeOrder!.length - 1;
                  return (
                    <span key={visitIndex} className="flex items-center gap-1 sm:gap-2">
                      <span className="flex size-8 sm:size-9 shrink-0 items-center justify-center rounded-full border-2 border-primary bg-primary/15 font-sans text-sm font-bold text-primary tabular-nums">
                        {oneBasedIndex}
                      </span>
                      {!isLast && (
                        <svg
                          className="size-4 sm:size-5 shrink-0 text-muted-foreground"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                          aria-hidden
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                        </svg>
                      )}
                    </span>
                  );
                })}
              </div>
            </section>
          )}

          <DamageMap
            points={routingResults.points}
            hub={routingResults.hub ?? undefined}
            routeOrder={routingResults.routeOrder ?? undefined}
            className="mb-10"
          />

          <section className="mb-10 border border-border bg-card p-6">
            <h2 className="heading-sm text-foreground mb-2">Route options</h2>
            <p className="text-sm text-foreground/60 font-serif mb-4">
              Change algorithm or damage priority, then recompute the visit order (no need to re-run damage assessment).
            </p>
            <div className="flex flex-col gap-4">
              <div>
                <label className="block text-sm font-sans font-medium text-foreground mb-1">Algorithm</label>
                <select
                  value={routingAlgorithm}
                  onChange={(e) => setRoutingAlgorithm(e.target.value as RoutingAlgorithm)}
                  className="border border-border bg-background px-3 py-2 text-foreground font-sans text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                >
                  <option value="greedy">Greedy — damage vs distance</option>
                  <option value="tsp">TSP — OR-Tools optimal</option>
                </select>
              </div>
              <div className="flex flex-col sm:flex-row sm:items-center gap-4 flex-wrap">
                <div className="flex items-center gap-3 min-w-0">
                  <label className="text-sm font-sans font-medium text-foreground shrink-0">Damage priority</label>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={damagePriority}
                    onChange={(e) => setDamagePriority(Number(e.target.value))}
                    className="w-32 sm:w-40 h-2 bg-muted rounded-lg appearance-none cursor-pointer shrink-0"
                  />
                  <span className="text-sm font-sans text-foreground tabular-nums shrink-0">{damagePriority}%</span>
                </div>
                <Button
                  variant="transparent"
                  showArrow={false}
                  onClick={() => void handleRecomputeRoute()}
                  disabled={recomputeRouteLoading}
                >
                  {recomputeRouteLoading ? "Updating…" : "Recompute route"}
                </Button>
              </div>
            </div>
          </section>

          <section className="mb-10 border border-border bg-card p-6">
            <h2 className="heading-sm text-foreground mb-2">AI summary</h2>
            <p className="text-sm text-foreground/60 font-serif mb-4">
              Generate a plain-language summary of the damage masks and routing decisions.
            </p>
            {!summary ? (
              <>
                <Button
                  variant="filled"
                  showArrow={false}
                  onClick={handleGenerateSummary}
                  disabled={summaryLoading}
                >
                  {summaryLoading ? "Generating…" : "Generate AI summary"}
                </Button>
                {summaryError && (
                  <p className="mt-3 text-sm text-foreground/80 font-sans">{summaryError}</p>
                )}
              </>
            ) : (
              <div className="prose-copy text-foreground/90 whitespace-pre-wrap">{summary}</div>
            )}
          </section>

          <div className="flex justify-center">
            <Button variant="filled" showArrow={false} onClick={handleNewRoutingAssessment}>
              New assessment
            </Button>
          </div>
        </>
      )}

      {routingStatus === "processing" && (
        <section className="flex-1 flex flex-col items-center justify-center min-h-[40vh] py-12">
          <div
            className="size-14 rounded-full border-2 border-border border-t-primary animate-spin"
            aria-label="Loading"
          />
          <p className="mt-6 prose-copy text-foreground/80 text-center">Assessing damage…</p>
        </section>
      )}

      {routingStatus === "idle" && (
        <>
          <section className="mb-10">
            <div className="mb-6 flex flex-wrap items-center gap-3">
              <Button variant="transparent" showArrow={false} onClick={handleSeedColorado} className="text-xs py-2 px-5">
                Seed Colorado (7 stops)
              </Button>
              <Button variant="transparent" showArrow={false} onClick={handleSeedJapan} className="text-xs py-2 px-5">
                Seed Japan (14 stops)
              </Button>
              <Button
                variant="transparent"
                showArrow={false}
                onClick={handleRandomSeed}
                disabled={!canRandomSeed}
                className="text-xs py-2 px-5"
              >
                Random seed
              </Button>
              <span className="text-sm text-foreground/60 font-sans">
                Seed: 7 stops around Denver, 14 around Tokyo. Random seed: shuffle lat/long and image pairs (need 2+ complete locations).
              </span>
            </div>
            <div className="space-y-6">
              {routingEntries.map((entry) => (
                <div
                  key={entry.id}
                  className="border border-border bg-card p-6 flex flex-col gap-6"
                >
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="min-w-0">
                      <h3 className="heading-sm mb-2 text-foreground">Pre-disaster image</h3>
                      <UploadZone
                        onFileSelect={(file) => {
                          updateRoutingEntry(entry.id, { preFile: file });
                        }}
                      />
                      {entry.preFile && (
                        <p className="mt-2 text-sm text-foreground/60 font-sans">{entry.preFile.name}</p>
                      )}
                    </div>
                    <div className="min-w-0">
                      <h3 className="heading-sm mb-2 text-foreground">Post-disaster image</h3>
                      <UploadZone
                        onFileSelect={(file) => {
                          updateRoutingEntry(entry.id, { postFile: file });
                        }}
                      />
                      {entry.postFile && (
                        <p className="mt-2 text-sm text-foreground/60 font-sans">{entry.postFile.name}</p>
                      )}
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-4 sm:gap-6 items-end">
                    <label className="flex flex-col gap-1">
                      <span className="text-sm font-sans font-medium text-foreground">Latitude</span>
                      <input
                        type="text"
                        placeholder="e.g. 39.0"
                        value={entry.lat}
                        onChange={(e) => updateRoutingEntry(entry.id, { lat: e.target.value })}
                        className="border border-border bg-background px-3 py-2 text-foreground placeholder:text-muted-foreground font-sans text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                      />
                    </label>
                    <label className="flex flex-col gap-1">
                      <span className="text-sm font-sans font-medium text-foreground">Longitude</span>
                      <input
                        type="text"
                        placeholder="e.g. -105.0"
                        value={entry.lng}
                        onChange={(e) => updateRoutingEntry(entry.id, { lng: e.target.value })}
                        className="border border-border bg-background px-3 py-2 text-foreground placeholder:text-muted-foreground font-sans text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none"
                      />
                    </label>
                    <button
                      type="button"
                      onClick={() => removeRoutingEntry(entry.id)}
                      className="btn-transparent shrink-0 text-xs py-2 px-4"
                      aria-label="Remove entry"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ))}
            </div>
            <button
              type="button"
              onClick={addRoutingEntry}
              className="mt-4 btn-transparent text-sm py-2 px-5"
            >
              + Add location
            </button>
          </section>

          <section className="mb-10 border border-border bg-card p-6">
            <h3 className="heading-sm mb-2 text-foreground">Emergency service hub</h3>
            <p className="text-sm text-foreground/60 font-serif mb-4">
              Add a hub location (e.g. dispatch or relief center). Required for routing.
            </p>
            <div className="flex flex-wrap gap-4 sm:gap-6">
              <label className="flex flex-col gap-1">
                <span className="text-sm font-sans font-medium text-foreground">Hub latitude</span>
                <input
                  type="text"
                  placeholder="e.g. 39.7392"
                  value={hubLat}
                  onChange={(e) => setHubLat(e.target.value)}
                  className="border border-border bg-background px-3 py-2 text-foreground placeholder:text-muted-foreground font-sans text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none w-32"
                />
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-sm font-sans font-medium text-foreground">Hub longitude</span>
                <input
                  type="text"
                  placeholder="e.g. -104.9903"
                  value={hubLng}
                  onChange={(e) => setHubLng(e.target.value)}
                  className="border border-border bg-background px-3 py-2 text-foreground placeholder:text-muted-foreground font-sans text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none w-36"
                />
              </label>
            </div>
          </section>

          <section className="mb-10 border border-border bg-card p-6">
            <h3 className="heading-sm mb-2 text-foreground">Routing options</h3>
            <div className="mb-4">
              <label className="block text-sm font-sans font-medium text-foreground mb-2">Algorithm</label>
              <select
                value={routingAlgorithm}
                onChange={(e) => setRoutingAlgorithm(e.target.value as RoutingAlgorithm)}
                className="border border-border bg-background px-3 py-2 text-foreground font-sans text-sm focus:border-primary focus:ring-1 focus:ring-primary outline-none"
              >
                <option value="greedy">Greedy — damage vs distance (fast, tunable)</option>
                <option value="tsp">TSP — OR-Tools optimal (best route for chosen priority)</option>
              </select>
              <p className="text-xs text-foreground/60 font-serif mt-1">
                {routingAlgorithm === "greedy"
                  ? "At each step picks the next site by balancing damage and distance."
                  : "Solves for the route that minimizes damage-weighted travel cost (open route, no return to hub)."}
              </p>
            </div>
            <div>
              <label className="block text-sm font-sans font-medium text-foreground mb-2">Damage priority</label>
              <p className="text-sm text-foreground/60 font-serif mb-2">
                0% = distance only; 100% = strongly favor high-damage sites first.
              </p>
              <div className="flex flex-col gap-3 max-w-md">
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={damagePriority}
                  onChange={(e) => setDamagePriority(Number(e.target.value))}
                  className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer"
                />
                <span className="text-sm font-sans text-foreground tabular-nums">{damagePriority}%</span>
              </div>
            </div>
          </section>

          {(routingError || seedError) && (
            <div className="mb-6 border border-border bg-card px-4 py-3 text-sm text-foreground/90 font-sans">
              {routingError ?? seedError}
            </div>
          )}

          <div className="flex justify-center">
            <Button
              variant="filled"
              showArrow={true}
              disabled={!canRunRouting}
              onClick={() => void handleRunRouting()}
            >
              Run routing assessment
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
