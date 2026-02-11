const BASE_URL =
  (import.meta as unknown as { env: { VITE_API_URL?: string } }).env?.VITE_API_URL ??
  "http://localhost:8000";

export type PredictStats = Record<
  string,
  { pixels: number; percent: number }
>;

export type PredictResponse = {
  mask_image_base64: string;
  shape: [number, number];
  damage_counts: {
    no_damage?: number;
    minor_damage?: number;
    major_damage?: number;
    destroyed?: number;
  };
  stats: PredictStats;
  damage_score: number;
};

export type HealthResponse = {
  status: string;
  model_loaded: boolean;
};

/**
 * Check server health and whether the model is loaded.
 */
export async function healthCheck(): Promise<HealthResponse> {
  const res = await fetch(`${BASE_URL}/health`);
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<HealthResponse>;
}

/**
 * Submit pre- and post-disaster images and get damage segmentation prediction.
 * Returns colorized mask (base64 PNG), shape, and per-class stats.
 */
export async function predict(
  preImage: File,
  postImage: File
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("pre_image", preImage);
  form.append("post_image", postImage);

  const res = await fetch(`${BASE_URL}/predict`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    let message = `${res.status} ${res.statusText}`;
    try {
      const json = JSON.parse(text) as { detail?: string };
      if (json.detail) message = json.detail;
    } catch {
      if (text) message = text;
    }
    throw new Error(message);
  }

  return res.json() as Promise<PredictResponse>;
}

export type SeedColoradoEntry = {
  pre_base64: string;
  post_base64: string;
  lat: number;
  lng: number;
};

export type SeedColoradoResponse = {
  entries: SeedColoradoEntry[];
  hub?: { lat: number; lng: number };
};

/**
 * Fetch 7 pre/post image pairs with coordinates surrounding the emergency hub for routing seed.
 * Returns base64 image data; convert to File in the UI before setting state.
 */
export async function seedColorado(): Promise<SeedColoradoResponse> {
  const res = await fetch(`${BASE_URL}/seed/colorado`);
  if (!res.ok) {
    const text = await res.text();
    let message = `${res.status} ${res.statusText}`;
    try {
      const json = JSON.parse(text) as { detail?: string };
      if (json.detail) message = json.detail;
    } catch {
      if (text) message = text;
    }
    throw new Error(message);
  }
  return res.json() as Promise<SeedColoradoResponse>;
}

export async function seedJapan(): Promise<SeedColoradoResponse> {
  const res = await fetch(`${BASE_URL}/seed/japan`);
  if (!res.ok) {
    const text = await res.text();
    let message = `${res.status} ${res.statusText}`;
    try {
      const json = JSON.parse(text) as { detail?: string };
      if (json.detail) message = json.detail;
    } catch {
      if (text) message = text;
    }
    throw new Error(message);
  }
  return res.json() as Promise<SeedColoradoResponse>;
}

export type RouteResponse = {
  order: number[];
  total_distance_km: number;
  total_cost_km: number;
  algorithm?: string;
};

export type RoutingAlgorithm = "greedy" | "tsp";

/**
 * Compute visit order from hub to all sites.
 * algorithm: "greedy" (fast, damage vs distance) or "tsp" (OR-Tools optimal).
 * damageWeight: 0 = distance only, 1 = full damage prioritization (default).
 */
export async function computeRoute(
  hub: { lat: number; lng: number },
  sites: { lat: number; lng: number; damage_score: number }[],
  damageWeight: number = 1,
  algorithm: RoutingAlgorithm = "greedy"
): Promise<RouteResponse> {
  const res = await fetch(`${BASE_URL}/route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      hub: { lat: hub.lat, lng: hub.lng },
      sites: sites.map((s) => ({ lat: s.lat, lng: s.lng, damage_score: s.damage_score })),
      damage_weight: Math.max(0, Math.min(1, damageWeight)),
      algorithm: algorithm === "tsp" ? "tsp" : "greedy",
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    let message = `${res.status} ${res.statusText}`;
    try {
      const json = JSON.parse(text) as { detail?: string };
      if (json.detail) message = json.detail;
    } catch {
      if (text) message = text;
    }
    throw new Error(message);
  }
  return res.json() as Promise<RouteResponse>;
}

export type SummarySiteInput = {
  label: string;
  lat: number;
  lng: number;
  damage_score: number;
  stats: Record<string, { pixels: number; percent: number }>;
};

export type SummaryRequest = {
  hub: { lat: number; lng: number } | null;
  sites: SummarySiteInput[];
  route_order: number[] | null;
  total_distance_km: number | null;
  total_cost_km: number | null;
};

export type SummaryResponse = {
  summary: string;
};

/**
 * Generate an AI summary explaining damage masks and routing decisions.
 * Requires OPENAI_API_KEY to be set on the server.
 */
export async function generateSummary(req: SummaryRequest): Promise<SummaryResponse> {
  const res = await fetch(`${BASE_URL}/summary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      hub: req.hub,
      sites: req.sites.map((s) => ({
        label: s.label,
        lat: s.lat,
        lng: s.lng,
        damage_score: s.damage_score,
        stats: s.stats,
      })),
      route_order: req.route_order,
      total_distance_km: req.total_distance_km,
      total_cost_km: req.total_cost_km,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    let message = `${res.status} ${res.statusText}`;
    try {
      const json = JSON.parse(text) as { detail?: string };
      if (json.detail) message = json.detail;
    } catch {
      if (text) message = text;
    }
    throw new Error(message);
  }
  return res.json() as Promise<SummaryResponse>;
}
