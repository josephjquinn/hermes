import { useMemo, useEffect, useRef, useState } from "react";
import MapGL from "react-map-gl/maplibre";
import { Marker, Popup, Source, Layer } from "react-map-gl";
import type { MapRef } from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";

function segmentLengthMeters(a: [number, number], b: [number, number]): number {
  const [lng1, lat1] = a;
  const [lng2, lat2] = b;
  const R = 6371000;
  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLng = ((lng2 - lng1) * Math.PI) / 180;
  const x = Math.sin(dLat / 2) ** 2 + Math.cos((lat1 * Math.PI) / 180) * Math.cos((lat2 * Math.PI) / 180) * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
}

const segmentLengths = (coords: [number, number][]): number[] =>
  coords.length < 2 ? [] : coords.slice(1).map((c, i) => segmentLengthMeters(coords[i], c));

function positionAlongRoute(coords: [number, number][], progress: number): [number, number] | null {
  if (coords.length < 2 || progress <= 0) return coords[0];
  if (progress >= 1) return coords[coords.length - 1];
  const lengths = segmentLengths(coords);
  const total = lengths.reduce((a, b) => a + b, 0);
  let d = progress * total;
  for (let i = 0; i < lengths.length; i++) {
    if (d <= lengths[i]) {
      const t = lengths[i] > 0 ? d / lengths[i] : 0;
      return [
        coords[i][0] + t * (coords[i + 1][0] - coords[i][0]),
        coords[i][1] + t * (coords[i + 1][1] - coords[i][1]),
      ];
    }
    d -= lengths[i];
  }
  return coords[coords.length - 1];
}

function growingRouteCoordinates(
  coords: [number, number][],
  progress: number
): [number, number][] | null {
  if (coords.length < 2 || progress <= 0) return null;
  const lengths = segmentLengths(coords);
  const total = lengths.reduce((a, b) => a + b, 0);
  if (total <= 0) return null;
  const head = positionAlongRoute(coords, progress);
  if (!head) return null;
  const distance = progress * total;
  let cum = 0;
  let lastCompletedIndex = 0;
  for (let i = 0; i < lengths.length; i++) {
    if (cum + lengths[i] <= distance) {
      cum += lengths[i];
      lastCompletedIndex = i + 1;
    } else break;
  }
  const result: [number, number][] = coords.slice(0, lastCompletedIndex + 1);
  const headEqLast = result.length > 0 && result[result.length - 1][0] === head[0] && result[result.length - 1][1] === head[1];
  if (!headEqLast) result.push(head);
  return result.length >= 2 ? result : null;
}

export type DamagePointStats = Record<string, { pixels: number; percent: number }>;

export type DamagePoint = {
  lat: number;
  lng: number;
  damage_score: number;
  label?: string;
  mask_image_base64?: string;
  stats?: DamagePointStats;
  pre_image_base64?: string;
  post_image_base64?: string;
};

const DEFAULT_CENTER = { longitude: 35.0, latitude: 39.0 };
const DEFAULT_ZOOM = 4;
const ZOOMED_IN_ZOOM = 12;

const MAP_STYLE = {
  version: 8,
  name: "OSM Detail",
  sources: {
    "osm-raster": {
      type: "raster",
      tiles: [
        "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
        "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
      ],
      tileSize: 256,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    },
  },
  layers: [
    {
      id: "osm-raster",
      type: "raster",
      source: "osm-raster",
      minzoom: 0,
      maxzoom: 19,
    },
  ],
};

type DamageMapProps = {
  points: DamagePoint[];
  hub?: { lat: number; lng: number };
  routeOrder?: number[];
  className?: string;
};

const ROUTE_ANIMATION_DURATION_MS = 12000;

export function DamageMap({ points, hub, routeOrder, className = "" }: DamageMapProps) {
  const mapRef = useRef<MapRef>(null);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [hoveredHub, setHoveredHub] = useState(false);
  const [routeProgress, setRouteProgress] = useState(0);
  const animRef = useRef<number>(0);
  const startRef = useRef<number>(0);

  const bounds = useMemo(() => {
    const all = [...points.map((p) => [p.lng, p.lat] as [number, number])];
    if (hub) all.push([hub.lng, hub.lat]);
    if (all.length < 2) return null;
    const lngs = all.map((c) => c[0]);
    const lats = all.map((c) => c[1]);
    return [
      [Math.min(...lngs), Math.min(...lats)],
      [Math.max(...lngs), Math.max(...lats)],
    ] as [[number, number], [number, number]];
  }, [points, hub]);

  const initialViewState = useMemo(() => {
    if (bounds) {
      const lng = (bounds[0][0] + bounds[1][0]) / 2;
      const lat = (bounds[0][1] + bounds[1][1]) / 2;
      return { longitude: lng, latitude: lat, zoom: ZOOMED_IN_ZOOM };
    }
    if (points.length > 0) {
      return {
        longitude: points[0].lng,
        latitude: points[0].lat,
        zoom: ZOOMED_IN_ZOOM,
      };
    }
    return { ...DEFAULT_CENTER, zoom: DEFAULT_ZOOM };
  }, [bounds, points]);

  const routeGeoJson = useMemo(() => {
    if (!hub || !routeOrder || routeOrder.length <= 1) return null;
    const coords: [number, number][] = [[hub.lng, hub.lat]];
    for (let k = 1; k < routeOrder.length; k++) {
      const p = points[routeOrder[k] - 1];
      if (p) coords.push([p.lng, p.lat]);
    }
    if (coords.length < 2) return null;
    return {
      type: "Feature" as const,
      properties: {},
      geometry: { type: "LineString" as const, coordinates: coords },
    };
  }, [hub, routeOrder, points]);

  const routeCoords = routeGeoJson?.geometry.coordinates ?? null;

  useEffect(() => {
    if (!routeCoords || routeCoords.length < 2) return;
    startRef.current = 0;
    const tick = (now: number) => {
      if (!startRef.current) startRef.current = now;
      const elapsed = now - startRef.current;
      const progress = (elapsed / ROUTE_ANIMATION_DURATION_MS) % 1;
      if (elapsed >= ROUTE_ANIMATION_DURATION_MS) startRef.current = now;
      setRouteProgress(progress);
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(animRef.current);
  }, [routeCoords]);

  useEffect(() => {
    if (!bounds || !mapRef.current) return;
    const map = mapRef.current.getMap();
    map.fitBounds(bounds, { padding: 60, maxZoom: 16 });
  }, [bounds]);

  const visitOrderForIndex = useMemo(() => {
    if (!routeOrder) return null;
    const m = new Map<number, number>();
    routeOrder.forEach((siteIndex, visit) => {
      if (siteIndex >= 1) m.set(siteIndex - 1, visit);
    });
    return m;
  }, [routeOrder]);

  const selectedPoint = selectedIndex != null ? points[selectedIndex] : null;

  const growingRouteGeoJson = useMemo(() => {
    if (!routeCoords || routeCoords.length < 2) return null;
    const coordinates = growingRouteCoordinates(routeCoords, routeProgress);
    if (!coordinates) return null;
    return {
      type: "Feature" as const,
      properties: {},
      geometry: { type: "LineString" as const, coordinates },
    };
  }, [routeCoords, routeProgress]);

  const handleResetZoom = () => {
    const map = mapRef.current?.getMap();
    if (!map) return;
    if (bounds) {
      map.fitBounds(bounds, { padding: 60, maxZoom: 16 });
    } else {
      map.flyTo({
        center: [initialViewState.longitude, initialViewState.latitude],
        zoom: initialViewState.zoom,
        duration: 500,
      });
    }
  };

  return (
    <div className={`flex flex-col lg:flex-row gap-6 ${className}`} style={{ minHeight: 480 }}>
      <div className="flex-1 min-w-0 h-[480px] lg:h-[560px] border border-border overflow-hidden relative">
        <button
          type="button"
          onClick={handleResetZoom}
          className="absolute top-3 left-3 z-10 px-3 py-1.5 rounded bg-card/95 border border-border shadow-md text-xs font-sans font-medium text-foreground hover:opacity-90 transition-opacity"
          aria-label="Reset zoom"
          title="Reset zoom"
        >
          Reset zoom
        </button>
        <MapGL
          ref={mapRef}
          initialViewState={initialViewState}
          style={{ width: "100%", height: "100%" }}
          mapStyle={MAP_STYLE as import("maplibre-gl").StyleSpecification}
        >
          {growingRouteGeoJson && (
            <Source id="route-growing" type="geojson" data={growingRouteGeoJson}>
              <Layer
                id="route-growing-glow"
                type="line"
                paint={{
                  "line-color": "hsl(210 70% 50% / 0.4)",
                  "line-width": 10,
                  "line-blur": 3,
                }}
              />
              <Layer
                id="route-growing-line"
                type="line"
                layout={{ "line-cap": "round", "line-join": "round" }}
                paint={{
                  "line-color": "hsl(210 70% 42%)",
                  "line-width": 5,
                }}
              />
            </Source>
          )}

          {hub && (
            <>
              <Marker longitude={hub.lng} latitude={hub.lat} anchor="center">
                <div
                  className="cursor-default w-8 h-8 flex items-center justify-center border-2 border-white shadow-md"
                  style={{ background: "hsl(210 70% 45%)" }}
                  aria-label="Emergency service hub"
                  onMouseEnter={() => setHoveredHub(true)}
                  onMouseLeave={() => setHoveredHub(false)}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                  </svg>
                </div>
              </Marker>
              {hoveredHub && (
                <Popup
                  longitude={hub.lng}
                  latitude={hub.lat}
                  anchor="bottom"
                  onClose={() => setHoveredHub(false)}
                  closeButton={false}
                  className="damage-map-popup damage-map-popup-hub"
                >
                  <div className="hub-tooltip-content text-xs font-sans text-black">
                    <span className="font-semibold">Emergency service hub</span>
                    <br />
                    <span className="text-black/90">
                      {hub.lat.toFixed(5)}, {hub.lng.toFixed(5)}
                    </span>
                  </div>
                </Popup>
              )}
            </>
          )}

          {points.map((p, i) => {
            const label = i + 1; /* location number */
            return (
              <Marker key={i} longitude={p.lng} latitude={p.lat} anchor="center">
                <div
                  className="cursor-pointer w-9 h-9 rounded-full flex items-center justify-center border-2 border-white shadow-md font-sans font-bold text-sm text-white"
                  style={{ background: "hsl(0 0% 12%)" }}
                  onClick={() => setSelectedIndex(i)}
                  onMouseEnter={() => setHoveredIndex(i)}
                  onMouseLeave={() => setHoveredIndex(null)}
                >
                  {label}
                </div>
              </Marker>
            );
          })}

          {(hoveredIndex != null && (points[hoveredIndex]?.mask_image_base64 || points[hoveredIndex]?.stats)) && (
            <Popup
              longitude={points[hoveredIndex].lng}
              latitude={points[hoveredIndex].lat}
              anchor="top"
              onClose={() => setHoveredIndex(null)}
              closeButton={false}
              className="damage-map-popup damage-map-popup-location"
              offset={16}
            >
              <div className="location-tooltip">
                <div className="location-tooltip__row">
                  <span className="location-tooltip__label" title={points[hoveredIndex].label ?? `Location ${hoveredIndex + 1}`}>
                    {visitOrderForIndex != null && visitOrderForIndex.has(hoveredIndex)
                      ? `Stop ${visitOrderForIndex.get(hoveredIndex)! + 1} · ${points[hoveredIndex].label ?? `Location ${hoveredIndex + 1}`}`
                      : points[hoveredIndex].label ?? `Location ${hoveredIndex + 1}`}
                  </span>
                  <span className="location-tooltip__score">{points[hoveredIndex].damage_score}</span>
                </div>
                {points[hoveredIndex].mask_image_base64 && (
                  <div className="location-tooltip__mask">
                    <img
                      src={`data:image/png;base64,${points[hoveredIndex].mask_image_base64}`}
                      alt="Damage mask"
                    />
                  </div>
                )}
              </div>
            </Popup>
          )}
        </MapGL>
      </div>

      {selectedPoint && (
        <aside className="w-full shrink-0 border border-border bg-card overflow-hidden flex flex-col aspect-square max-w-[360px] mx-auto lg:max-w-none lg:w-[480px] lg:h-[480px] lg:aspect-auto lg:mx-0 lg:self-start">
          <div className="flex items-center justify-between px-3 py-2 border-b border-border shrink-0">
            <h3 className="heading-sm text-foreground font-semibold truncate pr-2">
              {visitOrderForIndex != null && visitOrderForIndex.has(selectedIndex!)
                ? `Stop ${visitOrderForIndex.get(selectedIndex!)! + 1} · ${selectedPoint.label ?? `Location ${selectedIndex! + 1}`}`
                : selectedPoint.label ?? `Location ${selectedIndex! + 1}`}
            </h3>
            <button
              type="button"
              onClick={() => setSelectedIndex(null)}
              className="text-muted-foreground hover:text-foreground p-1 font-sans font-medium transition-colors shrink-0"
              aria-label="Close"
            >
              ✕
            </button>
          </div>
          <div className="grid grid-cols-2 grid-rows-2 flex-1 min-h-0 gap-px bg-border">
            <div className="bg-card flex flex-col min-h-0 overflow-hidden">
              <div className="text-[10px] font-sans font-medium text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0">Post</div>
              <div className="flex-1 min-h-0 flex items-center justify-center bg-muted/30">
                {selectedPoint.post_image_base64 ? (
                  <img src={selectedPoint.post_image_base64} alt="Post-disaster" className="w-full h-full object-cover" />
                ) : (
                  <span className="text-xs text-muted-foreground">—</span>
                )}
              </div>
            </div>
            <div className="bg-card flex flex-col min-h-0 overflow-hidden">
              <div className="text-[10px] font-sans font-medium text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0">Pre</div>
              <div className="flex-1 min-h-0 flex items-center justify-center bg-muted/30">
                {selectedPoint.pre_image_base64 ? (
                  <img src={selectedPoint.pre_image_base64} alt="Pre-disaster" className="w-full h-full object-cover" />
                ) : (
                  <span className="text-xs text-muted-foreground">—</span>
                )}
              </div>
            </div>
            <div className="bg-card flex flex-col min-h-0 overflow-hidden">
              <div className="text-[10px] font-sans font-medium text-muted-foreground uppercase tracking-wider px-2 py-1 shrink-0">Damage mask</div>
              <div className="flex-1 min-h-0 flex items-center justify-center bg-muted/30">
                {selectedPoint.mask_image_base64 ? (
                  <img
                    src={`data:image/png;base64,${selectedPoint.mask_image_base64}`}
                    alt="Damage mask"
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <span className="text-xs text-muted-foreground">—</span>
                )}
              </div>
            </div>
            <div className="bg-card flex flex-col min-h-0 overflow-y-auto p-3">
              <div className="text-[10px] font-sans font-medium text-muted-foreground uppercase tracking-wider mb-1">Damage score</div>
              <p className="font-sans font-bold tabular-nums text-foreground text-sm">{selectedPoint.damage_score} <span className="font-normal text-muted-foreground text-xs">/ 100</span></p>
              <div className="text-[10px] font-sans font-medium text-muted-foreground uppercase tracking-wider mt-2 mb-0.5">Coordinates</div>
              <p className="font-mono text-xs text-foreground">{selectedPoint.lat.toFixed(5)}, {selectedPoint.lng.toFixed(5)}</p>
              {selectedPoint.stats && (
                <>
                  <div className="text-[10px] font-sans font-medium text-muted-foreground uppercase tracking-wider mt-2 mb-1">Breakdown</div>
                  <ul className="space-y-0.5 text-xs font-serif text-foreground">
                    {Object.entries(selectedPoint.stats)
                      .filter(([name]) => name !== "background")
                      .map(([name, { percent }]) => (
                        <li key={name} className="flex justify-between gap-2">
                          <span>{name.replace(/_/g, " ")}</span>
                          <span className="tabular-nums font-sans shrink-0">{percent}%</span>
                        </li>
                      ))}
                  </ul>
                </>
              )}
            </div>
          </div>
        </aside>
      )}
    </div>
  );
}
