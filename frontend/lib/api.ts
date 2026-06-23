/**
 * DeviceDNA — Centralized API Client
 * 
 * Single source of truth for all backend HTTP calls.
 * PRD Gap 4 fix: replaces per-component mock data with real API calls.
 * 
 * Backend base: http://localhost:8000 (via Next.js rewrites in next.config.mjs)
 */

const API_BASE = typeof window !== 'undefined' ? `http://${window.location.hostname}:8000` : (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');

// ─── Types ───────────────────────────────────────────────────────────────────

export interface DeviceInfo {
  id: string;
  name: string;
  device_class: 'camera' | 'sensor' | 'thermostat' | 'access_control' | 'medical' | 'industrial';
  ip_address: string;
  vlan: number;
  trust_score: number;
  status: 'trusted' | 'guarded' | 'suspicious' | 'critical';
  pillars: {
    digital_twin?: number;
    isolation_forest?: number;
    lstm?: number;
    gnn?: number;
    drift?: number;
  };
}

export interface TrustScoreRecord {
  score: number;
  device_id: string;
  device_class: string;
  timestamp: string;
  vae_score: number;
  if_score: number;
  lstm_score: number;
  gnn_score: number;
  ensemble_score: number;
  policy_penalty: number;
  peer_penalty: number;
  penalty: number;
  status?: string;
}

export interface TrustHistoryPoint {
  timestamp: string;
  trust_score: number;
}

export interface AlertRecord {
  id: string;
  device_id: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  alert_type: string;
  message: string;
  trust_score: number;
  vae_score: number;
  if_score: number;
  lstm_score: number;
  gnn_score: number;
  tib: string | null;
  is_resolved: boolean;
  timestamp: string;
}

export interface GMVAERoute {
  device_id: string;
  routed_to_class: string;
  routing_confidence: number;
  routed_at: string;
  status: string;
  global_reconstruction_error: number;
}

export interface GMVAEComparison {
  device_id: string;
  device_class: string;
  timestamp: string;
  signals: {
    S1_reconstruction_difference: number;
    S2_latent_distance_drift: number;
    S3_cluster_confidence_anomaly: number;
    S4_latent_entropy: number;
    S5_temporal_latent_velocity: number;
    S6_graph_inconsistency: number;
  };
  composite_vae_score: number;
  isolation_forest_score: number;
  lstm_temporal_score: number;
  gnn_graph_score: number;
  ensemble_score: number;
  overall_penalty: number;
  trust_score: number;
}

// ─── Generic Fetch Helper ────────────────────────────────────────────────────

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options?.headers ?? {}),
    },
  });

  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error');
    throw new Error(`API ${path} failed [${res.status}]: ${text}`);
  }

  return res.json() as Promise<T>;
}

// ─── Health ──────────────────────────────────────────────────────────────────

export async function fetchHealth(): Promise<{ status: string; service: string }> {
  return apiFetch('/api/health');
}

// ─── Devices ─────────────────────────────────────────────────────────────────

/**
 * Get full 50-device fleet with current trust scores from Redis.
 * Falls back to 100.0 for devices not yet scored.
 * Used to seed the Network Topology Map before WebSocket events arrive.
 */
export async function fetchAllDevices(): Promise<DeviceInfo[]> {
  return apiFetch('/api/trust/devices/all');
}

/**
 * Get a dict of device_id → trust_score for all scored devices in Redis.
 */
export async function fetchScoredDevices(): Promise<Record<string, number>> {
  return apiFetch('/api/trust/devices');
}

/**
 * Get the latest full trust score record for a single device.
 */
export async function fetchDeviceTrustScore(deviceId: string): Promise<TrustScoreRecord> {
  return apiFetch(`/api/trust/${deviceId}/current`);
}

/**
 * Get historical trust scores for a device (InfluxDB time series).
 */
export async function fetchTrustHistory(
  deviceId: string,
  hours: number = 6
): Promise<TrustHistoryPoint[]> {
  return apiFetch(`/api/trust/${deviceId}/history?hours=${hours}`);
}

// ─── GMVAE Endpoints (PRD-required) ──────────────────────────────────────────

/**
 * PRD endpoint: GET /api/gmvae/route/{device_id}
 * Returns which specialist class the device was routed to and confidence.
 */
export async function fetchGMVAERoute(deviceId: string): Promise<GMVAERoute> {
  return apiFetch(`/api/trust/gmvae/route/${deviceId}`);
}

/**
 * PRD endpoint: GET /api/gmvae/comparison/{device_id}
 * Returns all 6 GMVAE comparison signals (S1–S6).
 */
export async function fetchGMVAEComparison(deviceId: string): Promise<GMVAEComparison> {
  return apiFetch(`/api/trust/gmvae/comparison/${deviceId}`);
}

// ─── Alerts ──────────────────────────────────────────────────────────────────

/**
 * Get the last 50 alerts from PostgreSQL, ordered newest first.
 */
export async function fetchAlerts(): Promise<AlertRecord[]> {
  return apiFetch('/api/alerts');
}

/**
 * Mark an alert as resolved.
 */
export async function resolveAlert(alertId: string): Promise<AlertRecord> {
  return apiFetch(`/api/alerts/${alertId}/resolve`, { method: 'POST' });
}

// ─── Trust Evaluation (force-compute) ────────────────────────────────────────

export interface EvaluatePayload {
  device_id: string;
  device_class: string;
  current_features: number[]; // 14-dimensional
}

/**
 * Force an immediate trust evaluation for a device with provided 14D feature vector.
 * Used by the What-If simulator panel.
 */
export async function evaluateDeviceTrust(payload: EvaluatePayload) {
  return apiFetch('/api/trust/evaluate', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

// ─── Response Engine ─────────────────────────────────────────────────────────

export async function isolateDevice(deviceId: string) {
  return apiFetch(`/api/response/isolate/${deviceId}`, { method: 'POST' });
}

export async function releaseDevice(deviceId: string) {
  return apiFetch(`/api/response/release/${deviceId}`, { method: 'POST' });
}
