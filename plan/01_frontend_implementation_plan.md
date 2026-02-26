# DeviceDNA — Frontend Implementation Plan

> **Framework**: Next.js 14 (App Router) + TypeScript  
> **Styling**: Tailwind CSS v3 + shadcn/ui  
> **Visualizations**: D3.js, Recharts, Visx  
> **Real-time**: Socket.IO Client  
> **State**: Zustand + TanStack React Query

---

## 1. Design System & Visual Identity

### 1.1 Color Palette

```
Theme: Dark-first (SOC analysts work in dark rooms)

Background Hierarchy:
  --bg-deep:       hsl(222, 47%, 5%)     # Deepest background
  --bg-primary:    hsl(222, 47%, 8%)     # Main panels
  --bg-secondary:  hsl(222, 40%, 12%)    # Cards, drawers
  --bg-elevated:   hsl(222, 35%, 16%)    # Hover states, modals
  --bg-surface:    hsl(222, 30%, 20%)    # Input fields

Trust Score Colors (Critical to the brand):
  --trust-critical: hsl(0, 85%, 55%)     # 0–19  — Deep Red
  --trust-danger:   hsl(15, 90%, 55%)    # 20–39 — Orange-Red
  --trust-warning:  hsl(40, 95%, 55%)    # 40–59 — Amber
  --trust-guarded:  hsl(55, 90%, 55%)    # 60–79 — Yellow-Green
  --trust-normal:   hsl(145, 70%, 50%)   # 80–89 — Green
  --trust-trusted:  hsl(155, 80%, 45%)   # 90–100 — Emerald

Accent Colors:
  --accent-cyan:    hsl(190, 95%, 55%)   # Primary accent / links
  --accent-purple:  hsl(265, 85%, 65%)   # Secondary accent / AI features
  --accent-blue:    hsl(210, 90%, 60%)   # Info / default
  --accent-pink:    hsl(330, 80%, 60%)   # Alerts highlights

Severity Badges:
  --severity-critical: hsl(0, 85%, 50%)
  --severity-high:     hsl(25, 90%, 55%)
  --severity-medium:   hsl(45, 90%, 55%)
  --severity-low:      hsl(195, 80%, 55%)

Border & Glow:
  --border-subtle:  hsl(222, 30%, 18%)
  --border-active:  hsl(190, 60%, 40%)
  --glow-cyan:      0 0 20px hsla(190, 95%, 55%, 0.3)
  --glow-red:       0 0 20px hsla(0, 85%, 55%, 0.3)
```

### 1.2 Typography

```css
/* Google Fonts: Inter (UI) + JetBrains Mono (Data/Code) */
--font-ui:    'Inter', system-ui, sans-serif;
--font-mono:  'JetBrains Mono', monospace;

/* Scale */
--text-xs:    0.75rem;   /* 12px — labels, timestamps */
--text-sm:    0.875rem;  /* 14px — secondary text */
--text-base:  1rem;      /* 16px — body text */
--text-lg:    1.125rem;  /* 18px — panel headers */
--text-xl:    1.25rem;   /* 20px — section titles */
--text-2xl:   1.5rem;    /* 24px — page titles */
--text-3xl:   1.875rem;  /* 30px — hero numbers (trust scores) */
--text-4xl:   2.25rem;   /* 36px — dashboard KPIs */
```

### 1.3 Component Design Principles

- **Glassmorphism** on elevated panels: `backdrop-blur-xl bg-white/5 border border-white/10`
- **Subtle gradients** on interactive elements
- **Micro-animations**: 200ms ease-out transitions on hover, 300ms on open/close
- **Glow effects** on critical alerts (pulsing red glow)
- **Monospace numbers** for all numeric data (trust scores, IPs, ports)
- **Scanline aesthetic** — subtle horizontal lines in backgrounds for SOC feel

---

## 2. Page Architecture & Routing

```
app/
├── layout.tsx                    # Root: dark theme, font loading, providers
├── page.tsx                      # Landing page / login
├── (auth)/
│   ├── login/page.tsx
│   └── register/page.tsx
├── dashboard/
│   ├── layout.tsx                # Dashboard shell: sidebar + header + main
│   ├── page.tsx                  # Overview (all widgets combined)
│   ├── topology/
│   │   └── page.tsx              # Full-screen network topology map
│   ├── devices/
│   │   ├── page.tsx              # Device list (table + filters)
│   │   └── [deviceId]/
│   │       └── page.tsx          # Device detail (full page view)
│   ├── alerts/
│   │   ├── page.tsx              # Alert queue (full view)
│   │   └── [alertId]/
│   │       └── page.tsx          # Alert detail + TIB
│   ├── policies/
│   │   ├── page.tsx              # Policy list + NLP console
│   │   └── [policyId]/
│   │       └── page.tsx          # Policy detail + evaluation history
│   ├── drift/
│   │   └── page.tsx              # Drift heatmap + analysis
│   ├── replay/
│   │   └── page.tsx              # Attack replay mode
│   ├── predict/
│   │   └── page.tsx              # Predictive risk forecasts
│   └── settings/
│       └── page.tsx              # Platform settings, response mode toggle
```

---

## 3. Component Hierarchy

### 3.1 Layout Components

```
components/layout/
├── Sidebar.tsx                   # Left sidebar navigation
│   ├── Logo + branding
│   ├── NavItem (icon + label + badge)
│   │   ├── Overview
│   │   ├── Network Topology
│   │   ├── Devices
│   │   ├── Alerts (with unread count badge)
│   │   ├── Policies
│   │   ├── Drift Analysis
│   │   ├── Attack Replay
│   │   ├── Predictions
│   │   └── Settings
│   └── User profile / logout
├── Header.tsx                    # Top header bar
│   ├── Breadcrumbs
│   ├── Search bar (global device search)
│   ├── Notification bell (WebSocket alerts)
│   ├── Response mode indicator (Advisory/Semi/Full)
│   └── Live device count + network status
├── DashboardShell.tsx            # Wraps sidebar + header + main content
└── PageTransition.tsx            # Framer Motion page transitions
```

### 3.2 Dashboard Overview Components

```
components/dashboard/
├── OverviewGrid.tsx              # CSS Grid layout for dashboard widgets
├── NetworkHealthCard.tsx         # Overall network trust score (big number)
├── DeviceStatusPie.tsx           # Pie chart: trusted/guarded/suspicious/critical
├── ActiveAlertsCount.tsx         # Animated counter for active alerts
├── TrustDistributionBar.tsx      # Histogram of all device trust scores
├── RecentActivityFeed.tsx        # Timeline of recent events (real-time)
├── TopRiskDevices.tsx            # Top 5 at-risk devices mini-list
└── MiniTopologyPreview.tsx       # Small topology map (click to expand)
```

### 3.3 Visualization Components

```
components/visualizations/
├── NetworkTopologyMap.tsx        # D3.js force-directed graph
│   Props:
│   ├── devices: Device[]        # Nodes
│   ├── connections: Edge[]      # Edges
│   ├── onDeviceClick: (id) => void
│   ├── highlightDevice?: string
│   ├── filterByTrust?: [min, max]
│   └── showLabels: boolean
│   Features:
│   ├── Node color = trust score color
│   ├── Node size = traffic volume
│   ├── Edge thickness = flow frequency
│   ├── Edge color = normal (dim) / anomalous (red glow)
│   ├── Zoom + pan (d3-zoom)
│   ├── Click node → device detail drawer
│   ├── Hover → tooltip with device summary
│   ├── Device class icons inside nodes
│   └── Animated pulse on critical nodes
│
├── TrustScoreTimeline.tsx        # Recharts line chart
│   Props:
│   ├── deviceIds: string[]      # Multi-device overlay
│   ├── timeRange: [start, end]
│   ├── showThresholds: boolean
│   └── onPointClick: (timestamp) => void
│   Features:
│   ├── Color-coded line (green→yellow→red gradient by score)
│   ├── Threshold reference lines at 20, 40, 80
│   ├── Shaded danger zones below thresholds
│   ├── Tooltip with exact score + contributing factors
│   ├── Brush for time range selection
│   ├── Zoom: 1 hour → 90 days
│   └── Annotation markers for alert events
│
├── DriftHeatmap.tsx              # Visx calendar heatmap
│   Props:
│   ├── devices: Device[]
│   ├── driftData: DriftRecord[]
│   ├── dateRange: [start, end]
│   └── onCellClick: (device, date) => void
│   Features:
│   ├── Calendar-style grid (days × devices)
│   ├── Color intensity = CUSUM accumulated drift magnitude
│   ├── Row = device, Column = day
│   ├── Click cell → drift detail popup
│   ├── Hover → tooltip with drift features
│   └── Color scale legend
│
├── TrustScoreGauge.tsx           # Circular gauge for single device trust
│   ├── Animated fill based on score
│   ├── Color transitions (green→red)
│   ├── Inner text: score number
│   └── Outer ring: trend indicator (↑↓→)
│
├── AnomalyRadar.tsx              # Radar chart for ensemble model scores
│   ├── 4 axes: VAE, IF, LSTM, GNN
│   ├── Filled polygon = current scores
│   └── Ghost polygon = baseline normal
│
├── PredictiveForecastChart.tsx   # LSTM prediction line chart
│   ├── Historical line (solid)
│   ├── Predicted trajectory (dashed)
│   ├── Confidence interval (shaded band)
│   ├── Threshold lines at 20, 40
│   └── Breach probability annotation
│
├── AttackReplayTimeline.tsx      # Replay slider + network graph animation
│   ├── Time slider control
│   ├── Play/pause/speed controls
│   ├── NetworkTopologyMap integration (animated per frame)
│   ├── Event log sidebar (synced to timeline)
│   └── Trust score ticker during replay
│
└── WhatIfSimulator.tsx           # Interactive simulation panel
    ├── Action selector (isolate, policy change, etc.)
    ├── Target device/group selector
    ├── "Simulate" button → API call
    ├── Before/after trust score comparison
    └── Network impact visualization
```

### 3.4 Data Display Components

```
components/dashboard/
├── AlertQueue.tsx               # Main alert list
│   ├── AlertRow.tsx             # Single alert row
│   │   ├── Severity badge (Critical/High/Medium/Low)
│   │   ├── Device identifier + icon
│   │   ├── Alert headline (from TIB)
│   │   ├── Timestamp (relative + absolute)
│   │   ├── Quick action buttons (investigate, dismiss, respond)
│   │   └── Expandable TIB preview
│   ├── AlertFilters.tsx         # Filter by severity, device, type, date
│   └── AlertPagination.tsx
│
├── DeviceTable.tsx              # Device list view
│   ├── DeviceRow.tsx
│   │   ├── Device icon (by class)
│   │   ├── Device name + ID
│   │   ├── Device class badge
│   │   ├── Trust score gauge (mini)
│   │   ├── Status indicator (online/offline/sandboxed)
│   │   ├── VLAN tag
│   │   ├── Last seen timestamp
│   │   └── Action menu
│   ├── DeviceFilters.tsx        # Filter by class, trust range, VLAN, status
│   └── DeviceSortControls.tsx
│
├── ThreatIntelBrief.tsx         # Full TIB display
│   ├── Headline (large, bold)
│   ├── Severity + confidence badges
│   ├── Evidence list (bullet points with values)
│   ├── Anomaly type classification (MITRE ATT&CK reference)
│   ├── SHAP attribution bar chart
│   ├── Recommended actions list
│   └── Timeline of related events
│
├── DeviceDetailPanel.tsx        # Full device deep-dive
│   ├── Device identity section (name, MAC, IP, class, VLAN)
│   ├── TrustScoreGauge (large)
│   ├── Trust score history (TrustScoreTimeline)
│   ├── DNA fingerprint visualization (radar chart)
│   ├── Active policies list
│   ├── Anomaly scores (AnomalyRadar)
│   ├── Communication peers list
│   ├── Recent alerts for this device
│   └── Response action buttons
│
├── PolicyConsole.tsx            # NLP policy interface
│   ├── NLP input field (large textarea)
│   ├── "Parse Policy" button
│   ├── Generated rule preview (JSON view)
│   ├── Confidence score display
│   ├── "Approve & Activate" button
│   ├── Active policies list
│   └── Policy evaluation log
│
└── ResponseControlPanel.tsx     # Response action controls
    ├── Response mode toggle (Advisory/Semi/Full)
    ├── Active containment actions list
    ├── Sandbox status panel
    ├── Response audit log
    └── Manual action trigger (isolate/throttle/block specific device)
```

---

## 4. State Management Architecture

### 4.1 Zustand Stores

```typescript
// stores/deviceStore.ts
interface DeviceStore {
  devices: Map<string, Device>;
  selectedDeviceId: string | null;
  filters: DeviceFilters;
  setDevices: (devices: Device[]) => void;
  updateDeviceTrustScore: (id: string, score: number) => void;
  selectDevice: (id: string | null) => void;
  setFilters: (filters: Partial<DeviceFilters>) => void;
}

// stores/alertStore.ts
interface AlertStore {
  alerts: Alert[];
  unreadCount: number;
  filters: AlertFilters;
  addAlert: (alert: Alert) => void;
  markAsRead: (id: string) => void;
  dismissAlert: (id: string) => void;
}

// stores/realtimeStore.ts
interface RealtimeStore {
  connected: boolean;
  lastHeartbeat: Date | null;
  trustScoreUpdates: Map<string, number>; // deviceId → latest score
  networkStatus: 'healthy' | 'degraded' | 'critical';
}

// stores/dashboardStore.ts
interface DashboardStore {
  sidebarOpen: boolean;
  activeView: 'overview' | 'topology' | 'devices' | ...;
  responseMode: 'advisory' | 'semi-auto' | 'full-auto';
  timeRange: [Date, Date];
  toggleSidebar: () => void;
}
```

### 4.2 React Query (TanStack) — Server State

```typescript
// hooks/useDevices.ts
export function useDevices(filters?: DeviceFilters) {
  return useQuery({
    queryKey: ['devices', filters],
    queryFn: () => api.getDevices(filters),
    refetchInterval: 30_000, // Refetch every 30s
  });
}

// hooks/useTrustScoreHistory.ts
export function useTrustScoreHistory(deviceId: string, range: TimeRange) {
  return useQuery({
    queryKey: ['trust-history', deviceId, range],
    queryFn: () => api.getTrustScoreHistory(deviceId, range),
  });
}

// hooks/useAlerts.ts
export function useAlerts(filters?: AlertFilters) {
  return useInfiniteQuery({
    queryKey: ['alerts', filters],
    queryFn: ({ pageParam }) => api.getAlerts({ ...filters, cursor: pageParam }),
    getNextPageParam: (lastPage) => lastPage.nextCursor,
  });
}

// hooks/usePrediction.ts
export function useDevicePrediction(deviceId: string) {
  return useQuery({
    queryKey: ['prediction', deviceId],
    queryFn: () => api.getDevicePrediction(deviceId),
    staleTime: 5 * 60 * 1000, // 5 min
  });
}
```

---

## 5. WebSocket Real-Time Architecture

```typescript
// lib/websocket.ts
import { io, Socket } from 'socket.io-client';

interface ServerEvents {
  'trust_score_update': { deviceId: string; score: number; timestamp: string };
  'new_alert': { alert: Alert };
  'device_status_change': { deviceId: string; status: DeviceStatus };
  'drift_detected': { deviceId: string; feature: string; magnitude: number };
  'response_action': { deviceId: string; action: ResponseAction; result: string };
  'network_topology_update': { edges: Edge[] };
}

class DeviceDNASocket {
  private socket: Socket;

  connect(token: string) {
    this.socket = io(process.env.NEXT_PUBLIC_WS_URL, {
      auth: { token },
      transports: ['websocket'],
    });

    this.socket.on('trust_score_update', (data) => {
      useDeviceStore.getState().updateDeviceTrustScore(data.deviceId, data.score);
      useRealtimeStore.getState().setLastUpdate(data.deviceId, data);
    });

    this.socket.on('new_alert', (data) => {
      useAlertStore.getState().addAlert(data.alert);
      // Show toast notification
      toast.alert(data.alert.headline, { severity: data.alert.severity });
    });
  }
}
```

---

## 6. Key UI Patterns

### 6.1 Trust Score Display Pattern

Every trust score display follows this pattern:
```
┌─────────────────────────────┐
│       ⎡ 17 ⎤               │  ← Large monospace number
│    ╱─────────╲              │  ← Circular gauge
│   ▼ CRITICAL ▼              │  ← Status badge (color-coded)
│  ↓ 74 pts from yesterday    │  ← Trend indicator
└─────────────────────────────┘
```

### 6.2 Alert Pattern

```
┌────────────────────────────────────────────────────┐
│ 🔴 CRITICAL │ Camera #14 │ 2 min ago              │
│ ─────────────────────────────────────────────────  │
│ Camera #14 is sending large volumes of data to     │
│ unknown external servers in the middle of the      │
│ night.                                             │
│                                                    │
│ [View Brief] [Investigate] [Isolate] [Dismiss]     │
└────────────────────────────────────────────────────┘
```

### 6.3 Network Topology Interaction

```
Click node     → Open device detail drawer (right side)
Hover node     → Tooltip: name, class, trust score, IP
Double-click   → Navigate to full device page
Right-click    → Context menu: Isolate, Sandbox, View Alerts
Drag           → Reposition node
Scroll         → Zoom in/out
Click edge     → Show flow details (volume, protocol, frequency)
Red pulse node → Critical trust score (<20) — auto-animated
```

---

## 7. API Client Architecture

```typescript
// lib/api.ts
class DeviceDNAAPI {
  private baseUrl: string;
  private token: string;

  // === Devices ===
  getDevices(filters?: DeviceFilters): Promise<PaginatedResponse<Device>>;
  getDevice(id: string): Promise<Device>;
  getDeviceDNA(id: string): Promise<DNAFingerprint>;
  getDeviceConnections(id: string): Promise<Connection[]>;

  // === Trust Scores ===
  getTrustScore(deviceId: string): Promise<TrustScore>;
  getTrustScoreHistory(deviceId: string, range: TimeRange): Promise<TrustScorePoint[]>;
  getNetworkTrustOverview(): Promise<NetworkTrustOverview>;

  // === Alerts ===
  getAlerts(params: AlertQueryParams): Promise<PaginatedResponse<Alert>>;
  getAlert(id: string): Promise<Alert>;
  getAlertBrief(id: string): Promise<ThreatIntelBrief>;
  dismissAlert(id: string, reason: string): Promise<void>;

  // === Drift ===
  getDriftData(params: DriftQueryParams): Promise<DriftRecord[]>;
  getDeviceDrift(deviceId: string): Promise<DeviceDrift>;

  // === Policies ===
  getPolicies(): Promise<Policy[]>;
  createPolicy(policy: CreatePolicyRequest): Promise<Policy>;
  parseNLPolicy(text: string): Promise<ParsedPolicy>;
  activatePolicy(id: string): Promise<void>;
  deactivatePolicy(id: string): Promise<void>;

  // === Response ===
  triggerResponse(deviceId: string, action: ResponseAction): Promise<ResponseResult>;
  getResponseLog(): Promise<ResponseLogEntry[]>;
  setResponseMode(mode: ResponseMode): Promise<void>;

  // === Predictions ===
  getDevicePrediction(deviceId: string): Promise<Prediction>;
  getTopRiskDevices(count: number): Promise<RiskPrediction[]>;

  // === Replay ===
  getReplayData(incidentId: string): Promise<ReplayFrame[]>;

  // === Simulator ===
  runWhatIfSimulation(params: WhatIfParams): Promise<SimulationResult>;

  // === Network ===
  getNetworkTopology(): Promise<NetworkTopology>;
}
```

---

## 8. Feature Build Order (Frontend Phases)

> [!IMPORTANT]
> Frontend development starts at **Phase 5** of the master plan, after the backend APIs are operational. However, we can build the **layout shell** and **mock data UI** in parallel during Phase 3–4.

### Sprint F1: Shell & Navigation (2 days)
- [ ] Next.js 14 project setup with Tailwind + shadcn/ui
- [ ] Dark theme system with CSS variables
- [ ] Google Fonts (Inter + JetBrains Mono)
- [ ] Sidebar navigation component
- [ ] Header with search, notifications, status
- [ ] Dashboard grid layout system
- [ ] Page routing structure
- [ ] Loading/skeleton states

### Sprint F2: Core Visualizations (3 days)
- [ ] NetworkTopologyMap (D3.js) with mock data
- [ ] TrustScoreTimeline (Recharts) with mock data
- [ ] TrustScoreGauge component
- [ ] DriftHeatmap (Visx) with mock data
- [ ] AnomalyRadar chart

### Sprint F3: Data Integration (2 days)
- [ ] API client class implementation
- [ ] React Query hooks for all endpoints
- [ ] WebSocket client and store integration
- [ ] Replace mock data with live API data
- [ ] Real-time trust score updates working

### Sprint F4: Alert System (2 days)
- [ ] AlertQueue component with filtering
- [ ] ThreatIntelBrief full display
- [ ] Alert notifications (toast + badge)
- [ ] Alert detail page

### Sprint F5: Device Management (2 days)
- [ ] DeviceTable with sorting/filtering
- [ ] DeviceDetailPanel (full view)
- [ ] Device DNA fingerprint visualization
- [ ] Communication peers view

### Sprint F6: Advanced Features (3 days)
- [ ] PolicyConsole with NLP input
- [ ] AttackReplayTimeline with playback controls
- [ ] WhatIfSimulator interface
- [ ] PredictiveForecastChart
- [ ] ResponseControlPanel

### Sprint F7: Polish (2 days)
- [ ] Micro-animations (Framer Motion)
- [ ] Responsive design verification
- [ ] Performance optimization (lazy loading, code splitting)
- [ ] Accessibility audit
- [ ] Error boundary implementation

---

## 9. TypeScript Type Definitions

```typescript
// types/device.ts
interface Device {
  id: string;
  name: string;
  macAddress: string;
  ipAddress: string;
  deviceClass: 'camera' | 'sensor' | 'thermostat' | 'access_control' | 'medical' | 'industrial';
  vlan: number;
  status: 'online' | 'offline' | 'sandboxed' | 'isolated' | 'quarantined';
  trustScore: number;
  trustLevel: 'trusted' | 'normal' | 'guarded' | 'suspicious' | 'critical';
  lastSeen: string;
  enrolledAt: string;
  baselineComplete: boolean;
}

// types/trust.ts
interface TrustScore {
  deviceId: string;
  score: number;
  level: TrustLevel;
  pillars: {
    twinDeviation: number;     // Weight: 0.35
    mlAnomaly: number;         // Weight: 0.25
    policyConformance: number; // Weight: 0.20
    peerComparison: number;    // Weight: 0.10
    threatIntel: number;       // Weight: 0.10
  };
  trend: 'improving' | 'stable' | 'degrading';
  updatedAt: string;
}

// types/alert.ts
interface Alert {
  id: string;
  deviceId: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  type: 'hard_drift' | 'soft_drift' | 'anomaly' | 'policy_violation' | 'gnn_cluster' | 'prediction';
  headline: string;
  status: 'active' | 'investigating' | 'dismissed' | 'resolved';
  confidence: number;
  brief?: ThreatIntelBrief;
  createdAt: string;
}

// types/brief.ts
interface ThreatIntelBrief {
  alertId: string;
  headline: string;
  severity: AlertSeverity;
  confidence: number;
  evidence: EvidenceItem[];
  anomalyType: string;
  mitreMapping?: string;
  context: string;
  recommendedActions: string[];
  shapValues: ShapAttribution[];
  generatedAt: string;
}

interface EvidenceItem {
  feature: string;
  currentValue: number;
  baselineValue: number;
  deviation: number;
  direction: 'increase' | 'decrease';
  humanReadable: string;
}

// types/policy.ts
interface Policy {
  id: string;
  tier: 1 | 2 | 3;
  name: string;
  description: string;
  naturalLanguage?: string;
  rule: PolicyRule;
  status: 'active' | 'inactive' | 'pending_review';
  targetDeviceClass?: string;
  targetDeviceId?: string;
  createdAt: string;
  createdBy: string;
}

// types/network.ts
interface NetworkTopology {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
  timestamp: string;
}

interface TopologyNode {
  id: string;
  deviceName: string;
  deviceClass: string;
  trustScore: number;
  trafficVolume: number;
  x?: number;
  y?: number;
}

interface TopologyEdge {
  source: string;
  target: string;
  weight: number;
  protocol: string;
  isAnomalous: boolean;
}
```

---

## 10. Performance Targets

| Metric | Target |
|--------|--------|
| First Contentful Paint | < 1.2s |
| Time to Interactive | < 2.5s |
| Dashboard full render | < 3s |
| Topology map (50 nodes) render | < 500ms |
| Trust score timeline (30d) render | < 300ms |
| WebSocket latency | < 100ms |
| Bundle size (gzipped) | < 500KB initial |
| Lighthouse Performance | > 90 |

---

## 11. Accessibility Requirements

- Full keyboard navigation for all interactive elements
- ARIA labels on all visualization elements
- Screen reader announcements for real-time alerts
- High contrast mode support
- Focus indicators on all interactive elements
- Color-blind friendly trust score indicators (shapes + text backup)
