'use client';

import { Activity, ShieldAlert, Wifi, Server, CheckCircle2, X } from 'lucide-react';
import { cn } from '@/lib/utils';
import NetworkTopologyMap from '@/components/visualizations/NetworkTopologyMap';
import TrustScoreTimeline from '@/components/visualizations/TrustScoreTimeline';
import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect, useCallback, useRef } from 'react';
import { io } from 'socket.io-client';
import { AreaChart, Area, ResponsiveContainer } from 'recharts';
import { PanelGroup, Panel, PanelResizeHandle } from 'react-resizable-panels';
import HITLPanel from '@/components/HITLPanel';
import HardwareNodesPanel from '@/components/HardwareNodesPanel';
import SandboxInterceptCard from '@/components/visualizations/SandboxInterceptCard';

// Pre-compute stable particle data to avoid SSR/client hydration mismatch
type ParticleData = {
  x: number; y: number; opacity: number; scale: number;
  animY: number; duration: number; large: boolean;
};

// Particle Background for deep navy with shallow DOF
const ParticleField = ({ success }: { success: boolean }) => {
  const [particles, setParticles] = useState<ParticleData[]>([]);

  useEffect(() => {
    setParticles(
      Array.from({ length: 40 }).map(() => ({
        x: Math.random() * window.innerWidth,
        y: Math.random() * window.innerHeight,
        opacity: Math.random() * 0.3 + 0.1,
        scale: Math.random() * 1.5 + 0.5,
        animY: Math.random() * -200 - 50,
        duration: Math.random() * 10 + 10,
        large: Math.random() > 0.5,
      }))
    );
  }, []);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
      <div className="absolute inset-0 bg-[#040814]" />
      {/* Cinematic ambient glow that turns cooler teal on success */}
      <div className={cn("absolute top-[-20%] left-[-10%] w-[60%] h-[60%] rounded-full blur-[180px] opacity-20 transition-colors duration-1000", success ? "bg-[#2dd4bf]" : "bg-[#0ea5e9]")} />
      <div className={cn("absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full blur-[150px] opacity-10 transition-colors duration-1000", success ? "bg-[#22c55e]" : "bg-[#3edcff]")} />
      <div className={cn("absolute top-[40%] left-[50%] w-[30%] h-[30%] rounded-full blur-[200px] opacity-10 transition-colors duration-1000", success ? "bg-[#2dd4bf]" : "bg-[#0ea5e9]")} />

      {/* Particles with DOF blur — only rendered client-side after mount */}
      {particles.map((p, i) => (
        <motion.div
          key={i}
          initial={{ x: p.x, y: p.y, opacity: p.opacity, scale: p.scale }}
          animate={{ y: [null, p.animY], opacity: [null, 0] }}
          transition={{ duration: p.duration, repeat: Infinity, ease: 'linear' }}
          className={cn("absolute bg-white rounded-full", p.large ? "w-1.5 h-1.5 blur-[2px]" : "w-1 h-1")}
        />
      ))}
    </div>
  );
};

export default function DashboardOverview() {
  const [isolatedNode, setIsolatedNode] = useState<string | null>(null);
  const [sandboxedNode, setSandboxedNode] = useState<string | null>(null);

  const [activeDevices, setActiveDevices] = useState(0);
  const [criticalAlerts, setCriticalAlerts] = useState(0);
  const [avgTrustScore, setAvgTrustScore] = useState(0);
  const [threatsMitigated, setThreatsMitigated] = useState(0);
  const [trustScores, setTrustScores] = useState<Record<string, number>>({});
  const [latestAlert, setLatestAlert] = useState<{device: string, type: string, time: string} | null>(null);
  const [socketEvents, setSocketEvents] = useState<{timestamp: string, event: string, data: any}[]>([]);

  // Sparkline state
  const [selectedNode, setSelectedNode] = useState<{ id: string; score: number } | null>(null);
  const selectedNodeRef = useRef<string | null>(null);
  const [sparkData, setSparkData] = useState<{ trust_score: number }[]>([]);
  const [sparkLoading, setSparkLoading] = useState(false);

  const handleNodeClick = useCallback((nodeId: string, nodeScore: number) => {
    setSelectedNode({ id: nodeId, score: nodeScore });
    selectedNodeRef.current = nodeId;
    setSparkLoading(true);
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    fetch(`${apiUrl}/api/trust/${nodeId}/history?hours=6`)
      .then(res => res.json())
      .then((data: any[]) => {
        setSparkData(data.map((p: any) => ({ trust_score: p.trust_score ?? 0 })));
      })
      .catch(() => setSparkData([]))
      .finally(() => setSparkLoading(false));
  }, []);

  const handleCloseSparkline = useCallback(() => {
    setSelectedNode(null);
    selectedNodeRef.current = null;
  }, []);

  const getSparkColor = (score: number) => {
    if (score >= 65) return '#22c55e';
    if (score >= 40) return '#f97316';
    return '#ef4444';
  };

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
    
    // Fetch initial devices from Redis to populate the dashboard before WebSocket messages arrive
    fetch(`${apiUrl}/api/trust/devices`)
      .then(res => res.json())
      .then(data => {
        setTrustScores(data);
        const values = Object.values(data) as number[];
        setActiveDevices(values.length);
        if (values.length > 0) {
            setAvgTrustScore(values.reduce((a, b) => a + b, 0) / values.length);
        }
        setCriticalAlerts(values.filter(v => v < 40).length);
      })
      .catch(err => console.error("Failed to fetch initial devices", err));

    // Fetch initial mitigated threats count
    fetch(`${apiUrl}/api/alerts/count/resolved`)
      .then(res => res.json())
      .then(data => {
          if (data && typeof data.count === 'number') {
              setThreatsMitigated(data.count);
          }
      })
      .catch(err => console.error("Failed to fetch mitigated threats count", err));

    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';
    const socket = io(wsUrl, {
       transports: ['polling', 'websocket'],
    });

    socket.on('connect', () => {
       console.log('Connected to socket.io backend');
    });

    socket.on('trust_update', (data) => {
        setTrustScores(prev => {
            const newScores = { ...prev, [data.device_id]: data.score };
            const values = Object.values(newScores) as number[];
            setActiveDevices(values.length);
            const computedAvg = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : 0;
            if (Number.isFinite(computedAvg)) {
                setAvgTrustScore(computedAvg);
            }
            setCriticalAlerts(values.filter(v => v < 40).length);
            return newScores;
        });
        
        if (selectedNodeRef.current === data.device_id) {
            setSparkData(prev => {
                const newData = [...prev, { trust_score: data.score }];
                if (newData.length > 300) return newData.slice(newData.length - 300);
                return newData;
            });
        }
    });

    socket.on('new_alert', (data) => {
        setThreatsMitigated(prev => prev + 1);
        setLatestAlert({
            device: data.device || 'UNKNOWN',
            type: data.type || 'Suspicious Activity',
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        });
    });

    socket.on('isolate_device', (data) => {
        setIsolatedNode(data.device_id);
    });

    socket.on('honeypot_device', (data) => {
        setIsolatedNode(data.device_id);
    });

    socket.on('sandbox_device', (data) => {
        setSandboxedNode(data.device_id);
    });

    socket.onAny((event, data) => {
       console.log('SOCKET EVENT:', event, data);
       if (event === 'trust_update' || event === 'telemetry_ping' || event === 'new_alert') {
           setSocketEvents(prev => {
               const newEvent = { timestamp: new Date().toISOString(), event, data };
               return [newEvent, ...prev].slice(0, 50);
           });
       }
    });

    return () => {
       socket.disconnect();
    };
  }, []);

  const kpis = [
    { title: 'Active Devices', value: activeDevices.toString(), icon: Wifi, color: 'text-green-400', glow: 'bg-green-500' },
    { title: 'Critical Alerts', value: criticalAlerts.toString(), icon: ShieldAlert, color: 'text-red-500', glow: 'bg-red-600' },
    { title: 'Avg Trust Score', value: avgTrustScore.toFixed(1), icon: Activity, color: 'text-[#3edcff]', glow: 'bg-[#0ea5e9]' },
    { title: 'Threats Mitigated', value: threatsMitigated.toString(), icon: Server, color: 'text-indigo-400', glow: 'bg-indigo-500' },
  ];

  const handleIsolate = (nodeId: string) => {
     setIsolatedNode(nodeId);
     console.log('Isolating node with current state:', trustScores);
  };

  return (
    <div className="relative min-h-screen bg-[#040814] text-white p-6 md:p-8 font-sans flex flex-col items-center">
      <ParticleField success={!!isolatedNode} />
      
      {/* Content wrapper with z-index */}
      <div className="relative z-10 flex flex-col h-full w-full max-w-[1600px] gap-6">
        
        {/* Dynamic Dashboard Header */}
        <div className="w-full h-10 mb-[-10px] flex justify-end">
           <AnimatePresence>
              {isolatedNode && (
                 <motion.div 
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-3 px-4 py-2 bg-green-500/10 border border-green-500/30 rounded-lg backdrop-blur-md"
                 >
                    <CheckCircle2 className="w-5 h-5 text-green-500" />
                    <span className="text-green-400 font-mono font-bold tracking-widest text-sm drop-shadow-[0_0_5px_rgba(34,197,94,0.5)]">
                       THREAT NEUTRALIZED (RESPONSE TIME: N/A)
                    </span>
                 </motion.div>
              )}
           </AnimatePresence>
        </div>

        {/* Header & KPIs */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {kpis.map((kpi, idx) => (
            <div key={idx} className="bg-white/[0.03] backdrop-blur-2xl border border-white/[0.08] rounded-2xl p-6 relative overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.4)] ring-1 ring-white/5 transition-colors duration-1000">
               <div className={cn("absolute -bottom-10 -right-10 w-32 h-32 blur-[60px] rounded-full opacity-30", kpi.glow)} />
               <div className="flex justify-between items-start mb-4 relative z-10">
                 <span className="text-xs font-bold text-gray-300 tracking-widest uppercase opacity-80">{kpi.title}</span>
                 <kpi.icon className={cn("w-5 h-5 drop-shadow-[0_0_10px_currentColor]", kpi.color)} />
               </div>
               <div className="relative z-10 text-5xl font-bold font-mono tracking-tight text-white drop-shadow-md">
                 {kpi.value}
               </div>
            </div>
          ))}
        </div>

        {/* Hardware Nodes Tracker */}
        <HardwareNodesPanel />

        {/* Main Resizable Area */}
        <div className="flex-1 min-h-[550px] relative">
          <PanelGroup autoSaveId="dashboard-panels" direction="horizontal" className="w-full h-full">
            {/* Left Side: Topology Map + Sparkline */}
            <Panel defaultSize={75} minSize={50}>
              <div className="h-full flex flex-col gap-6 pr-3">
                {/* Center Screen: D3 Graph */}
                <div className="flex-1 bg-white/[0.03] backdrop-blur-2xl border border-white/[0.08] rounded-2xl flex flex-col relative overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.4)] ring-1 ring-white/5">
                  <div className="p-4 border-b border-white/5 bg-black/40 flex justify-between items-center z-10">
                    <h2 className="font-semibold text-sm text-gray-200 tracking-widest uppercase">Live Topology / Force-Directed Graph</h2>
                    <div className="flex items-center gap-2">
                       <span className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_#22c55e] animate-pulse" />
                       <span className="text-xs font-mono text-gray-400">SYNC ACTIVE</span>
                    </div>
                  </div>
                  <div className="flex-1 w-full relative bg-gradient-to-br from-transparent to-black/30 min-h-[300px]">
                    <NetworkTopologyMap onIsolate={handleIsolate} onNodeClick={handleNodeClick} liveScores={trustScores} externalIsolatedNode={isolatedNode} externalSandboxedNode={sandboxedNode} />
                  </div>
                </div>

                {/* Sparkline Panel — appears below topology map on node click */}
                <AnimatePresence>
                  {selectedNode && (
                    <motion.div
                      initial={{ opacity: 0, y: -10, height: 0 }}
                      animate={{ opacity: 1, y: 0, height: 'auto' }}
                      exit={{ opacity: 0, y: -10, height: 0 }}
                      transition={{ duration: 0.3, ease: 'easeOut' }}
                      className="bg-white/[0.03] backdrop-blur-2xl border border-white/[0.08] rounded-2xl overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.4)] ring-1 ring-white/5 flex flex-col md:flex-row"
                    >
                      <div className="p-4 flex flex-1 items-center justify-between">
                        <div className="flex items-center gap-4 flex-1">
                          <div className="flex flex-col min-w-[120px]">
                            <span className="text-[10px] font-bold text-gray-500 tracking-widest uppercase">Selected Device</span>
                            <span className="text-lg font-mono font-bold text-white tracking-tight">{selectedNode.id}</span>
                          </div>
                          <div className="flex flex-col items-center px-4 border-l border-white/10">
                            <span className="text-[10px] font-bold text-gray-500 tracking-widest uppercase">Trust Score</span>
                            <span className="text-2xl font-mono font-bold" style={{ color: getSparkColor(trustScores[selectedNode.id] ?? selectedNode.score) }}>
                              {(trustScores[selectedNode.id] ?? selectedNode.score).toFixed(1)}
                            </span>
                          </div>
                          <div className="flex-1 min-w-[200px] max-w-[400px] h-[50px] ml-4">
                            {sparkLoading ? (
                              <div className="w-full h-full bg-white/[0.03] rounded animate-pulse" />
                            ) : sparkData.length > 0 ? (
                              <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={sparkData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                                  <defs>
                                    <linearGradient id={`sparkGrad-${selectedNode.id}`} x1="0" y1="0" x2="0" y2="1">
                                      <stop offset="0%" stopColor={getSparkColor(selectedNode.score)} stopOpacity={0.4} />
                                      <stop offset="100%" stopColor={getSparkColor(selectedNode.score)} stopOpacity={0.02} />
                                    </linearGradient>
                                  </defs>
                                  <Area
                                    type="monotone"
                                    dataKey="trust_score"
                                    stroke={getSparkColor(selectedNode.score)}
                                    strokeWidth={1.5}
                                    fill={`url(#sparkGrad-${selectedNode.id})`}
                                    isAnimationActive={false}
                                  />
                                </AreaChart>
                              </ResponsiveContainer>
                            ) : (
                              <div className="w-full h-full flex items-center justify-center text-gray-600 font-mono text-[10px]">No history</div>
                            )}
                          </div>
                          <span className="text-[10px] font-mono text-gray-500 ml-2">6h history</span>
                        </div>
                        {selectedNode.id !== sandboxedNode && (
                          <button
                            onClick={handleCloseSparkline}
                            className="text-gray-500 hover:text-white transition-colors p-1 rounded hover:bg-white/10 ml-4 self-start"
                          >
                            <X className="w-4 h-4" />
                          </button>
                        )}
                      </div>
                      {selectedNode.id === sandboxedNode && (
                         <div className="w-full md:w-[450px] min-h-[200px] border-t md:border-t-0 md:border-l border-white/10 relative">
                           <SandboxInterceptCard deviceId={selectedNode.id} />
                           <button
                             onClick={handleCloseSparkline}
                             className="absolute top-4 right-4 text-gray-500 hover:text-white transition-colors p-1 rounded hover:bg-white/10 bg-black/40"
                           >
                             <X className="w-4 h-4" />
                           </button>
                         </div>
                      )}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </Panel>

            <PanelResizeHandle className="w-2 hover:bg-[#3edcff]/20 transition-colors cursor-col-resize flex flex-col justify-center items-center group z-50">
              <div className="w-1 h-8 bg-white/10 group-hover:bg-[#3edcff] rounded-full transition-colors" />
            </PanelResizeHandle>

            {/* Right Side Stack: Timeline & Logs */}
            <Panel defaultSize={25} minSize={15}>
              <div className="h-full flex flex-col gap-6 pl-3">
                 <div className="bg-white/[0.03] backdrop-blur-2xl border border-white/[0.08] rounded-2xl flex flex-col shadow-[0_8px_32px_rgba(0,0,0,0.4)] overflow-hidden ring-1 ring-white/5 flex-1 min-h-[250px]">
                   <div className="p-4 border-b border-white/5 bg-black/40 z-10">
                     <h2 className="font-semibold text-sm text-gray-200 tracking-widest uppercase">Trust Score Trajectory</h2>
                   </div>
                   <div className="flex-1 w-full p-4 relative bg-gradient-to-br from-transparent to-black/30">
                     <TrustScoreTimeline />
                   </div>
                 </div>

                 {/* WebSocket Event Log */}
                 <div className="bg-[#020617] border border-white/[0.08] rounded-2xl flex flex-col shadow-[0_8px_32px_rgba(0,0,0,0.4)] overflow-hidden ring-1 ring-white/5 h-[150px]">
                   <div className="p-3 border-b border-white/5 bg-black/40 z-10">
                     <h2 className="font-semibold text-xs text-gray-400 tracking-widest uppercase">Event Logs</h2>
                   </div>
                   <div id="ws-logs-container" className="flex-1 p-4 font-mono text-xs space-y-2 overflow-y-auto">
                      {socketEvents.length > 0 ? (
                        socketEvents.map((ev, idx) => (
                          <div key={idx} style={{ position: 'relative', zIndex: 9999, background: 'black', color: '#3edcff', fontFamily: 'monospace', padding: '2px', fontSize: '10px', borderBottom: '1px solid #333' }}>
                            [{ev.timestamp}] {ev.event} ={'>'} {JSON.stringify(ev.data).substring(0, 50)}...
                          </div>
                        ))
                      ) : (
                        <div className="text-gray-500">Listening to socket.io...</div>
                      )}
                   </div>
                 </div>
              </div>
            </Panel>
          </PanelGroup>
        </div>

        {/* Bottom Alert Queue */}
        {!isolatedNode && latestAlert && (
            <motion.div 
               exit={{ opacity: 0, scale: 0.95 }}
               className="bg-[#450a0a]/40 backdrop-blur-3xl border border-red-500/40 rounded-2xl p-5 shadow-[0_0_40px_rgba(220,38,38,0.15)] ring-1 ring-red-500/50 flex items-center gap-6 relative overflow-hidden"
            >
               <div className="absolute inset-0 bg-red-600/10 animate-pulse" />
               <div className="w-4 h-4 rounded-full bg-red-500 shadow-[0_0_15px_#ef4444] animate-ping relative z-10" />
               <div className="relative z-10 flex-1">
                  <span className="text-red-500 font-mono font-bold text-xl md:text-2xl tracking-widest drop-shadow-[0_0_8px_rgba(239,68,68,0.8)] uppercase">
                  {latestAlert.type} — {latestAlert.device} — CRITICAL
                  </span>
               </div>
               <div className="relative z-10 text-red-400/80 font-mono text-sm tracking-widest border border-red-500/30 px-3 py-1 rounded bg-red-950/50">
                  DETECTED @ {latestAlert.time}
               </div>
            </motion.div>
        )}

        <HITLPanel />

      </div>
    </div>
  );
}
