'use client';

import { Activity, ShieldAlert, Wifi, Server, CheckCircle2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import NetworkTopologyMap from '@/components/visualizations/NetworkTopologyMap';
import TrustScoreTimeline from '@/components/visualizations/TrustScoreTimeline';
import { motion, AnimatePresence } from 'framer-motion';
import { useState, useEffect } from 'react';
import { io } from 'socket.io-client';

// Particle Background for deep navy with shallow DOF
const ParticleField = ({ success }: { success: boolean }) => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none z-0">
      <div className="absolute inset-0 bg-[#040814]" />
      {/* Cinematic ambient glow that turns cooler teal on success */}
      <div className={cn("absolute top-[-20%] left-[-10%] w-[60%] h-[60%] rounded-full blur-[180px] opacity-20 transition-colors duration-1000", success ? "bg-[#2dd4bf]" : "bg-[#0ea5e9]")} />
      <div className={cn("absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] rounded-full blur-[150px] opacity-10 transition-colors duration-1000", success ? "bg-[#22c55e]" : "bg-[#3edcff]")} />
      <div className={cn("absolute top-[40%] left-[50%] w-[30%] h-[30%] rounded-full blur-[200px] opacity-10 transition-colors duration-1000", success ? "bg-[#2dd4bf]" : "bg-[#0ea5e9]")} />
      
      {/* Particles with DOF blur */}
      {Array.from({ length: 40 }).map((_, i) => (
        <motion.div
          key={i}
          initial={{
            x: Math.random() * (typeof window !== 'undefined' ? window.innerWidth : 2000),
            y: Math.random() * (typeof window !== 'undefined' ? window.innerHeight : 1000),
            opacity: Math.random() * 0.3 + 0.1,
            scale: Math.random() * 1.5 + 0.5,
          }}
          animate={{
            y: [null, Math.random() * -200 - 50],
            opacity: [null, 0],
          }}
          transition={{
            duration: Math.random() * 10 + 10,
            repeat: Infinity,
            ease: 'linear',
          }}
          className={cn("absolute bg-white rounded-full", Math.random() > 0.5 ? "w-1.5 h-1.5 blur-[2px]" : "w-1 h-1")}
        />
      ))}
    </div>
  );
};

export default function DashboardOverview() {
  const [isolatedNode, setIsolatedNode] = useState<string | null>(null);

  const [activeDevices, setActiveDevices] = useState(0);
  const [criticalAlerts, setCriticalAlerts] = useState(0);
  const [avgTrustScore, setAvgTrustScore] = useState(0);
  const [threatsMitigated, setThreatsMitigated] = useState(0);
  const [trustScores, setTrustScores] = useState<Record<string, number>>({});
  const [latestAlert, setLatestAlert] = useState<{device: string, type: string, time: string} | null>(null);

  useEffect(() => {
    const socket = io('http://localhost:8000', {
       transports: ['polling', 'websocket'],
    });

    socket.on('connect', () => {
       console.log('Connected to socket.io backend');
    });

    socket.on('trust_update', (data) => {
        console.log('🔥 RECEIVED TRUST UPDATE:', data);
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
    });

    socket.on('new_alert', (data) => {
        setThreatsMitigated(prev => prev + 1);
        setLatestAlert({
            device: data.device || 'UNKNOWN',
            type: data.type || 'Suspicious Activity',
            time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        });
    });

    socket.onAny((event, data) => {
       console.log('SOCKET EVENT:', event, data);
       if (event === 'trust_update' || event === 'telemetry_ping' || event === 'new_alert') {
           const div = document.createElement('div');
           div.style.cssText = 'position:relative;z-index:9999;background:black;color:#3edcff;font-family:monospace;padding:2px;font-size:10px;border-bottom:1px solid #333;';
           div.innerText = `[${new Date().toISOString()}] ${event} => ${JSON.stringify(data).substring(0, 50)}...`;
           const container = document.getElementById('ws-logs-container');
           if (container) {
               container.prepend(div);
               if (container.children.length > 10) {
                   container.lastChild?.remove();
               }
           }
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
    <div className="relative min-h-screen bg-[#040814] text-white p-6 md:p-8 font-sans overflow-hidden flex flex-col items-center">
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
                       THREAT NEUTRALIZED — RESPONSE TIME: 4.2s
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

        {/* Main Grid Area */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 flex-1 min-h-[550px]">
          
          {/* Center Screen: D3 Graph */}
          <div className="lg:col-span-3 bg-white/[0.03] backdrop-blur-2xl border border-white/[0.08] rounded-2xl flex flex-col relative overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.4)] ring-1 ring-white/5">
            <div className="p-4 border-b border-white/5 bg-black/40 flex justify-between items-center z-10">
              <h2 className="font-semibold text-sm text-gray-200 tracking-widest uppercase">Live Topology / Force-Directed Graph</h2>
              <div className="flex items-center gap-2">
                 <span className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_#22c55e] animate-pulse" />
                 <span className="text-xs font-mono text-gray-400">SYNC ACTIVE</span>
              </div>
            </div>
            <div className="flex-1 w-full relative bg-gradient-to-br from-transparent to-black/30">
              <NetworkTopologyMap onIsolate={handleIsolate} />
            </div>
          </div>

          {/* Right Side Stack: Timeline & Logs */}
          <div className="lg:col-span-1 flex flex-col gap-6">
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
                  <div className="text-gray-500">Listening to socket.io...</div>
               </div>
             </div>
          </div>

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

      </div>
    </div>
  );
}
