'use client';

import { Activity, ShieldAlert, Wifi, Server, CheckCircle2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import NetworkTopologyMap from '@/components/visualizations/NetworkTopologyMap';
import TrustScoreTimeline from '@/components/visualizations/TrustScoreTimeline';
import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';

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

  const kpis = [
    { title: 'Active Devices', value: '50', icon: Wifi, color: 'text-green-400', glow: 'bg-green-500' },
    { title: 'Critical Alerts', value: isolatedNode ? '2' : '3', icon: ShieldAlert, color: 'text-red-500', glow: 'bg-red-600' },
    { title: 'Avg Trust Score', value: isolatedNode ? '89.4' : '74.2', icon: Activity, color: 'text-[#3edcff]', glow: 'bg-[#0ea5e9]' },
    { title: 'Threats Mitigated', value: isolatedNode ? '12' : '11', icon: Server, color: 'text-indigo-400', glow: 'bg-indigo-500' },
  ];

  const handleIsolate = (nodeId: string) => {
     setIsolatedNode(nodeId);
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
               <div className="flex-1 p-4 font-mono text-xs space-y-2 overflow-y-auto">
                  <div className="text-gray-500">Listening to socket.io...</div>
                  {isolatedNode && (
                     <motion.div 
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="text-green-400 border-l-2 border-green-500 pl-2"
                     >
                        [03:17:42] device_isolated — {isolatedNode}
                     </motion.div>
                  )}
               </div>
             </div>
          </div>

        </div>

        {/* Bottom Alert Queue */}
        {!isolatedNode && (
            <motion.div 
               exit={{ opacity: 0, scale: 0.95 }}
               className="bg-[#450a0a]/40 backdrop-blur-3xl border border-red-500/40 rounded-2xl p-5 shadow-[0_0_40px_rgba(220,38,38,0.15)] ring-1 ring-red-500/50 flex items-center gap-6 relative overflow-hidden"
            >
               <div className="absolute inset-0 bg-red-600/10 animate-pulse" />
               <div className="w-4 h-4 rounded-full bg-red-500 shadow-[0_0_15px_#ef4444] animate-ping relative z-10" />
               <div className="relative z-10 flex-1">
                  <span className="text-red-500 font-mono font-bold text-xl md:text-2xl tracking-widest drop-shadow-[0_0_8px_rgba(239,68,68,0.8)]">
                  BOTNET C2 BEACONING — MED-0007 — CRITICAL
                  </span>
               </div>
               <div className="relative z-10 text-red-400/80 font-mono text-sm tracking-widest border border-red-500/30 px-3 py-1 rounded bg-red-950/50">
                  DETECTED @ 02:14 AM
               </div>
            </motion.div>
        )}

      </div>
    </div>
  );
}
