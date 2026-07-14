'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Crosshair, Network, ShieldX, TrendingDown, Lock, Search, Shield, Ban, RefreshCcw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useParams } from 'next/navigation';
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  Tooltip, CartesianGrid, ReferenceLine
} from 'recharts';

// ---------- Types ----------
interface HistoryPoint {
  timestamp: string;
  trust_score: number | null;
  vae_score: number | null;
  if_score: number | null;
  lstm_score: number | null;
  gnn_score: number | null;
}

interface ChartPoint {
  time: string;
  trust_score: number;
  vae_scaled: number;
}

// ---------- Trust History Chart ----------
function TrustHistoryChart({ deviceId }: { deviceId: string }) {
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`${(typeof window !== 'undefined' ? `http://${window.location.hostname}:8000` : 'http://localhost:8000')}/api/trust/${deviceId}/history?hours=24`)
      .then(res => res.json())
      .then((data: HistoryPoint[]) => {
        const mapped = data.map(p => ({
          time: p.timestamp
            ? new Date(p.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            : '',
          trust_score: p.trust_score ?? 0,
          vae_scaled: (p.vae_score ?? 0) * 100,
        }));
        setChartData(mapped);
      })
      .catch(() => setChartData([]))
      .finally(() => setLoading(false));
  }, [deviceId]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const trust = payload[0]?.value as number;
      const vae = payload[1]?.value as number;
      return (
        <div className="bg-black/90 backdrop-blur-md border border-white/10 p-3 rounded-lg shadow-2xl z-50 text-white min-w-[140px]">
          <p className="text-gray-400 text-[10px] mb-1.5 font-mono">{label}</p>
          <div className="flex items-center gap-2 mb-1">
            <span className="w-2 h-2 rounded-full bg-[#3b82f6]" />
            <span className="text-xs font-mono text-gray-300">Trust</span>
            <span className={cn("text-sm font-bold font-mono ml-auto", trust < 40 ? 'text-red-500' : trust < 65 ? 'text-orange-400' : 'text-[#3b82f6]')}>
              {trust.toFixed(1)}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-[#a855f7]" />
            <span className="text-xs font-mono text-gray-300">VAE</span>
            <span className="text-sm font-mono text-[#a855f7] ml-auto">{vae.toFixed(1)}</span>
          </div>
        </div>
      );
    }
    return null;
  };

  if (loading) {
    return (
      <div className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] rounded-3xl p-6 ring-1 ring-white/5">
        <div className="flex items-center gap-2 mb-4">
          <TrendingDown className="w-4 h-4 text-gray-400" />
          <h2 className="text-sm font-bold text-gray-300 tracking-widest uppercase">Trust Score History</h2>
        </div>
        <div className="space-y-3 animate-pulse">
          <div className="h-4 bg-white/5 rounded w-3/4" />
          <div className="h-[200px] bg-white/[0.03] rounded-xl" />
          <div className="h-3 bg-white/5 rounded w-1/2" />
        </div>
      </div>
    );
  }

  if (chartData.length === 0) {
    return (
      <div className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] rounded-3xl p-6 ring-1 ring-white/5">
        <div className="flex items-center gap-2 mb-4">
          <TrendingDown className="w-4 h-4 text-gray-400" />
          <h2 className="text-sm font-bold text-gray-300 tracking-widest uppercase">Trust Score History</h2>
        </div>
        <div className="flex flex-col items-center justify-center h-[200px] text-gray-500 font-mono text-sm">
          <Activity className="w-8 h-8 mb-3 opacity-30" />
          No historical data available yet.
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] rounded-3xl p-6 ring-1 ring-white/5">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-sm font-bold text-gray-300 tracking-widest uppercase flex items-center gap-2">
          <TrendingDown className="w-4 h-4 text-gray-400" />
          Trust Score History
        </h2>
        <div className="flex items-center gap-4 text-[10px] font-mono text-gray-500">
          <span className="flex items-center gap-1"><span className="w-3 h-[2px] bg-[#3b82f6] inline-block rounded" /> Trust</span>
          <span className="flex items-center gap-1"><span className="w-3 h-[2px] bg-[#a855f7] inline-block rounded" /> VAE</span>
        </div>
      </div>
      <div style={{ width: '100%', height: 220 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" vertical={false} />
            <XAxis
              dataKey="time"
              stroke="rgba(255,255,255,0.2)"
              tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }}
              tickMargin={8}
              minTickGap={40}
            />
            <YAxis
              stroke="rgba(255,255,255,0.2)"
              tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }}
              domain={[0, 100]}
              ticks={[0, 20, 40, 65, 100]}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.15)', strokeDasharray: '4 4' }} />

            {/* Reference Lines */}
            <ReferenceLine y={20} stroke="#ef4444" strokeDasharray="6 4" strokeWidth={1.5}
              label={{ position: 'insideTopRight', value: 'Critical', fill: '#ef4444', fontSize: 9, fontWeight: 700 }} />
            <ReferenceLine y={40} stroke="#f97316" strokeDasharray="6 4" strokeWidth={1.5}
              label={{ position: 'insideTopRight', value: 'High', fill: '#f97316', fontSize: 9, fontWeight: 700 }} />
            <ReferenceLine y={65} stroke="#eab308" strokeDasharray="6 4" strokeWidth={1.5}
              label={{ position: 'insideTopRight', value: 'Suspicious', fill: '#eab308', fontSize: 9, fontWeight: 700 }} />

            {/* VAE secondary line — thin, no dots */}
            <Line
              type="monotone"
              dataKey="vae_scaled"
              stroke="#a855f7"
              strokeWidth={1.5}
              dot={false}
              activeDot={false}
              strokeOpacity={0.6}
              isAnimationActive={false}
            />

            {/* Primary trust score line */}
            <Line
              type="monotone"
              dataKey="trust_score"
              stroke="#3b82f6"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 5, fill: '#0f172a', stroke: '#3b82f6', strokeWidth: 2 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// Mock Heatmap Data (Days x Hours)
const generateHeatmap = () => {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const map = [];
  for (let d = 0; d < 7; d++) {
    for (let h = 0; h < 24; h++) {
      const isAnomaly = d === 2 && h >= 2 && h <= 4;
      map.push({
        day: days[d],
        hour: h,
        value: isAnomaly ? Math.random() * 40 + 60 : Math.random() * 15,
        isAnomaly
      });
    }
  }
  return { days, map };
};

// ---------- Response Status Panel ----------
function ResponseStatusPanel({ deviceId }: { deviceId: string }) {
  const [status, setStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [isIsolating, setIsIsolating] = useState(false);
  const [lastRefreshed, setLastRefreshed] = useState<Date>(new Date());

  const fetchStatus = () => {
    fetch(`${(typeof window !== 'undefined' ? `http://${window.location.hostname}:8000` : 'http://localhost:8000')}/api/response/${deviceId}/status`)
      .then(res => res.json())
      .then(data => {
        setStatus(data);
        setLastRefreshed(new Date());
      })
      .catch(err => console.error(err))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => clearInterval(interval);
  }, [deviceId]);

  const handleIsolate = async () => {
    setIsIsolating(true);
    try {
      const res = await fetch(`${(typeof window !== 'undefined' ? `http://${window.location.hostname}:8000` : 'http://localhost:8000')}/api/response/${deviceId}/isolate`, {
        method: 'POST'
      });
      if (res.ok) {
        fetchStatus();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsIsolating(false);
    }
  };

  return (
    <div className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] rounded-3xl p-6 ring-1 ring-white/5 mt-8">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-sm font-bold text-gray-300 tracking-widest uppercase flex items-center gap-2">
          <Shield className="w-4 h-4 text-gray-400" />
          Autonomous Response Status
        </h2>
        <button 
          onClick={handleIsolate}
          disabled={isIsolating || (status && status.isolated)}
          className={cn(
            "px-4 py-1.5 rounded-lg text-xs font-bold tracking-widest uppercase flex items-center gap-2 transition-all",
            status?.isolated
              ? "bg-red-500/10 text-red-500 border border-red-500/20 cursor-not-allowed"
              : isIsolating
              ? "bg-gray-800 text-gray-500 cursor-not-allowed"
              : "bg-red-600 hover:bg-red-500 text-white shadow-[0_0_10px_rgba(220,38,38,0.3)]"
          )}
        >
          {isIsolating ? <RefreshCcw className="w-3 h-3 animate-spin" /> : <Lock className="w-3 h-3" />}
          {status?.isolated ? 'Isolated' : 'Manual Isolate'}
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Isolated */}
        <div className={cn(
          "p-4 rounded-xl border flex flex-col items-center justify-center gap-2 transition-all",
          status?.isolated 
            ? "bg-red-500/10 border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.2)]" 
            : "bg-[#070b14]/50 border-white/5"
        )}>
          <Lock className={cn("w-6 h-6", status?.isolated ? "text-red-500" : "text-gray-600")} />
          <span className={cn("text-xs font-bold uppercase tracking-widest", status?.isolated ? "text-red-500" : "text-gray-500")}>
            Isolated
          </span>
        </div>

        {/* Sandboxed */}
        <div className={cn(
          "p-4 rounded-xl border flex flex-col items-center justify-center gap-2 transition-all",
          status?.sandboxed 
            ? "bg-orange-500/10 border-orange-500/50 shadow-[0_0_15px_rgba(249,115,22,0.2)]" 
            : "bg-[#070b14]/50 border-white/5"
        )}>
          <Shield className={cn("w-6 h-6", status?.sandboxed ? "text-orange-500" : "text-gray-600")} />
          <span className={cn("text-xs font-bold uppercase tracking-widest", status?.sandboxed ? "text-orange-500" : "text-gray-500")}>
            Sandboxed
          </span>
        </div>

        {/* Forensic Capture */}
        <div className={cn(
          "p-4 rounded-xl border flex flex-col items-center justify-center gap-2 transition-all",
          status?.forensic_capture 
            ? "bg-yellow-500/10 border-yellow-500/50 shadow-[0_0_15px_rgba(234,179,8,0.2)]" 
            : "bg-[#070b14]/50 border-white/5"
        )}>
          <Search className={cn("w-6 h-6", status?.forensic_capture ? "text-yellow-500" : "text-gray-600")} />
          <span className={cn("text-xs font-bold uppercase tracking-widest", status?.forensic_capture ? "text-yellow-500" : "text-gray-500")}>
            Forensic
          </span>
        </div>

        {/* Blocked IPs */}
        <div className={cn(
          "p-4 rounded-xl border flex flex-col items-center justify-center gap-2 transition-all",
          status?.blocked_ips?.length > 0
            ? "bg-blue-500/10 border-blue-500/50 shadow-[0_0_15px_rgba(59,130,246,0.2)]" 
            : "bg-[#070b14]/50 border-white/5"
        )}>
          <Ban className={cn("w-6 h-6", status?.blocked_ips?.length > 0 ? "text-blue-500" : "text-gray-600")} />
          <span className={cn("text-xs font-bold uppercase tracking-widest text-center", status?.blocked_ips?.length > 0 ? "text-blue-500" : "text-gray-500")}>
            {status?.blocked_ips?.length > 0 ? `Blocked IPs: ${status.blocked_ips.length}` : 'Blocked IPs: 0'}
          </span>
        </div>
      </div>

      <div className="mt-4 text-right">
        <span className="text-[10px] font-mono text-gray-500">
          Status refreshes every 30 seconds • Last refreshed {lastRefreshed.toLocaleTimeString()}
        </span>
      </div>
    </div>
  );
}

export default function NodeInspection() {
  const params = useParams();
  const nodeId = params.id || 'SIM-0001';

  const [score, setScore] = useState(91);
  const [showAlert, setShowAlert] = useState(false);
  const [showBrief, setShowBrief] = useState(false);
  const [heatmapData, setHeatmapData] = useState<{ days: string[]; map: ReturnType<typeof generateHeatmap>['map'] }>({ days: [], map: [] });

  useEffect(() => {
    const { days, map } = generateHeatmap();
    setHeatmapData({ days, map });
  }, []);

  const { days, map } = heatmapData;

  useEffect(() => {
    const t1 = setTimeout(() => {
      const interval = setInterval(() => {
        setScore(prev => {
          if (prev <= 52) {
            clearInterval(interval);
            return 52;
          }
          return prev - 2;
        });
      }, 50);
      setTimeout(() => setShowAlert(true), 400);
      setTimeout(() => setShowBrief(true), 1200);
    }, 1500);
    return () => clearTimeout(t1);
  }, []);

  const getScoreColor = (val: number) => {
    if (val >= 80) return '#22c55e';
    if (val >= 60) return '#eab308';
    return '#ef4444';
  };

  const getScoreGlow = (val: number) => {
    if (val >= 80) return 'rgba(34, 197, 94, 0.4)';
    if (val >= 60) return 'rgba(234, 179, 8, 0.4)';
    return 'rgba(239, 68, 68, 0.6)';
  };

  const currentColor = getScoreColor(score);
  const currentGlow = getScoreGlow(score);
  const isCritical = score < 60;

  const radius = 120;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div className="relative min-h-screen bg-[#04060a] text-white p-6 font-sans overflow-hidden flex flex-col items-center justify-center">
      
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="absolute inset-0 bg-[#020408]" />
        <div 
           className="absolute top-[50%] left-[50%] -translate-x-1/2 -translate-y-1/2 w-[60%] h-[60%] rounded-full blur-[250px] transition-colors duration-500"
           style={{ backgroundColor: isCritical ? '#7f1d1d' : '#0ea5e9', opacity: isCritical ? 0.3 : 0.1 }}
        />
        <div className={cn("absolute bottom-0 left-0 w-full h-32 bg-gradient-to-t from-red-900/40 to-transparent transition-opacity duration-1000", isCritical ? "opacity-100" : "opacity-0")} />
        <div className={cn("absolute top-0 left-0 w-32 h-full bg-gradient-to-r from-red-900/30 to-transparent transition-opacity duration-1000", isCritical ? "opacity-100" : "opacity-0")} />
      </div>

      <div className="absolute top-0 left-0 w-full flex justify-center p-6 z-50">
        <AnimatePresence>
          {showAlert && (
            <motion.div
              initial={{ y: -100, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              className="bg-[#450a0a]/90 backdrop-blur-2xl border border-red-500/50 rounded-2xl p-4 shadow-[0_20px_50px_rgba(220,38,38,0.3)] ring-1 ring-red-500 flex items-center gap-4 max-w-4xl w-full"
            >
              <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center relative">
                 <ShieldX className="text-red-500 w-6 h-6 z-10" />
                 <div className="absolute inset-0 bg-red-500 rounded-full animate-ping opacity-20" />
              </div>
              <div className="flex-1">
                <h3 className="text-red-500 font-mono font-bold text-lg tracking-widest uppercase drop-shadow-[0_0_8px_rgba(239,68,68,0.8)]">
                  SLOW DATA EXFILTRATION DETECTED
                </h3>
                <div className="text-red-200/80 font-mono text-xs mt-1 flex gap-4 uppercase tracking-wider">
                  <span>Target: {nodeId}</span>
                  <span>Anomaly: total_bytes</span>
                  <span>CUSUM Accumulator: 4.2</span>
                  <span className="font-bold text-red-400">Confidence: 94%</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="relative z-10 w-full max-w-6xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-start mt-12">
        <div className="flex flex-col gap-8">
          <div className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] rounded-3xl p-8 relative overflow-hidden ring-1 ring-white/5">
             <div className="flex justify-between items-start mb-8">
               <div>
                  <div className="text-xs font-bold text-gray-500 tracking-widest uppercase mb-1">Target Node Isolation</div>
                  <h1 className="text-4xl font-mono font-bold text-white tracking-tight">{nodeId}</h1>
                  <div className="text-sm text-gray-400 mt-2 font-mono flex items-center gap-2">
                     <Network className="w-4 h-4" /> 10.4.52.199 • Medical Imaging Sensor
                  </div>
               </div>
               <div className={cn("px-4 py-1.5 rounded border font-mono text-xs tracking-widest uppercase font-bold transition-colors duration-500", 
                  isCritical ? "bg-red-500/10 border-red-500/50 text-red-500" : "bg-green-500/10 border-green-500/50 text-green-500")}>
                 {isCritical ? 'ISOLATION PENDING' : 'ONLINE'}
               </div>
             </div>

             <div className="relative flex justify-center items-center py-8">
                <div 
                   className="absolute w-64 h-64 rounded-full blur-[40px] transition-all duration-300"
                   style={{ backgroundColor: currentColor, opacity: 0.15 }}
                />
                
                <svg width="280" height="280" viewBox="0 0 280 280" className="transform -rotate-90 relative z-10">
                  <circle 
                    cx="140" cy="140" r={radius}
                    fill="transparent"
                    stroke="#1e293b"
                    strokeWidth="16"
                  />
                  <circle 
                    cx="140" cy="140" r={radius}
                    fill="transparent"
                    stroke={currentColor}
                    strokeWidth="16"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    className="transition-all duration-75 ease-out"
                    style={{ filter: `drop-shadow(0 0 10px ${currentColor})` }}
                  />
                </svg>

                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-20">
                   <div className="text-gray-400 text-xs font-bold tracking-widest uppercase mb-1">Trust Score</div>
                   <motion.div 
                     key={score}
                     initial={{ scale: 1.2 }}
                     animate={{ scale: 1 }}
                     className="text-7xl font-mono font-bold tracking-tighter"
                     style={{ color: currentColor, textShadow: `0 0 20px ${currentGlow}` }}
                   >
                     {score}
                   </motion.div>
                   <div className="text-gray-500 text-xs font-mono mt-2">/ 100</div>
                   <div className={cn("mt-4 text-xs font-bold uppercase tracking-widest transition-colors", 
                     score >= 80 ? 'text-green-500' : score >= 60 ? 'text-yellow-500' : 'text-red-500 animate-pulse')}>
                     {score >= 80 ? 'GUARDED' : score >= 60 ? 'SUSPICIOUS' : 'CRITICAL'}
                   </div>
                </div>
             </div>
          </div>
          
          <ResponseStatusPanel deviceId={nodeId as string} />
        </div>

        <div className="flex flex-col gap-8">
           <TrustHistoryChart deviceId={nodeId as string} />
           
           <div className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] rounded-3xl p-6 ring-1 ring-white/5 flex flex-col">
              <div className="flex justify-between items-center mb-6">
                 <h2 className="text-sm font-bold text-gray-300 tracking-widest uppercase flex items-center gap-2">
                    <Activity className="w-4 h-4 text-gray-400" />
                    CUSUM Drift Heatmap
                 </h2>
                 <span className="text-xs font-mono text-gray-500">7-Day Rolling</span>
              </div>
              
              <div className="flex-1 w-full flex">
                 <div className="flex flex-col justify-between text-[9px] text-gray-600 font-mono pr-2 pb-6">
                    {days.map(d => <div key={d} className="h-[22px] flex items-center">{d}</div>)}
                 </div>
                 <div className="flex-1 flex flex-col relative">
                    <div className="grid grid-cols-24 gap-[2px] flex-1">
                       {map.map((cell, idx) => {
                          let bg = '#0f172a';
                          if (cell.value > 10) bg = '#1e293b';
                          if (cell.value > 30) bg = '#450a0a';
                          if (cell.value > 60) bg = '#991b1b';
                          if (cell.value > 80) bg = '#ef4444';
                          
                          return (
                             <motion.div 
                                key={idx} 
                                className="w-full h-[22px] rounded-sm relative group"
                                style={{ backgroundColor: bg }}
                                animate={cell.isAnomaly && isCritical ? {
                                   backgroundColor: ['#991b1b', '#ef4444', '#991b1b'],
                                   boxShadow: ['0 0 0px #ef4444', '0 0 10px #ef4444', '0 0 0px #ef4444']
                                } : {}}
                                transition={{ duration: 1.5, repeat: Infinity }}
                             >
                               <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-black text-[10px] font-mono rounded opacity-0 group-hover:opacity-100 pointer-events-none z-50 whitespace-nowrap">
                                  {cell.day} {cell.hour}:00 - Score: {cell.value.toFixed(1)}
                               </div>
                             </motion.div>
                          );
                       })}
                    </div>
                    <div className="flex justify-between text-[9px] text-gray-600 font-mono pt-2 px-1">
                       <span>00:00</span>
                       <span>06:00</span>
                       <span>12:00</span>
                       <span>18:00</span>
                       <span>23:00</span>
                    </div>
                 </div>
              </div>
           </div>

           <AnimatePresence>
             {showBrief && (
               <motion.div 
                 initial={{ y: 50, opacity: 0 }}
                 animate={{ y: 0, opacity: 1 }}
                 transition={{ duration: 0.8, ease: "easeOut" }}
                 className="bg-[#1a0505]/80 backdrop-blur-2xl border border-red-500/20 rounded-3xl p-6 relative overflow-hidden ring-1 ring-red-500/30"
               >
                 <div className="absolute top-0 right-0 w-64 h-64 bg-red-500/5 blur-[80px] rounded-full pointer-events-none" />
                 
                 <h2 className="text-sm font-bold text-red-400 tracking-widest uppercase flex items-center gap-2 mb-6">
                    <Crosshair className="w-4 h-4" />
                    Threat Intelligence Brief
                 </h2>
                 
                 <p className="text-sm text-gray-300 leading-relaxed mb-6 font-mono">
                    Device <span className="text-white font-bold">{nodeId}</span> initiating slow data exfiltration via HTTPS. Payload distributions violate historical baselines, evading volumetric thresholds. <span className="text-red-400 font-bold">Recommended action: Isolate.</span>
                 </p>

                 <div className="space-y-4">
                    <div className="text-xs font-bold text-gray-500 tracking-widest uppercase mb-2">Feature Attribution (SHAP Values)</div>
                    
                    <div>
                       <div className="flex justify-between text-xs font-mono mb-1">
                          <span className="text-red-400 font-bold">external_traffic_ratio</span>
                          <span className="text-gray-400">+0.48</span>
                       </div>
                       <div className="w-full h-2 bg-black/50 rounded-full overflow-hidden">
                          <motion.div 
                             initial={{ width: 0 }}
                             animate={{ width: '85%' }}
                             transition={{ duration: 1, delay: 0.5 }}
                             className="h-full bg-red-500 shadow-[0_0_10px_#ef4444]" 
                          />
                       </div>
                    </div>

                    <div>
                       <div className="flex justify-between text-xs font-mono mb-1">
                          <span className="text-red-400 font-bold">avg_packet_size</span>
                          <span className="text-gray-400">+0.32</span>
                       </div>
                       <div className="w-full h-2 bg-black/50 rounded-full overflow-hidden">
                          <motion.div 
                             initial={{ width: 0 }}
                             animate={{ width: '65%' }}
                             transition={{ duration: 1, delay: 0.7 }}
                             className="h-full bg-red-500 shadow-[0_0_10px_#ef4444]" 
                          />
                       </div>
                    </div>

                    <div>
                       <div className="flex justify-between text-xs font-mono mb-1">
                          <span className="text-gray-400">connection_frequency</span>
                          <span className="text-gray-500">+0.05</span>
                       </div>
                       <div className="w-full h-2 bg-black/50 rounded-full overflow-hidden">
                          <div className="h-full bg-gray-500 w-[15%]" />
                       </div>
                    </div>
                 </div>

               </motion.div>
             )}
           </AnimatePresence>

        </div>
      </div>
    </div>
  );
}
