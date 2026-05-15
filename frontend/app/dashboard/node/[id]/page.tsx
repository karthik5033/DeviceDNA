'use client';

import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert, Activity, ArrowDownToLine, Crosshair, Cpu, Network, ShieldX } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useParams } from 'next/navigation';

// Mock Heatmap Data (Days x Hours)
const generateHeatmap = () => {
  const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
  const map = [];
  for (let d = 0; d < 7; d++) {
    for (let h = 0; h < 24; h++) {
      // Wednesday (index 2), 2AM - 4AM are anomalous
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

export default function NodeInspection() {
  const params = useParams();
  const nodeId = params.id || 'SIM-0001';

  const [score, setScore] = useState(91);
  const [showAlert, setShowAlert] = useState(false);
  const [showBrief, setShowBrief] = useState(false);

  const { days, map } = generateHeatmap();

  // Cinematic Sequence
  useEffect(() => {
    // 1. Initial delay
    const t1 = setTimeout(() => {
      // 2. Trigger score drop
      const interval = setInterval(() => {
        setScore(prev => {
          if (prev <= 52) {
            clearInterval(interval);
            return 52;
          }
          return prev - 2;
        });
      }, 50);

      // 3. Slide in alert banner
      setTimeout(() => setShowAlert(true), 400);

      // 4. Fade in Threat Brief
      setTimeout(() => setShowBrief(true), 1200);

    }, 1500);

    return () => clearTimeout(t1);
  }, []);

  const getScoreColor = (val: number) => {
    if (val >= 80) return '#22c55e'; // Green
    if (val >= 60) return '#eab308'; // Amber
    return '#ef4444'; // Red
  };

  const getScoreGlow = (val: number) => {
    if (val >= 80) return 'rgba(34, 197, 94, 0.4)';
    if (val >= 60) return 'rgba(234, 179, 8, 0.4)';
    return 'rgba(239, 68, 68, 0.6)';
  };

  const currentColor = getScoreColor(score);
  const currentGlow = getScoreGlow(score);
  const isCritical = score < 60;

  // SVG Donut Math
  const radius = 120;
  const circumference = 2 * Math.PI * radius;
  // Sweep from 0 to 100
  const strokeDashoffset = circumference - (score / 100) * circumference;

  return (
    <div className="relative min-h-screen bg-[#04060a] text-white p-6 font-sans overflow-hidden flex flex-col items-center justify-center">
      
      {/* Background Rim Lighting & Particles */}
      <div className="absolute inset-0 pointer-events-none z-0">
        <div className="absolute inset-0 bg-[#020408]" />
        {/* Dynamic Center Glow */}
        <div 
           className="absolute top-[50%] left-[50%] -translate-x-1/2 -translate-y-1/2 w-[60%] h-[60%] rounded-full blur-[250px] transition-colors duration-500"
           style={{ backgroundColor: isCritical ? '#7f1d1d' : '#0ea5e9', opacity: isCritical ? 0.3 : 0.1 }}
        />
        {/* Cinematic Red Rim Lighting */}
        <div className={cn("absolute bottom-0 left-0 w-full h-32 bg-gradient-to-t from-red-900/40 to-transparent transition-opacity duration-1000", isCritical ? "opacity-100" : "opacity-0")} />
        <div className={cn("absolute top-0 left-0 w-32 h-full bg-gradient-to-r from-red-900/30 to-transparent transition-opacity duration-1000", isCritical ? "opacity-100" : "opacity-0")} />
      </div>

      {/* Alert Banner */}
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

      {/* Main Content */}
      <div className="relative z-10 w-full max-w-6xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-start mt-12">
        
        {/* Left Column: Gauge & Info */}
        <div className="flex flex-col gap-8">
          
          {/* Node Identity */}
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

             {/* The Gauge */}
             <div className="relative flex justify-center items-center py-8">
                {/* Outer Glow */}
                <div 
                   className="absolute w-64 h-64 rounded-full blur-[40px] transition-all duration-300"
                   style={{ backgroundColor: currentColor, opacity: 0.15 }}
                />
                
                {/* SVG Donut */}
                <svg width="280" height="280" viewBox="0 0 280 280" className="transform -rotate-90 relative z-10">
                  {/* Background Track */}
                  <circle 
                    cx="140" cy="140" r={radius}
                    fill="transparent"
                    stroke="#1e293b"
                    strokeWidth="16"
                  />
                  {/* Progress Track */}
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

                {/* Center Value */}
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
        </div>

        {/* Right Column: Heatmap & Brief */}
        <div className="flex flex-col gap-8">
           
           {/* CUSUM Drift Heatmap */}
           <div className="bg-white/[0.02] backdrop-blur-xl border border-white/[0.05] rounded-3xl p-6 ring-1 ring-white/5 flex flex-col">
              <div className="flex justify-between items-center mb-6">
                 <h2 className="text-sm font-bold text-gray-300 tracking-widest uppercase flex items-center gap-2">
                    <Activity className="w-4 h-4 text-gray-400" />
                    CUSUM Drift Heatmap
                 </h2>
                 <span className="text-xs font-mono text-gray-500">7-Day Rolling</span>
              </div>
              
              {/* Heatmap Grid */}
              <div className="flex-1 w-full flex">
                 <div className="flex flex-col justify-between text-[9px] text-gray-600 font-mono pr-2 pb-6">
                    {days.map(d => <div key={d} className="h-[22px] flex items-center">{d}</div>)}
                 </div>
                 <div className="flex-1 flex flex-col relative">
                    <div className="grid grid-cols-24 gap-[2px] flex-1">
                       {map.map((cell, idx) => {
                          // Map value to color
                          let bg = '#0f172a'; // base dark
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
                               {/* Hover Tooltip */}
                               <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-black text-[10px] font-mono rounded opacity-0 group-hover:opacity-100 pointer-events-none z-50 whitespace-nowrap">
                                  {cell.day} {cell.hour}:00 - Score: {cell.value.toFixed(1)}
                               </div>
                             </motion.div>
                          );
                       })}
                    </div>
                    {/* X-axis labels */}
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

           {/* Threat Intelligence Brief */}
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

                 {/* Feature Attribution Bars */}
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
