'use client';

import { useState, useEffect } from 'react';

const generateHeatmapData = () => {
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const hours = Array.from({ length: 24 }, (_, i) => i);
  
  return days.map(day => ({
    day,
    bins: hours.map(hour => {
      // Create a pattern: Low drift usually, but high drift exclusively on Tue/Wed nights
      // to simulate the "Slow Data Exfiltration" scenario
      let driftScore = Math.random() * 15; // Normal noise 0-15%
      
      if ((day === 'Tue' || day === 'Wed') && hour >= 18 && hour <= 23) {
        driftScore = 60 + Math.random() * 40; // Exfiltration spike (60-100%)
      } else if (day === 'Wed' && hour < 4) {
        driftScore = 40 + Math.random() * 20; // Lingering morning drift
      }

      return { hour, score: driftScore };
    })
  }));
};

const getBarClasses = (score: number) => {
  if (score < 20) return 'bg-[#3edcff] opacity-10 hover:opacity-40'; // Baseline
  if (score < 40) return 'bg-gradient-to-t from-yellow-900/50 to-yellow-400 opacity-80 shadow-[0_0_8px_rgba(250,204,21,0.3)] hover:brightness-125';
  if (score < 60) return 'bg-gradient-to-t from-orange-900/50 to-orange-500 opacity-90 shadow-[0_0_12px_rgba(249,115,22,0.4)] hover:brightness-125';
  if (score < 80) return 'bg-gradient-to-t from-red-900/50 to-red-500 opacity-100 shadow-[0_0_15px_rgba(239,68,68,0.6)] hover:brightness-125';
  return 'bg-gradient-to-t from-[#ff003c]/20 to-[#ff003c] shadow-[0_0_20px_rgba(255,0,60,0.8)] animate-pulse border-t-2 border-white/80';
};

export default function DriftHeatmap() {
  const [data, setData] = useState<ReturnType<typeof generateHeatmapData>>([]);

  useEffect(() => {
    setData(generateHeatmapData());
  }, []);
  
  return (
    <div className="w-full flex flex-col h-full text-xs">
      <div className="flex justify-between items-start md:items-center mb-6 flex-col md:flex-row gap-4">
        <div>
          <h3 className="text-gray-200 font-semibold text-lg flex items-center gap-2 tracking-tight">
            <div className="w-2 h-2 rounded-full bg-[#ff003c] animate-pulse shadow-[0_0_10px_#ff003c]" />
            CUSUM Exfiltration Drift
          </h3>
          <p className="text-gray-500 text-sm mt-0.5">7-Day hour-by-hour statistical accumulation tracking evaluating slow-burn anomalies.</p>
        </div>
        
        {/* Heatmap Legend */}
        <div className="flex items-center gap-4 text-xs text-gray-400 bg-[#070b14] px-5 py-2.5 rounded-full border border-[#1e293b] shadow-inner">
          <span className="font-mono uppercase text-[10px] tracking-widest text-[#3edcff]/50">Baseline</span>
          <div className="flex gap-2 items-end h-4">
            <div className="w-1.5 h-1 bg-[#3edcff] opacity-20 rounded-t-sm"></div>
            <div className="w-1.5 h-2 bg-gradient-to-t from-yellow-900/50 to-yellow-400 rounded-t-sm"></div>
            <div className="w-1.5 h-2.5 bg-gradient-to-t from-orange-900/50 to-orange-500 rounded-t-sm"></div>
            <div className="w-1.5 h-3 bg-gradient-to-t from-red-900/50 to-red-500 rounded-t-sm"></div>
            <div className="w-1.5 h-4 bg-gradient-to-t from-[#ff003c]/20 to-[#ff003c] rounded-t-sm animate-pulse border-t border-white/50"></div>
          </div>
          <span className="font-mono uppercase text-[10px] tracking-widest text-[#ff003c] drop-shadow-[0_0_5px_#ff003c]">Critical</span>
        </div>
      </div>

      <div className="flex flex-1 mt-2">
        {/* Y Axis Labels (Days) */}
        <div className="flex flex-col justify-around pr-6 text-gray-500 font-mono text-xs uppercase tracking-widest">
          {data.map(d => (
            <div key={d.day} className="h-10 flex items-center justify-end">{d.day}</div>
          ))}
        </div>
        
        {/* Heatmap Grid - Seismograph Style */}
        <div className="flex flex-col flex-1 relative">
          
          {/* Subtle Background Grid Lines */}
          <div className="absolute inset-0 pointer-events-none flex justify-between">
             {[0, 6, 12, 18, 23].map(h => (
               <div key={h} className="w-px h-full bg-[#1e293b]/30"></div>
             ))}
          </div>

          <div className="flex flex-col flex-1 justify-around gap-4 z-10">
             {data.map(row => (
               <div key={row.day} className="flex h-10 items-end border-b border-[#1e293b]/80 relative group">
                 {row.bins.map(bin => (
                   <div 
                     key={`${row.day}-${bin.hour}`} 
                     className="flex-1 flex justify-center h-full relative group/bar"
                   >
                     {/* The Equalizer Bar */}
                     <div 
                        className={`absolute bottom-0 w-[60%] md:w-[70%] rounded-t-[2px] transition-all duration-500 ease-out ${getBarClasses(bin.score)}`}
                        style={{ height: `${Math.max(4, bin.score)}%` }}
                     />
                     
                     {/* Hover Interaction Area */}
                     <div className="absolute inset-0 z-20 peer cursor-pointer" />
                     
                     {/* Tooltip */}
                     <div className="opacity-0 peer-hover:opacity-100 absolute bottom-full left-1/2 -translate-x-1/2 mb-3 bg-[#070b14]/95 backdrop-blur-sm border border-[#3edcff]/30 text-white p-3 rounded-lg text-xs pointer-events-none w-44 z-50 transition-opacity shadow-[0_0_30px_rgba(62,220,255,0.15)]">
                        <div className="font-bold border-b border-[#1e293b] pb-2 mb-2 text-center text-gray-200 tracking-wide">
                          {row.day}, {bin.hour.toString().padStart(2, '0')}:00
                        </div>
                        <div className="flex justify-between items-center text-gray-400 font-mono">
                          Drift Score: <span className={bin.score > 60 ? 'text-[#ff003c] font-bold drop-shadow-[0_0_2px_#ff003c]' : 'text-[#3edcff]'}>{bin.score.toFixed(1)}%</span>
                        </div>
                        {bin.score > 60 && <div className="text-[#070b14] bg-[#ff003c] rounded px-2 py-1 text-center mt-2 text-[10px] font-bold animate-pulse tracking-widest uppercase">CUSUM ALARM</div>}
                     </div>
                   </div>
                 ))}
               </div>
             ))}
          </div>
          
          {/* X Axis Labels (Hours) */}
          <div className="flex justify-between mt-4 text-gray-500 font-mono text-[10px] tracking-widest px-2">
            <div>00:00</div>
            <div>06:00</div>
            <div>12:00</div>
            <div>18:00</div>
            <div>23:00</div>
          </div>
        </div>
      </div>
    </div>
  );
}
