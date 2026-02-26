'use client';

import { useMemo } from 'react';

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

const getColorBin = (score: number) => {
  if (score < 20) return 'bg-[#1e293b] hover:bg-[#334155]'; // Baseline
  if (score < 40) return 'bg-yellow-900/40 border border-yellow-700/50 hover:bg-yellow-900/60';
  if (score < 60) return 'bg-orange-600/50 border border-orange-500/50 hover:bg-orange-600/70';
  if (score < 80) return 'bg-red-500/80 border border-red-500 hover:bg-red-500';
  return 'bg-[#ff003c] border border-red-400 shadow-[0_0_12px_rgba(255,0,60,0.6)] z-10'; // High alert
};

export default function DriftHeatmap() {
  const data = useMemo(() => generateHeatmapData(), []);
  
  return (
    <div className="w-full flex flex-col h-full text-xs">
      <div className="flex justify-between items-start md:items-center mb-6 flex-col md:flex-row gap-4">
        <div>
          <h3 className="text-gray-200 font-semibold text-lg flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-orange-500 animate-pulse" />
            CUSUM Exfiltration Drift
          </h3>
          <p className="text-gray-500 text-sm">7-Day hour-by-hour statistical accumulation tracking evaluating slow-burn anomalies.</p>
        </div>
        
        {/* Heatmap Legend */}
        <div className="flex items-center gap-2 text-xs text-gray-500 bg-[#070b14] p-2 rounded-md border border-[#1e293b]">
          <span>Baseline</span>
          <div className="flex gap-1">
            <div className="w-4 h-4 rounded-sm bg-[#1e293b]"></div>
            <div className="w-4 h-4 rounded-sm bg-yellow-900/40 border border-yellow-700/50"></div>
            <div className="w-4 h-4 rounded-sm bg-orange-600/50 border border-orange-500/50"></div>
            <div className="w-4 h-4 rounded-sm bg-red-500/80 border border-red-500"></div>
            <div className="w-4 h-4 rounded-sm bg-[#ff003c] shadow-[0_0_8px_rgba(255,0,60,0.5)]"></div>
          </div>
          <span>Critical</span>
        </div>
      </div>

      <div className="flex flex-1">
        {/* Y Axis Labels (Days) */}
        <div className="flex flex-col justify-around pr-4 text-gray-500 font-mono">
          {data.map(d => (
            <div key={d.day} className="h-full flex items-center">{d.day}</div>
          ))}
        </div>
        
        {/* Heatmap Grid */}
        <div className="flex flex-col flex-1">
          <div className="flex flex-col flex-1 justify-around gap-1">
             {data.map(row => (
               <div key={row.day} className="flex flex-1 gap-1">
                 {row.bins.map(bin => (
                   <div 
                     key={`${row.day}-${bin.hour}`} 
                     className={`flex-1 rounded-[2px] cursor-pointer transition-all duration-300 relative group ${getColorBin(bin.score)}`}
                   >
                     {/* Custom Tooltip */}
                     <div className="opacity-0 group-hover:opacity-100 absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-[#000] border border-[#334155] text-white p-2 rounded text-[10px] pointer-events-none w-32 z-50 transition-opacity">
                        <div className="font-bold border-b border-[#334155] pb-1 mb-1 text-center">
                          {row.day}, {bin.hour.toString().padStart(2, '0')}:00
                        </div>
                        <div className="flex justify-between items-center text-gray-400">
                          Drift Score: <span className={bin.score > 60 ? 'text-red-500 font-bold' : 'text-[#3edcff]'}>{bin.score.toFixed(1)}%</span>
                        </div>
                        {bin.score > 60 && <div className="text-red-500 text-center mt-1 font-bold">CUSUM Alarm Triggered</div>}
                     </div>
                   </div>
                 ))}
               </div>
             ))}
          </div>
          
          {/* X Axis Labels (Hours) */}
          <div className="flex justify-between mt-3 text-gray-500 font-mono px-2">
            <span>00:00</span>
            <span>06:00</span>
            <span>12:00</span>
            <span>18:00</span>
            <span>23:00</span>
          </div>
        </div>
      </div>
    </div>
  );
}
