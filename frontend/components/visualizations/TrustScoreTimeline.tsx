'use client';

import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from 'recharts';
import { useEffect, useState } from 'react';

// Hardcoded data simulating a sharp drop at 02:14 AM
const generateTimeSeriesData = () => {
  const data = [];
  
  let time = new Date('2026-05-16T01:00:00');
  let currentScore = 94;
  
  while (time <= new Date('2026-05-16T02:20:00')) {
    const timeStr = time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    // Sharp drop exactly at 02:14 AM
    if (time.getHours() === 2 && time.getMinutes() >= 14) {
      if (time.getMinutes() === 14) {
          currentScore = 42; // Sudden plunge
      } else {
          currentScore = Math.max(30, currentScore - Math.random() * 3); // Stays critically low
      }
    } else {
       // Normal high trust variance
       currentScore = Math.max(90, Math.min(98, currentScore + (Math.random() - 0.5) * 3));
    }
    
    data.push({
      time: timeStr,
      score: currentScore,
      threshold: 60
    });
    
    time = new Date(time.getTime() + 60000); // 1 minute intervals
  }
  return data;
};

export default function TrustScoreTimeline() {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    setData(generateTimeSeriesData());
  }, []);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const score = payload[0].value;
      const isCritical = score < 60;
      
      return (
        <div className="bg-black/80 backdrop-blur-md border border-white/10 p-3 rounded-lg shadow-2xl z-50 text-white">
          <p className="text-gray-400 text-xs mb-1 font-mono">{label}</p>
          <div className="flex items-end gap-2">
            <span className={`text-2xl font-bold font-mono tracking-tighter ${
              isCritical ? 'text-red-500 drop-shadow-[0_0_8px_rgba(239,68,68,0.8)]' : 'text-[#3edcff]'
            }`}>
              {score.toFixed(1)}
            </span>
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-full absolute inset-0 text-xs p-4 pt-8">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
          <defs>
             <linearGradient id="scoreGlow" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#22c55e" stopOpacity={1}/>
                <stop offset="50%" stopColor="#eab308" stopOpacity={1}/>
                <stop offset="100%" stopColor="#ef4444" stopOpacity={1}/>
             </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis 
            dataKey="time" 
            stroke="rgba(255,255,255,0.3)" 
            tick={{ fill: 'rgba(255,255,255,0.5)' }} 
            tickMargin={10}
            minTickGap={30}
          />
          <YAxis 
            stroke="rgba(255,255,255,0.3)" 
            tick={{ fill: 'rgba(255,255,255,0.5)' }} 
            domain={[0, 100]} 
            ticks={[0, 20, 40, 60, 80, 100]}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ stroke: 'rgba(255,255,255,0.2)', strokeWidth: 1, strokeDasharray: '4 4' }} />
          
          <ReferenceLine y={60} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={2} label={{ position: 'insideTopLeft', value: 'THRESHOLD VIOLATION', fill: '#ef4444', fontSize: 10, fontWeight: 'bold' }} />
          
          <Line 
            type="monotone" 
            dataKey="score" 
            stroke="url(#scoreGlow)" 
            strokeWidth={3} 
            dot={false} 
            activeDot={{ r: 6, fill: '#000', stroke: '#ef4444', strokeWidth: 2 }} 
            isAnimationActive={true}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
