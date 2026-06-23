'use client';

import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from 'recharts';
import { useEffect, useState, useRef } from 'react';

export default function TrustScoreTimeline({ liveScores }: { liveScores: Record<string, number> }) {
  const [data, setData] = useState<any[]>([]);
  const [mounted, setMounted] = useState(false);
  const latestScoresRef = useRef(liveScores);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    latestScoresRef.current = liveScores;
  }, [liveScores]);

  useEffect(() => {
    const interval = setInterval(() => {
        const scores = Object.values(latestScoresRef.current);
        if (scores.length > 0) {
            let sum = 0;
            let minScore = 100;
            scores.forEach(score => {
                sum += score;
                if (score < minScore) minScore = score;
            });
            const globalAvg = sum / scores.length;
            
            const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
            
            setData(prev => {
                const newData = [...prev, { time: timeStr, score: minScore, avgScore: globalAvg, threshold: 40 }];
                if (newData.length > 60) return newData.slice(newData.length - 60);
                return newData;
            });
        }
    }, 1000);

    return () => {
       clearInterval(interval);
    };
  }, []);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const score = payload[0].value;
      const isCritical = score < 40;
      
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

  if (!mounted) {
    return <div className="w-full" style={{ height: 220 }} />;
  }

  return (
    <div className="w-full text-xs">
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ top: 10, right: 10, left: -25, bottom: 0 }}>
          <defs>
             <linearGradient id="scoreGlow" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3edcff" stopOpacity={0.5}/>
                <stop offset="100%" stopColor="#3edcff" stopOpacity={0.01}/>
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
          
          <ReferenceLine y={40} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={2} label={{ position: 'insideTopLeft', value: 'THRESHOLD VIOLATION', fill: '#ef4444', fontSize: 10, fontWeight: 'bold' }} />
          
          <Area 
            type="monotone" 
            dataKey="score" 
            stroke="#3edcff" 
            fill="url(#scoreGlow)"
            strokeWidth={3} 
            activeDot={{ r: 6, fill: '#000', stroke: '#ef4444', strokeWidth: 2 }} 
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
