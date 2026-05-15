'use client';

import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine } from 'recharts';
import { useEffect, useState, useRef } from 'react';
import { io } from 'socket.io-client';

export default function TrustScoreTimeline() {
  const [data, setData] = useState<any[]>([]);
  const batchRef = useRef<{ scores: number[]; currentSecond: number }>({ 
      scores: [], 
      currentSecond: Math.floor(Date.now() / 1000) 
  });

  useEffect(() => {
    const socket = io('http://localhost:8000', {
       transports: ['websocket'],
    });

    socket.on('trust_update', (msg) => {
        const now = Date.now();
        const currentSec = Math.floor(now / 1000);
        
        if (currentSec > batchRef.current.currentSecond) {
            // Flush the old batch to state
            if (batchRef.current.scores.length > 0) {
                const avgScore = batchRef.current.scores.reduce((a, b) => a + b, 0) / batchRef.current.scores.length;
                const timeStr = new Date(batchRef.current.currentSecond * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                
                setData(prev => {
                    const newData = [...prev, { time: timeStr, score: avgScore, threshold: 60 }];
                    // Keep maximum of 60 points rolling window
                    if (newData.length > 60) return newData.slice(newData.length - 60);
                    return newData;
                });
            }
            // Reset for the new second
            batchRef.current = { scores: [msg.score], currentSecond: currentSec };
        } else {
            // Accumulate in current second
            batchRef.current.scores.push(msg.score);
        }
    });

    return () => {
       socket.disconnect();
    };
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
