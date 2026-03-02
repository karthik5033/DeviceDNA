'use client';

import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { useEffect, useState } from 'react';
import { io } from 'socket.io-client';

const socket = io('http://localhost:8000');

// Initial mock generation for 12 hours of trust score history (5-min intervals)
const generateTimeSeriesData = () => {
  const data = [];
  const now = new Date();
  
  let currentScore = 95;
  for (let i = 144; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 5 * 60000); // Backwards in 5 min steps
    currentScore = Math.max(85, Math.min(100, currentScore + (Math.random() - 0.5) * 5)); // Normal variance
    
    data.push({
      time: time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      rawTime: time,
      score: currentScore,
      threshold: 60 // Alert threshold line
    });
  }
  return data;
};

export default function TrustScoreTimeline() {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    // Generate initial flat baseline
    setData(generateTimeSeriesData());

    socket.on('new_alert', (alert) => {
       // Plunge the trust score when an anomaly happens
       setData(prev => {
          const newData = [...prev];
          // Take the very last data point and tank its score into the red
          const lastPoint = newData[newData.length - 1];
          const dropScore = alert.score || 25; // Critical threshold
          newData.push({
             ...lastPoint,
             time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
             score: dropScore
          });
          // Also remove the oldest point to keep the array length stable
          newData.shift();
          return newData;
       });
    });

    socket.on('device_isolated', (res) => {
       // Snap back to healthy if mitigated
       setData(prev => {
          const newData = [...prev];
          const lastPoint = newData[newData.length - 1];
          newData.push({
             ...lastPoint,
             time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
             score: 95
          });
          newData.shift();
          return newData;
       });
    });

    // Simulate standard ticking time advancement
    const timer = setInterval(() => {
       setData(prev => {
          const newData = [...prev];
          const lastPoint = newData[newData.length - 1];
          
          let nextScore = lastPoint.score;
          // Gradual automatic recovery if it was dropped
          if (nextScore < 90) {
              nextScore += 2;
          } else {
              // Normal variance
              nextScore = Math.max(85, Math.min(100, nextScore + (Math.random() - 0.5) * 3));
          }

          newData.push({
             time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
             score: nextScore,
             threshold: 60
          });
          newData.shift();
          return newData;
       });
    }, 5000); // Tick every 5 sec

    return () => {
       socket.off('new_alert');
       socket.off('device_isolated');
       clearInterval(timer);
    }
  }, []);

  // Custom detailed tooltip
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const score = payload[0].value;
      const isCritical = score < 60;
      
      return (
        <div className="bg-[#0b101e]/95 border border-[#1e293b] p-3 rounded-lg shadow-xl backdrop-blur z-50">
          <p className="text-gray-400 text-xs mb-1 font-mono">{label}</p>
          <div className="flex items-end gap-2">
            <span className={`text-2xl font-bold font-mono tracking-tighter ${
              isCritical ? 'text-red-500' : 'text-[#3edcff]'
            }`}>
              {score.toFixed(1)}
            </span>
            <span className="text-xs text-gray-500 mb-1">/ 100</span>
          </div>
          {isCritical && (
            <p className="text-xs text-red-500 mt-1 uppercase tracking-wider font-bold animate-pulse">● Anomaly Detected</p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full h-full min-h-[300px] flex flex-col relative text-xs">
      {/* Absolute Header Overlay */}
      <div className="absolute top-0 left-0 p-4 z-10 w-full flex justify-between items-center pointer-events-none">
        <h3 className="text-gray-300 font-semibold text-sm">System-Wide Trust Trajectory</h3>
        <span className="text-[#3edcff] font-mono bg-[#3edcff]/10 border border-[#3edcff]/30 px-2 py-1 rounded hidden sm:block">
          Live Stream
        </span>
      </div>

      <div className="flex-1 w-full mt-10">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 20, right: 20, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3edcff" stopOpacity={0.8}/>
                <stop offset="50%" stopColor="#eab308" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0.8}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
            <XAxis 
              dataKey="time" 
              stroke="#475569" 
              tick={{ fill: '#64748b' }} 
              tickMargin={10}
              minTickGap={30}
            />
            <YAxis 
              stroke="#475569" 
              tick={{ fill: '#64748b' }} 
              domain={[0, 100]} 
              ticks={[0, 20, 40, 60, 80, 100]}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#334155', strokeWidth: 1, strokeDasharray: '4 4' }} />
            
            {/* The Alert Threshold Line */}
            <Line 
              type="monotone" 
              dataKey="threshold" 
              stroke="#ef4444" 
              strokeWidth={1} 
              strokeDasharray="4 4" 
              dot={false} 
              activeDot={false}
              isAnimationActive={false}
            />
            
            {/* The dynamic Trust Score Line */}
            <Line 
              type="monotone" 
              dataKey="score" 
              stroke="url(#scoreGradient)" 
              strokeWidth={3} 
              dot={false} 
              activeDot={{ r: 6, fill: '#070b14', stroke: '#3edcff', strokeWidth: 2 }} 
              animationDuration={800}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
