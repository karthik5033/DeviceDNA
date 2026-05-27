'use client';

import DriftHeatmap from '@/components/visualizations/DriftHeatmap';
import { BrainCircuit, Activity, BarChart3, AlertTriangle, ShieldCheck } from 'lucide-react';
import { cn } from '@/lib/utils';

export default function PredictiveRiskPage() {
  const stats = [
    { title: 'Current CUSUM Accumulator', value: '14.2%', desc: 'Below 60% threshold', icon: Activity, color: 'text-green-500' },
    { title: 'Predicted Threat Risk', value: 'Low', desc: 'Next 24 Hours', icon: BrainCircuit, color: 'text-[#3edcff]' },
    { title: 'LSTM Forecast Variance', value: '2.4%', desc: 'MSE Expected', icon: BarChart3, color: 'text-yellow-500' },
  ];

  return (
    <div className="flex flex-col gap-6 max-w-[1600px] mx-auto fade-in">
      <div>
        <h1 className="text-3xl font-bold tracking-tighter mb-1 font-sans">Predictive Risk & Drift</h1>
        <p className="text-gray-400 text-sm">Statistical CUSUM accumulation mapping and LSTM forecasting logic.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stats.map((stat, idx) => (
          <div key={idx} className="bg-[#111827] border border-[#1e293b] rounded-xl p-6 relative overflow-hidden group hover:border-[#334155] transition-all">
            <div className="flex justify-between items-start mb-4">
              <span className="text-sm font-medium text-gray-400">{stat.title}</span>
              <stat.icon className={cn("w-5 h-5", stat.color)} />
            </div>
            <div className="flex flex-col gap-1 z-10 relative">
              <span className="text-4xl font-bold font-mono tracking-tight">{stat.value}</span>
              <span className="text-xs text-gray-500 uppercase tracking-widest">{stat.desc}</span>
            </div>
            <div className={cn("absolute -bottom-10 -right-10 w-24 h-24 blur-[40px] rounded-full opacity-10 group-hover:opacity-20 transition-all duration-500", stat.color.replace('text-', 'bg-'))} />
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-4 pb-12">
        {/* The Core Drift Map Layer */}
        <div className="lg:col-span-2 bg-[#111827] border border-[#1e293b] rounded-xl p-6 min-h-[480px] shadow-2xl relative overflow-hidden">
          {/* Subtle Background glow */}
          <div className="absolute top-0 right-0 w-full h-full bg-gradient-to-bl from-red-500/5 to-transparent pointer-events-none" />
          <DriftHeatmap />
        </div>
        
        {/* Right side LSTM Forecasting Pane */}
        <div className="bg-[#111827] border border-[#1e293b] rounded-xl p-8 flex flex-col shadow-lg hover:border-[#334155] transition-all relative overflow-hidden group">
          <div className="absolute -top-4 -right-4 p-4 opacity-5 text-[#3edcff] transition-transform duration-700 group-hover:scale-110">
             <BrainCircuit size={160} strokeWidth={1} />
          </div>

          <h2 className="text-xl font-bold text-gray-100 mb-5 flex items-center gap-3 relative z-10 tracking-tight">
            <ShieldCheck className="text-[#3edcff] w-6 h-6" />
            LSTM AI Forecasting
          </h2>
          <div className="space-y-4 text-sm text-gray-400 leading-relaxed relative z-10">
            <p>
              The Multi-Dimensional <span className="text-gray-200 font-medium">Long Short-Term Memory (LSTM)</span> recurrent neural network continuously predicts the subsequent state of telemetry vectors.
            </p>
            <p>
              When live data diverges radically from the 12-hour predicted sequence, anomalous threat events are probabilistically isolated prior to complete infiltration.
            </p>
          </div>

          <div className="mt-8 bg-[#070b14] border border-[#1e293b] rounded-xl p-5 relative z-10 overflow-hidden shadow-inner group/chart">
             {/* Glowing accent line */}
             <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-[#3edcff]/50 to-transparent" />
             
             <div className="flex justify-between items-start mb-6">
               <div>
                 <div className="text-[#3edcff] font-mono text-[10px] uppercase tracking-widest mb-1.5 flex items-center gap-2">
                   <div className="w-1.5 h-1.5 rounded-full bg-[#3edcff] shadow-[0_0_5px_#3edcff] animate-pulse" />
                   Active Forecasting Sequence
                 </div>
                 <div className="text-gray-300 font-semibold text-sm">Tracking (14-Dim) Vectors</div>
               </div>
               <Activity className="w-5 h-5 text-[#3edcff] opacity-70 group-hover/chart:opacity-100 transition-opacity" />
             </div>

             {/* Fake mock graph bars */}
             <div className="flex items-end justify-between h-16 gap-1.5">
                {[30, 45, 25, 60, 40, 85, 55, 35, 75, 45, 90, 65].map((h, i) => (
                  <div key={i} className="w-full bg-[#1e293b]/50 rounded-t-[2px] relative overflow-hidden" style={{ height: '100%' }}>
                     <div 
                       className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-[#3edcff]/80 to-[#3edcff] rounded-t-[2px] transition-all duration-1000 ease-in-out opacity-80 group-hover/chart:opacity-100 group-hover/chart:animate-pulse" 
                       style={{ height: `${h}%`, animationDelay: `${i * 100}ms` }}
                     ></div>
                  </div>
                ))}
             </div>
          </div>
        </div>
      </div>
    </div>
  );
}
