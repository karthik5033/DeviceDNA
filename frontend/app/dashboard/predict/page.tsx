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
        <div className="bg-[#111827] border border-[#1e293b] rounded-xl p-6 flex flex-col shadow-lg hover:border-[#334155] transition-colors relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4 opacity-10">
             <BrainCircuit size={100} />
          </div>

          <h2 className="text-lg font-semibold text-gray-200 mb-4 flex items-center gap-2 relative z-10">
            <ShieldCheck className="text-green-500 mb-0.5" />
            LSTM AI Forecasting
          </h2>
          <p className="text-sm text-gray-400 mb-6 leading-relaxed relative z-10">
            The Multi-Dimensional <span className="text-[#3edcff] font-semibold">Long Short-Term Memory (LSTM)</span> recurrent neural network continuously predicts the subsequent state of telemetry vectors. <br/><br/>
            When live data diverges radically from the 12-hour predicted sequence, anomalous threat events are probabilistically isolated prior to complete infiltration.
          </p>

          <div className="flex-1 mt-auto border border-dashed border-[#334155] bg-[#070b14]/50 rounded-lg flex flex-col items-center justify-center text-gray-500 font-mono text-sm p-4 text-center leading-loose relative z-10 group">
             <Activity className="w-8 h-8 text-[#3edcff] mb-3 animate-pulse opacity-50 group-hover:opacity-100 transition-opacity" />
             Active Forecasting Sequence<br/>
             <span className="text-xs text-yellow-500">Tracking (14-Dim) Vectors...</span>
          </div>
        </div>
      </div>
    </div>
  );
}
