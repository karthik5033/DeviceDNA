'use client';

import { useState } from 'react';
import { ShieldAlert, Terminal, Eye, AlertTriangle, ArrowRight, Zap, Flame, Fingerprint, RefreshCcw } from 'lucide-react';
import { cn } from '@/lib/utils';

// Faked Live Alert Pipeline
const MOCK_ALERTS = [
  { id: 'ALT-9402', device: 'SIM-0012', severity: 'critical', type: 'C2 Beaconing', model: 'Isolation Forest', message: 'Categorical port anomaly. Exfiltrating 443 to external static IP block 192.168.10.x', time: '1m ago', score: 32 },
  { id: 'ALT-9401', device: 'SIM-0044', severity: 'high', type: 'Lateral Movement', model: 'GraphSAGE GNN', message: 'Device mapped scanning adjacent IP camera block 10.0.1.x using SMB protocols.', time: '12m ago', score: 48 },
  { id: 'ALT-9400', device: 'SIM-0003', severity: 'high', type: 'Policy Violation', model: 'NLP Engine', message: '"Thermostats shall not contact Chinese endpoints." DNS resolution flagged.', time: '15m ago', score: 55 },
  { id: 'ALT-9399', device: 'SIM-0021', severity: 'medium', type: 'Slow Exfiltration', model: 'CUSUM Drift', message: 'Consecutive bytes uploaded threshold triggered over 72 hour cumulative window.', time: '2h ago', score: 62 },
  { id: 'ALT-9398', device: 'SIM-0019', severity: 'low', type: 'Suspicious Flow', model: 'LSTM TimeSeries', message: 'Predicted temporal flow variance exceeded MSE threshold by 4%. Watch.', time: '5h ago', score: 78 },
];

export default function AlertsPage() {
  const [filter, setFilter] = useState('all');

  return (
    <div className="flex flex-col gap-6 max-w-[1600px] mx-auto fade-in">
      
      {/* Alert Ribbon Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 border-b border-[#1e293b] pb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tighter mb-1 font-sans flex items-center gap-3">
            <ShieldAlert className="text-red-500" size={28} />
            Actionable Threat Alerts
          </h1>
          <p className="text-gray-400 text-sm">Real-time fusion engine reporting anomalies from the 5 ML Pillars.</p>
        </div>
        
        {/* Toggles */}
        <div className="flex gap-2 bg-[#111827] border border-[#1e293b] rounded-lg p-1">
          {['all', 'critical', 'high', 'medium'].map((level) => (
            <button
              key={level}
              onClick={() => setFilter(level)}
              className={cn(
                "px-4 py-1.5 rounded-md text-sm font-medium capitalize transition-all",
                filter === level 
                  ? "bg-[#1e293b] text-white shadow-sm" 
                  : "text-gray-500 hover:text-gray-300"
              )}
            >
              {level}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Queue vs Detail */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        
        {/* Full Alert Feed List */}
        <div className="xl:col-span-2 flex flex-col gap-3">
          {MOCK_ALERTS.filter(a => filter === 'all' || a.severity === filter).map((alert) => (
            <div key={alert.id} className="bg-[#111827] border border-[#1e293b] rounded-xl p-4 hover:border-[#334155] transition-colors relative group overflow-hidden pl-5">
              
              {/* Severity Side Bar */}
              <div className={cn("absolute left-0 top-0 w-1.5 h-full", 
                alert.severity === 'critical' ? 'bg-red-500 shadow-[0_0_15px_#ef4444]' :
                alert.severity === 'high' ? 'bg-orange-500 shadow-[0_0_15px_#f97316]' :
                alert.severity === 'medium' ? 'bg-yellow-500' : 'bg-gray-500'
              )}></div>

              <div className="flex justify-between items-start mb-2">
                <div className="flex gap-3 items-center">
                  <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-[#1e293b] text-gray-300">{alert.id}</span>
                  <span className="font-mono text-[#3edcff] font-bold tracking-tight text-sm flex items-center gap-1">
                    <Terminal size={14} /> {alert.device}
                  </span>
                </div>
                <div className="flex gap-4 items-center">
                  <span className="text-xs font-mono text-gray-500">{alert.time}</span>
                  <div className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-[#070b14] border border-[#1e293b]">
                    <span className={alert.score < 40 ? 'text-red-500 font-bold' : alert.score < 60 ? 'text-orange-500' : 'text-yellow-500'}>
                      {alert.score.toFixed(1)}
                    </span>
                    <span className="text-gray-600">Trust</span>
                  </div>
                </div>
              </div>

              <h2 className="text-lg font-bold text-gray-200 mb-1 flex items-center gap-2 tracking-tight">
                {alert.severity === 'critical' ? <Flame size={18} className="text-red-500 animate-pulse" /> : <AlertTriangle size={18} className="text-orange-500" />}
                {alert.type}
              </h2>
              
              <p className="text-sm text-gray-400 mb-4">{alert.message}</p>
              
              <div className="flex justify-between items-center text-xs">
                 <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-[#1e293b]/50 border border-[#334155] text-gray-400 font-medium">
                    <Fingerprint size={12} className="text-[#3edcff]" />
                    Detected By: <span className="text-white">{alert.model}</span>
                 </div>
                 
                 <button className="flex items-center gap-1 text-[#3edcff] hover:text-white transition-colors group-hover:underline decoration-[#3edcff] underline-offset-4">
                    View Explainable Proof <ArrowRight size={14} />
                 </button>
              </div>
            </div>
          ))}
        </div>

        {/* Explainable AI Action Panel */}
        <div className="hidden xl:flex flex-col gap-4">
          <div className="bg-[#111827] border border-[#1e293b] rounded-xl p-6 shadow-xl sticky top-24">
            <h2 className="text-lg font-bold text-gray-200 flex items-center gap-2 mb-4 border-b border-[#1e293b] pb-2">
              <Eye className="text-[#3edcff]" /> Threat Context
            </h2>
            
            <p className="text-sm text-gray-400 mb-6 italic leading-relaxed">
              Select an alert from the queue to run the <span className="text-[#3edcff] font-bold">SHAP Explainable AI</span> explainer algorithm.<br/><br/>
              SHAP will deconstruct the neural network's decision boundary and pinpoint exactly which bytes, ports, or duration features triggered the anomaly flag.
            </p>
            
            <div className="border border-dashed border-[#1e293b] bg-[#070b14]/50 rounded-lg h-48 flex flex-col items-center justify-center text-gray-600 space-y-3 p-4 text-center">
              <RefreshCcw size={24} className="animate-spin duration-[3000ms]" />
              <span className="text-xs font-mono">Awaiting Feature Array Input...</span>
            </div>
            
            <button className="w-full mt-6 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/50 rounded-lg text-sm font-bold transition-colors flex justify-center items-center gap-2">
              <Zap size={16} /> Isolate Device Immediately
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
