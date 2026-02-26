'use client';

import { Activity, ShieldAlert, Wifi, Server } from 'lucide-react';
import { cn } from '@/lib/utils';
import NetworkTopologyMap from '@/components/visualizations/NetworkTopologyMap';
import TrustScoreTimeline from '@/components/visualizations/TrustScoreTimeline';

export default function DashboardOverview() {
  
  // Fake data while real APIs are pending
  const kpis = [
    { title: 'Global Trust Score', value: '94.2', desc: 'Slightly degraded (-1.2)', icon: ShieldAlert, color: 'text-[#3edcff]' },
    { title: 'Active Devices', value: '50', desc: '100% online', icon: Wifi, color: 'text-green-500' },
    { title: 'Network Flow Rate', value: '1.2k', desc: 'Flows / minute', icon: Activity, color: 'text-yellow-500' },
    { title: 'Active Alerts', value: '3', desc: '2 High, 1 Med', icon: Server, color: 'text-red-500' },
  ];

  return (
    <div className="flex flex-col gap-6 max-w-[1600px] mx-auto">
      
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tighter mb-1">SOC Overview</h1>
        <p className="text-gray-400">Live multi-dimensional telemetry and trust analysis.</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {kpis.map((kpi, idx) => (
          <div key={idx} className="bg-[#111827] border border-[#1e293b] rounded-xl p-6 hover:border-[#334155] transition-colors relative overflow-hidden group">
            <div className="flex justify-between items-start mb-4">
              <span className="text-sm font-medium text-gray-400">{kpi.title}</span>
              <kpi.icon className={cn("w-5 h-5", kpi.color)} />
            </div>
            
            <div className="flex flex-col gap-1">
              <span className="text-4xl font-bold font-mono tracking-tight">{kpi.value}</span>
              <span className="text-xs text-gray-500">{kpi.desc}</span>
            </div>
            
            <div className={cn("absolute -bottom-10 -right-10 w-24 h-24 blur-3xl rounded-full opacity-10 group-hover:opacity-20 transition-opacity", kpi.color.replace('text-', 'bg-'))} />
          </div>
        ))}
      </div>

      {/* Main Grid Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1">
        
        {/* Network Topology Map */}
        <div className="lg:col-span-2 bg-[#111827] border border-[#1e293b] rounded-xl flex flex-col min-h-[500px] relative overflow-hidden shadow-2xl">
          <div className="p-4 border-b border-[#1e293b] flex justify-between items-center z-10 bg-[#111827]">
            <h2 className="font-semibold text-sm text-gray-300">Live Trust Score Topology</h2>
            <div className="flex gap-2 text-xs">
              <span className="px-2 py-1 rounded bg-[#1e293b] text-gray-300 border border-[#3edcff]/50 font-medium">Live</span>
              <span className="px-2 py-1 rounded border border-[#1e293b] text-gray-500 hover:text-white transition-colors cursor-pointer">1H</span>
              <span className="px-2 py-1 rounded border border-[#1e293b] text-gray-500 hover:text-white transition-colors cursor-pointer">24H</span>
            </div>
          </div>
          <div className="flex-1 w-full bg-[#070b14]">
            <NetworkTopologyMap />
          </div>
        </div>

        {/* Right Side Stack: Alerts + Timeline Graph */}
        <div className="lg:col-span-1 flex flex-col gap-6">
          
          {/* Alert Queue Panel */}
          <div className="bg-[#111827] border border-[#1e293b] rounded-xl flex flex-col h-[350px]">
            <div className="p-4 border-b border-[#1e293b] flex justify-between items-center">
              <h2 className="font-semibold text-sm text-gray-300 flex items-center gap-2">
                <ShieldAlert size={16} className="text-red-500" /> 
                Actionable Alerts
              </h2>
              <span className="text-xs bg-red-500/10 border border-red-500/30 text-red-500 px-2 py-0.5 rounded animate-pulse font-bold">3 LIVE</span>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
              {/* Fake Alert Items */}
              {[1, 2, 3].map((alert) => (
                <div key={alert} className="border border-red-500/20 bg-red-500/5 rounded-lg p-3 cursor-pointer hover:bg-red-500/10 transition-colors">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-[10px] font-bold px-2 py-0.5 rounded text-red-500 border border-red-500/30 uppercase tracking-wider backdrop-blur-sm bg-[#070b14]/50">Critical</span>
                    <span className="text-[10px] text-gray-500 font-mono">2m ago</span>
                  </div>
                  <h3 className="text-sm font-semibold mb-1 text-gray-200">Device SIM-{alert.toString().padStart(4, '0')} C2 Beaconing</h3>
                  <p className="text-xs text-gray-400 line-clamp-2 leading-relaxed">
                    Isolation Forest flagged anomalous structural change. External destination port anomaly detected (Out Of Baseline).
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Trust Score 12-Hour Timeline Graph using Recharts */}
          <div className="flex-1 w-full bg-[#111827] border border-[#1e293b] rounded-xl flex flex-col h-[280px] p-2 hover:border-[#334155] transition-colors overflow-hidden relative shadow-inner">
            <TrustScoreTimeline />
          </div>
          
        </div>

      </div>
    </div>
  );
}
