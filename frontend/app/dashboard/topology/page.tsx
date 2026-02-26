'use client';

import NetworkTopologyMap from '@/components/visualizations/NetworkTopologyMap';
import { Network, Maximize2, Settings2, Filter, Download } from 'lucide-react';

export default function TopologyPage() {
  return (
    <div className="flex flex-col h-[calc(100vh-80px)] fade-in">
      {/* Interactive Header Toolbar */}
      <div className="flex justify-between items-center mb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tighter mb-1 font-sans flex items-center gap-3">
            <Network className="text-[#3edcff]" size={28} />
            Global Network Topology
          </h1>
          <p className="text-gray-400 text-sm">Force-directed map of exactly 50 node associations predicting Lateral Movement (GNN).</p>
        </div>
        
        {/* Graph Controls */}
        <div className="flex gap-2">
          <button className="flex items-center gap-2 px-3 py-1.5 bg-[#111827] border border-[#1e293b] rounded-md hover:border-[#334155] transition-colors text-sm text-gray-400 hover:text-white">
            <Filter size={16} /> Filter Nodes
          </button>
          <button className="flex items-center gap-2 px-3 py-1.5 bg-[#111827] border border-[#1e293b] rounded-md hover:border-[#334155] transition-colors text-sm text-gray-400 hover:text-white">
            <Settings2 size={16} /> Physics Layout
          </button>
          <button className="flex items-center gap-2 px-3 py-1.5 bg-[#111827] border border-[#1e293b] rounded-md hover:border-[#334155] transition-colors text-sm text-[#3edcff] border-[#3edcff]/30 hover:bg-[#3edcff]/10">
            <Download size={16} /> Export PCAP
          </button>
        </div>
      </div>

      {/* Full Canvas Graph Area */}
      <div className="flex-1 bg-[#111827] border border-[#1e293b] rounded-xl relative overflow-hidden shadow-2xl group">
        
        {/* Expand Graphic Icon */}
        <div className="absolute top-4 right-4 z-20 opacity-0 group-hover:opacity-100 transition-opacity">
          <button className="p-2 bg-[#070b14]/80 border border-[#1e293b] rounded text-gray-400 hover:text-white backdrop-blur">
            <Maximize2 size={18} />
          </button>
        </div>

        {/* Legend Overlay Floating */}
        <div className="absolute bottom-6 right-6 z-20 bg-[#070b14]/90 backdrop-blur border border-[#1e293b] rounded-lg p-3 text-xs text-gray-400 pointer-events-none shadow-xl">
          <div className="font-semibold text-gray-300 mb-2 border-b border-[#1e293b] pb-1">Node Trust State (GNN)</div>
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-green-500 shadow-[0_0_5px_#22c55e]"></div> 100-80: Healthy baseline</div>
            <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-yellow-500 shadow-[0_0_5px_#eab308]"></div> 79-60: Guarded (Minor Drift)</div>
            <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-full bg-orange-500 shadow-[0_0_5px_#f97316]"></div> 59-40: Suspicious Flow</div>
            <div className="flex items-center gap-2"><div className="w-4 h-4 rounded-full bg-red-500 shadow-[0_0_8px_#ef4444] animate-pulse"></div> &lt;40: Critical C2/Lateral Activity</div>
          </div>
        </div>

        {/* The D3 Instance Engine Core */}
        <div className="absolute inset-0 bg-[#070b14] radial-gradient-topology pointer-events-auto">
          <NetworkTopologyMap />
        </div>
      </div>
    </div>
  );
}
