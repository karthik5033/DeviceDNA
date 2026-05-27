'use client';

'use client';

import NetworkTopologyMap from '@/components/visualizations/NetworkTopologyMap';
import { Network, Maximize2, Settings2, Filter, Download } from 'lucide-react';
import { useState, useEffect, useCallback } from 'react';
import { io } from 'socket.io-client';
import { ChevronDown, Check } from 'lucide-react';

export default function TopologyPage() {
  const [trustScores, setTrustScores] = useState<Record<string, number>>({});
  
  // Filter state
  const [showFilterMenu, setShowFilterMenu] = useState(false);
  const [activeFilter, setActiveFilter] = useState<'all' | 'anomalous' | 'healthy'>('all');
  
  // Physics state
  const [showPhysicsMenu, setShowPhysicsMenu] = useState(false);
  const [physicsMode, setPhysicsMode] = useState<'dynamic' | 'frozen'>('dynamic');
  
  // Export state
  const [isExporting, setIsExporting] = useState(false);

  useEffect(() => {
    // Fetch initial devices from Redis
    fetch('http://localhost:8000/api/trust/devices')
      .then(res => res.json())
      .then(data => setTrustScores(data))
      .catch(err => console.error("Failed to fetch initial devices", err));

    const socket = io('http://localhost:8000', {
      transports: ['polling', 'websocket'],
    });

    socket.on('trust_update', (data) => {
      setTrustScores(prev => ({ ...prev, [data.device_id]: data.score }));
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  const handleIsolate = useCallback((nodeId: string) => {
    console.log('Topology: Isolating node', nodeId);
  }, []);

  const handleNodeClick = useCallback((nodeId: string, score: number) => {
    console.log('Topology: Node clicked', nodeId, score);
  }, []);

  const handleExportPcap = () => {
    setIsExporting(true);
    setTimeout(() => {
      setIsExporting(false);
      // Simulate file download
      const element = document.createElement("a");
      const file = new Blob(["dummy pcap data for demonstration"], {type: 'application/vnd.tcpdump.pcap'});
      element.href = URL.createObjectURL(file);
      element.download = `capture_snapshot_${new Date().getTime()}.pcap`;
      document.body.appendChild(element);
      element.click();
      document.body.removeChild(element);
    }, 1500);
  };

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
        <div className="flex gap-2 relative">
          
          {/* Filter Dropdown */}
          <div className="relative">
            <button 
              onClick={() => setShowFilterMenu(!showFilterMenu)}
              className="flex items-center gap-2 px-3 py-1.5 bg-[#111827] border border-[#1e293b] rounded-md hover:border-[#334155] transition-colors text-sm text-gray-400 hover:text-white"
            >
              <Filter size={16} /> 
              {activeFilter === 'all' ? 'Filter Nodes' : activeFilter === 'anomalous' ? 'Showing Anomalous' : 'Showing Healthy'}
              <ChevronDown size={14} className="ml-1 opacity-50" />
            </button>
            {showFilterMenu && (
              <div className="absolute top-full right-0 mt-1 w-48 bg-[#070b14] border border-[#1e293b] rounded-md shadow-2xl z-50 overflow-hidden text-sm text-gray-300">
                <button onClick={() => { setActiveFilter('all'); setShowFilterMenu(false); }} className="w-full text-left px-4 py-2 hover:bg-white/5 flex justify-between items-center">
                  Show All Nodes {activeFilter === 'all' && <Check size={14} className="text-[#3edcff]" />}
                </button>
                <button onClick={() => { setActiveFilter('anomalous'); setShowFilterMenu(false); }} className="w-full text-left px-4 py-2 hover:bg-white/5 flex justify-between items-center text-orange-400">
                  Only Anomalous {activeFilter === 'anomalous' && <Check size={14} className="text-[#3edcff]" />}
                </button>
                <button onClick={() => { setActiveFilter('healthy'); setShowFilterMenu(false); }} className="w-full text-left px-4 py-2 hover:bg-white/5 flex justify-between items-center text-green-400">
                  Only Healthy {activeFilter === 'healthy' && <Check size={14} className="text-[#3edcff]" />}
                </button>
              </div>
            )}
          </div>

          {/* Physics Dropdown */}
          <div className="relative">
            <button 
              onClick={() => setShowPhysicsMenu(!showPhysicsMenu)}
              className="flex items-center gap-2 px-3 py-1.5 bg-[#111827] border border-[#1e293b] rounded-md hover:border-[#334155] transition-colors text-sm text-gray-400 hover:text-white"
            >
              <Settings2 size={16} /> 
              {physicsMode === 'dynamic' ? 'Physics: Dynamic' : 'Physics: Frozen'}
              <ChevronDown size={14} className="ml-1 opacity-50" />
            </button>
            {showPhysicsMenu && (
              <div className="absolute top-full right-0 mt-1 w-48 bg-[#070b14] border border-[#1e293b] rounded-md shadow-2xl z-50 overflow-hidden text-sm text-gray-300">
                <button onClick={() => { setPhysicsMode('dynamic'); setShowPhysicsMenu(false); }} className="w-full text-left px-4 py-2 hover:bg-white/5 flex justify-between items-center">
                  Dynamic (Force) {physicsMode === 'dynamic' && <Check size={14} className="text-[#3edcff]" />}
                </button>
                <button onClick={() => { setPhysicsMode('frozen'); setShowPhysicsMenu(false); }} className="w-full text-left px-4 py-2 hover:bg-white/5 flex justify-between items-center text-gray-400">
                  Freeze Layout {physicsMode === 'frozen' && <Check size={14} className="text-[#3edcff]" />}
                </button>
              </div>
            )}
          </div>

          <button 
            onClick={handleExportPcap}
            disabled={isExporting}
            className={`flex items-center gap-2 px-3 py-1.5 bg-[#111827] border rounded-md transition-colors text-sm ${isExporting ? 'text-gray-500 border-gray-700 opacity-50' : 'text-[#3edcff] border-[#3edcff]/30 hover:bg-[#3edcff]/10'}`}
          >
            <Download size={16} className={isExporting ? 'animate-bounce' : ''} /> 
            {isExporting ? 'Exporting...' : 'Export PCAP'}
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
        <div className="absolute inset-0 bg-[#070b14] radial-gradient-topology pointer-events-auto" onClick={() => { setShowFilterMenu(false); setShowPhysicsMenu(false); }}>
          <NetworkTopologyMap 
            onIsolate={handleIsolate} 
            onNodeClick={handleNodeClick} 
            liveScores={trustScores} 
            filterMode={activeFilter}
            physicsMode={physicsMode}
          />
        </div>
      </div>
    </div>
  );
}
