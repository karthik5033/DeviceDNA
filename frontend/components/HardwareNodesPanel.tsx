'use client';

import { useState, useEffect } from 'react';
import { Server } from 'lucide-react';
import { cn } from '@/lib/utils';
import { motion } from 'framer-motion';

type HardwareDevice = {
  device_id: string;
  device_class: string;
  source: string;
  last_seen: string;
  status: string;
};

// Helper for relative time
function getRelativeTime(timestamp: string) {
  try {
    // If timestamp doesn't have Z, JS might assume local, so append Z if missing and not already handled
    const cleanStamp = timestamp.endsWith('Z') || timestamp.includes('+') ? timestamp : timestamp + 'Z';
    const time = new Date(cleanStamp).getTime();
    const now = Date.now();
    const diff = Math.floor((now - time) / 1000);
    
    if (isNaN(diff)) return 'Unknown';
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return `${Math.floor(diff / 3600)}h ago`;
  } catch {
    return 'Unknown';
  }
}

export default function HardwareNodesPanel() {
  const [devices, setDevices] = useState<HardwareDevice[]>([]);

  useEffect(() => {
    const fetchDevices = () => {
      fetch('http://localhost:8000/api/hardware/devices')
        .then(res => res.json())
        .then(data => {
          if (Array.isArray(data)) {
            // Filter physical devices only
            setDevices(data.filter(d => d.source === 'physical'));
          }
        })
        .catch(err => console.error("Failed to fetch hardware devices", err));
    };

    fetchDevices();
    const interval = setInterval(fetchDevices, 5000);
    return () => clearInterval(interval);
  }, []);

  if (devices.length === 0) return null;

  return (
    <div className="bg-white/[0.03] backdrop-blur-2xl border border-white/[0.08] rounded-2xl p-6 relative overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.4)] ring-1 ring-white/5 w-full">
      <div className="flex justify-between items-center mb-6">
        <h2 className="font-semibold text-sm text-gray-200 tracking-widest uppercase flex items-center gap-2">
          <Server className="w-4 h-4 text-[#3edcff]" />
          Physical Hardware Nodes
        </h2>
        <div className="flex items-center gap-2">
           <span className="w-2 h-2 rounded-full bg-green-500 shadow-[0_0_8px_#22c55e] animate-pulse" />
           <span className="text-[10px] font-mono text-gray-400">SYNC ACTIVE</span>
        </div>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        {devices.map(device => {
          const isOnline = device.status === 'online';
          
          return (
            <motion.div 
              key={device.device_id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className={cn(
                "border rounded-xl p-4 flex flex-col gap-3 transition-all duration-500 relative overflow-hidden",
                isOnline 
                  ? "bg-white/[0.02] border-white/10 hover:border-white/20" 
                  : "bg-black/40 border-red-900/30 opacity-50"
              )}
            >
              {/* Optional glow for online nodes */}
              {isOnline && (
                <div className="absolute -bottom-4 -right-4 w-16 h-16 bg-[#3edcff]/10 blur-xl rounded-full" />
              )}
              
              <div className="flex justify-between items-start z-10">
                <span className="font-mono font-bold text-white tracking-tight">{device.device_id}</span>
                <span className={cn(
                  "text-[9px] font-bold tracking-widest uppercase px-2 py-0.5 rounded border shadow-sm",
                  isOnline 
                    ? "text-green-400 border-green-500/30 bg-green-950/30 shadow-[0_0_5px_rgba(34,197,94,0.3)]" 
                    : "text-red-400 border-red-500/30 bg-red-950/30"
                )}>
                  {isOnline ? 'ONLINE' : 'OFFLINE'}
                </span>
              </div>
              
              <div className="flex flex-col gap-1 z-10">
                <span className="text-[10px] text-gray-500 uppercase tracking-widest">Class</span>
                <span className="text-xs text-gray-300 font-mono">{device.device_class}</span>
              </div>
              
              <div className="flex flex-col gap-1 z-10 mt-2 border-t border-white/5 pt-2">
                <span className="text-[10px] text-gray-500 uppercase tracking-widest">Last Seen</span>
                <span className="text-xs text-gray-400 font-mono">{getRelativeTime(device.last_seen)}</span>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
