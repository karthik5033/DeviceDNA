import React, { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Terminal, ShieldAlert, Activity, ShieldX } from 'lucide-react';

const mockLogs = [
  "Initializing intercept payload...",
  "Running nmap -sS -p 1-65535 10.0.0.1/24",
  "Port 22/tcp open (ssh)",
  "Port 80/tcp open (http)",
  "Attempting SSH brute force attack (admin/admin)...",
  "Connection refused by sandbox proxy.",
  "Attempting CVE-2021-44228 exploit on port 8080...",
  "Payload blocked by deep packet inspection.",
  "Injecting reverse shell script...",
  "Execution sandboxed: EPERM",
  "Data exfiltration attempt detected (1.4MB to 185.12.x.x)",
  "Dropping egress packets.",
  "Scanning internal subnets...",
  "ARP spoofing attempt detected and neutralized."
];

export default function SandboxInterceptCard({ deviceId }: { deviceId: string }) {
  const [logs, setLogs] = useState<string[]>([]);
  const [packetsBlocked, setPacketsBlocked] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let index = 0;
    
    // Initial burst
    setLogs(["[SYSTEM] Device isolated into VLAN 99 (Sandbox)", "[SYSTEM] Deep packet inspection enabled"]);
    
    const interval = setInterval(() => {
      setLogs(prev => [...prev, `[${new Date().toISOString().split('T')[1].slice(0, 8)}] ${mockLogs[index % mockLogs.length]}`]);
      setPacketsBlocked(prev => prev + Math.floor(Math.random() * 45) + 10);
      index++;
    }, 1200);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="flex flex-col h-full w-full bg-black/40 border-l border-yellow-500/20 p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-yellow-500" />
          <h3 className="text-xs font-bold text-yellow-500 tracking-widest uppercase">Live Intercept</h3>
        </div>
        <div className="flex gap-4">
          <div className="flex flex-col items-end">
             <span className="text-[9px] text-gray-500 uppercase tracking-widest">Egress Dropped</span>
             <span className="text-xs font-mono font-bold text-red-400">{packetsBlocked.toLocaleString()} pkt</span>
          </div>
          <div className="flex flex-col items-end">
             <span className="text-[9px] text-gray-500 uppercase tracking-widest">Status</span>
             <span className="text-xs font-mono font-bold text-yellow-500 animate-pulse">CONTAINED</span>
          </div>
        </div>
      </div>

      <div 
        ref={scrollRef}
        className="flex-1 bg-black/60 rounded border border-white/5 p-3 overflow-y-auto font-mono text-[10px] sm:text-xs text-green-400/80 shadow-inner custom-scrollbar"
      >
        <AnimatePresence initial={false}>
          {logs.map((log, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="mb-1 break-all"
            >
              {log.includes('blocked') || log.includes('dropped') || log.includes('neutralized') || log.includes('refused') ? (
                <span className="text-red-400">{log}</span>
              ) : log.includes('[SYSTEM]') ? (
                <span className="text-yellow-500">{log}</span>
              ) : (
                <span>{log}</span>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
        <div className="mt-2 flex items-center gap-2 text-yellow-500/50 animate-pulse">
          <Terminal className="w-3 h-3" />
          <span>Capturing payloads...</span>
        </div>
      </div>
    </div>
  );
}
