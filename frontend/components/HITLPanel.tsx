import React, { useState, useEffect } from 'react';
import { ShieldAlert, Check, X, Timer, Activity, Database, Globe } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { io, Socket } from 'socket.io-client';

interface PendingAction {
  device_id: string;
  target_tier: number;
  action: string;
  trigger_score: number;
  timestamp: string;
  expires_at: number;
  shap_evidence: {
    device_class: string;
    features: {
      total_flows: number;
      total_bytes: number;
      avg_packet_size: number;
      external_ratio: number;
      anomaly_ensemble: number;
      vae_score: number;
      gnn_score: number;
    };
  };
}

export default function HITLPanel() {
  const [pendingList, setPendingList] = useState<PendingAction[]>([]);
  const [timers, setTimers] = useState<Record<string, number>>({});

  // 1. Fetch pending lists from REST API
  const fetchPending = async () => {
    try {
      const res = await fetch('http://localhost:8000/api/response/pending');
      if (res.ok) {
        const data = await res.json();
        setPendingList(data);
      }
    } catch (err) {
      console.error('Failed to fetch pending HITL responses', err);
    }
  };

  useEffect(() => {
    fetchPending();
    
    // Poll every 3 seconds to keep sync
    const interval = setInterval(fetchPending, 3000);

    // 2. Connect WebSockets for real-time event updates
    const socket: Socket = io('http://localhost:8000', {
      transports: ['polling', 'websocket'],
    });

    socket.on('hitl_pending', (newAction: PendingAction) => {
      setPendingList((prev) => {
        // Avoid duplicates
        if (prev.some((a) => a.device_id === newAction.device_id)) return prev;
        return [...prev, newAction];
      });
    });

    socket.on('isolate_device', (data) => {
      setPendingList((prev) => prev.filter((a) => a.device_id !== data.device_id));
    });

    socket.on('honeypot_device', (data) => {
      setPendingList((prev) => prev.filter((a) => a.device_id !== data.device_id));
    });

    return () => {
      clearInterval(interval);
      socket.disconnect();
    };
  }, []);

  // 3. Countdown timer logic
  useEffect(() => {
    const timerInterval = setInterval(() => {
      const newTimers: Record<string, number> = {};
      const nowSecs = Date.now() / 1000;
      
      pendingList.forEach((item) => {
        const remaining = Math.max(0, Math.round(item.expires_at - nowSecs));
        newTimers[item.device_id] = remaining;
      });
      
      setTimers(newTimers);
    }, 1000);

    return () => clearInterval(timerInterval);
  }, [pendingList]);

  // 4. Action Handlers
  const handleApprove = async (deviceId: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/response/approve/${deviceId}`, {
        method: 'POST',
      });
      if (res.ok) {
        setPendingList((prev) => prev.filter((item) => item.device_id !== deviceId));
      }
    } catch (err) {
      console.error('Approval request failed', err);
    }
  };

  const handleDeny = async (deviceId: string) => {
    try {
      const res = await fetch(`http://localhost:8000/api/response/deny/${deviceId}`, {
        method: 'POST',
      });
      if (res.ok) {
        setPendingList((prev) => prev.filter((item) => item.device_id !== deviceId));
      }
    } catch (err) {
      console.error('Denial/Override request failed', err);
    }
  };

  if (pendingList.length === 0) return null;

  return (
    <div className="fixed bottom-6 right-6 z-[9999] flex flex-col gap-4 max-w-md w-full">
      <AnimatePresence>
        {pendingList.map((item) => {
          const timeLeft = timers[item.device_id] ?? 120;
          const progressPct = (timeLeft / 120) * 100;
          const evidence = item.shap_evidence?.features || {};

          return (
            <motion.div
              key={item.device_id}
              initial={{ opacity: 0, y: 50, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 20, scale: 0.95 }}
              className="bg-[#050b18]/90 backdrop-blur-xl border border-red-500/40 rounded-2xl overflow-hidden shadow-[0_0_50px_rgba(239,68,68,0.25)] ring-1 ring-white/10"
            >
              {/* Progress bar countdown */}
              <div className="w-full h-1 bg-white/5">
                <motion.div
                  className="h-full bg-red-500 shadow-[0_0_8px_#ef4444]"
                  initial={{ width: '100%' }}
                  animate={{ width: `${progressPct}%` }}
                  transition={{ ease: 'linear', duration: 1 }}
                />
              </div>

              <div className="p-5 flex flex-col gap-4">
                {/* Header */}
                <div className="flex justify-between items-start">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-red-500/10 rounded-lg border border-red-500/30">
                      <ShieldAlert className="w-5 h-5 text-red-500 animate-pulse" />
                    </div>
                    <div>
                      <h3 className="text-xs font-bold text-red-400 tracking-wider uppercase font-mono">
                        HITL Approval Required (Tier {item.target_tier})
                      </h3>
                      <p className="text-lg font-mono font-bold text-white leading-tight">
                        {item.device_id}
                      </p>
                    </div>
                  </div>
                  
                  {/* Timer display */}
                  <div className="flex items-center gap-1.5 px-2.5 py-1 bg-white/5 border border-white/10 rounded-lg font-mono text-sm text-red-400 font-semibold shadow-inner">
                    <Timer className="w-4 h-4" />
                    <span>{timeLeft}s</span>
                  </div>
                </div>

                {/* Main Stats / Trigger Brief */}
                <div className="bg-white/[0.02] border border-white/5 rounded-xl p-3 flex flex-col gap-2">
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-gray-400">Trigger Action:</span>
                    <span className="text-red-400 font-bold uppercase tracking-widest">{item.action}</span>
                  </div>
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-gray-400">Effective Trust:</span>
                    <span className="text-white font-bold">{item.trigger_score.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between text-xs font-mono">
                    <span className="text-gray-400">Device Class:</span>
                    <span className="text-gray-300 capitalize">{item.shap_evidence?.device_class || 'Unknown'}</span>
                  </div>
                </div>

                {/* Threat Intelligence / SHAP Evidence Brief */}
                <div className="flex flex-col gap-2">
                  <span className="text-[10px] font-bold text-gray-500 tracking-widest uppercase font-mono">
                    Top Anomaly Evidence (SHAP metrics)
                  </span>
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="bg-white/[0.01] border border-white/5 p-2 rounded-lg flex flex-col">
                      <span className="text-[10px] text-gray-500 flex items-center gap-1">
                        <Activity className="w-3 h-3 text-[#3edcff]" /> Flows Count
                      </span>
                      <span className="text-sm font-bold text-white">
                        {evidence.total_flows ? Math.round(evidence.total_flows) : 'Normal'}
                      </span>
                    </div>
                    <div className="bg-white/[0.01] border border-white/5 p-2 rounded-lg flex flex-col">
                      <span className="text-[10px] text-gray-500 flex items-center gap-1">
                        <Database className="w-3 h-3 text-indigo-400" /> Bytes Sent
                      </span>
                      <span className="text-sm font-bold text-white">
                        {evidence.total_bytes ? `${(evidence.total_bytes / 1024).toFixed(1)} KB` : 'Normal'}
                      </span>
                    </div>
                    <div className="bg-white/[0.01] border border-white/5 p-2 rounded-lg flex flex-col">
                      <span className="text-[10px] text-gray-500 flex items-center gap-1">
                        <Globe className="w-3 h-3 text-orange-400" /> Ext. Ratio
                      </span>
                      <span className="text-sm font-bold text-white">
                        {evidence.external_ratio ? `${(evidence.external_ratio * 100).toFixed(0)}%` : '0%'}
                      </span>
                    </div>
                    <div className="bg-white/[0.01] border border-white/5 p-2 rounded-lg flex flex-col">
                      <span className="text-[10px] text-gray-500 flex items-center gap-1">
                        <ShieldAlert className="w-3 h-3 text-red-400" /> GNN Score
                      </span>
                      <span className="text-sm font-bold text-white">
                        {evidence.gnn_score ? evidence.gnn_score.toFixed(3) : '0.000'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Bottom Action buttons */}
                <div className="grid grid-cols-2 gap-3 mt-1">
                  <button
                    onClick={() => handleDeny(item.device_id)}
                    className="flex items-center justify-center gap-2 py-2 px-4 border border-white/10 hover:border-white/20 bg-white/5 hover:bg-white/10 text-gray-300 hover:text-white rounded-xl font-mono text-sm font-semibold transition-all shadow-inner active:scale-[0.98]"
                  >
                    <X className="w-4 h-4" />
                    <span>Deny / Ignore</span>
                  </button>
                  <button
                    onClick={() => handleApprove(item.device_id)}
                    className="flex items-center justify-center gap-2 py-2 px-4 border border-red-500/40 hover:border-red-500/60 bg-red-600/80 hover:bg-red-600 text-white rounded-xl font-mono text-sm font-semibold transition-all shadow-[0_0_15px_rgba(239,68,68,0.2)] hover:shadow-[0_0_20px_rgba(239,68,68,0.4)] active:scale-[0.98]"
                  >
                    <Check className="w-4 h-4" />
                    <span>Approve Actuation</span>
                  </button>
                </div>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
