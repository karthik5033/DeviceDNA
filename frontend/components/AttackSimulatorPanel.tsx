'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert, Square, Shield, Clock, Activity, ChevronUp, ChevronDown, Radar } from 'lucide-react';

interface AttackSpec {
  id: number;
  name: string;
  description: string;
  targets: string[];
  default_duration_seconds: number;
}

interface AttackStatus {
  is_running: boolean;
  attack_id: number | null;
  started_at: number | null;
  active_targets: Record<string, any>;
}

const STAGE_LABELS: Record<string, string> = {
  beacon: 'C2 Beaconing Detected',
  ddos:   'DDoS Flood Detected',
  recon:  'Recon Scan Detected',
  lateral: 'Lateral Movement Detected',
  exfil:  'Data Exfil Detected',
};

const STAGE_COLORS: Record<string, string> = {
  beacon: '#f97316',
  ddos:   '#ef4444',
  recon:  '#eab308',
  lateral: '#a855f7',
  exfil:  '#3b82f6',
};

export default function AttackSimulatorPanel() {
  const [attacks, setAttacks] = useState<AttackSpec[]>([]);
  const [status, setStatus] = useState<AttackStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [activeStage, setActiveStage] = useState<string>('');
  const [isMinimized, setIsMinimized] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const apiUrl = typeof window !== 'undefined' ? `http://${window.location.hostname}:8000` : (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');

  const fetchAttacks = async () => {
    try {
      const res = await fetch(`${apiUrl}/api/attack/list`);
      if (res.ok) setAttacks(await res.json());
    } catch (_) {}
  };

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${apiUrl}/api/attack/status`);
      if (res.ok) {
        const data: AttackStatus = await res.json();
        setStatus(data);
        // Derive current stage from active target payloads
        const types = Object.values(data.active_targets).map((t: any) => t.type);
        setActiveStage(types[0] || '');
      }
    } catch (_) {}
  };

  useEffect(() => {
    fetchAttacks();
    fetchStatus();
    const interval = setInterval(fetchStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  // Elapsed timer
  useEffect(() => {
    if (status?.is_running && status.started_at) {
      timerRef.current = setInterval(() => {
        setElapsed(Math.floor(Date.now() / 1000 - status.started_at!));
      }, 1000);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
      setElapsed(0);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [status?.is_running, status?.started_at]);

  const triggerAttack = async (attackId: number) => {
    setLoading(true);
    try {
      const res = await fetch(`${apiUrl}/api/attack/trigger`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ attack_id: attackId }),
      });
      if (!res.ok) {
        const err = await res.json();
        alert(`Failed: ${err.detail}`);
      } else {
        await fetchStatus();
      }
    } catch (e) {
      alert('Could not reach backend. Is it running?');
    } finally {
      setLoading(false);
    }
  };

  const stopAttack = async () => {
    setLoading(true);
    try {
      await fetch(`${apiUrl}/api/attack/stop`, { method: 'POST' });
      await fetchStatus();
    } catch (_) {}
    finally { setLoading(false); }
  };

  const fmtElapsed = (s: number) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

  const stageColor = STAGE_COLORS[activeStage] || '#3edcff';
  const stageLabel = STAGE_LABELS[activeStage] || '';

  return (
    <div className="bg-white/[0.03] backdrop-blur-2xl border border-white/[0.08] rounded-2xl overflow-hidden shadow-[0_8px_32px_rgba(0,0,0,0.4)] ring-1 ring-white/5">
      {/* Header */}
      <div className="p-4 border-b border-white/5 bg-black/40 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Radar className="w-4 h-4 text-orange-400 drop-shadow-[0_0_8px_#f97316]" />
          <h2 className="font-semibold text-sm text-gray-200 tracking-widest uppercase">
            Threat Intelligence
          </h2>
        </div>
        {status?.is_running && (
          <div className="flex items-center gap-2 mr-2">
            <span className="w-2 h-2 rounded-full animate-ping" style={{ background: stageColor }} />
            <span className="text-xs font-mono font-bold" style={{ color: stageColor }}>
              {stageLabel}
            </span>
            <span className="text-xs font-mono text-gray-400 flex items-center gap-1">
              <Clock className="w-3 h-3" /> {fmtElapsed(elapsed)}
            </span>
          </div>
        )}
        <button 
          onClick={() => setIsMinimized(!isMinimized)}
          className="p-1 text-gray-500 hover:text-white hover:bg-white/10 rounded transition-colors"
        >
          {isMinimized ? <ChevronDown className="w-5 h-5" /> : <ChevronUp className="w-5 h-5" />}
        </button>
      </div>

      <AnimatePresence>
        {!isMinimized && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="overflow-hidden"
          >
            <div className="p-5">
              {/* Active threat banner */}
              <AnimatePresence>
                {status?.is_running && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="rounded-xl border p-3 flex flex-col gap-2 mb-4"
                    style={{ borderColor: `${stageColor}40`, background: `${stageColor}0d` }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <ShieldAlert className="w-4 h-4" style={{ color: stageColor }} />
                        <span className="font-mono font-bold text-sm" style={{ color: stageColor }}>
                          ⚠ THREAT {status.attack_id} DETECTED
                        </span>
                      </div>
                      <button
                        id="stop-attack-btn"
                        onClick={stopAttack}
                        disabled={loading}
                        className="flex items-center gap-2 px-3 py-1.5 bg-red-600/20 hover:bg-red-600/40 border border-red-500/50 rounded-lg text-red-400 text-xs font-mono font-bold tracking-widest transition-all active:scale-95 disabled:opacity-50"
                      >
                        <Square className="w-3 h-3" />
                        MITIGATE
                      </button>
                    </div>
                    {/* Compromised targets */}
                    <div className="flex flex-wrap gap-2 mt-1">
                      {Object.entries(status.active_targets).map(([device, payload]: [string, any]) => (
                        <span
                          key={device}
                          className="px-2 py-0.5 rounded font-mono text-[10px] font-bold"
                          style={{ background: `${STAGE_COLORS[payload.type] || stageColor}20`, color: STAGE_COLORS[payload.type] || stageColor, border: `1px solid ${STAGE_COLORS[payload.type] || stageColor}40` }}
                        >
                          {device} [{payload.type?.toUpperCase()}]
                        </span>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Threat pattern cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                {attacks.map((atk) => {
                  const isThis = status?.is_running && status.attack_id === atk.id;
                  const isOther = status?.is_running && status.attack_id !== atk.id;
                  let color = '#ef4444'; // Default red
                  if (atk.id === 1) color = '#eab308'; // Yellow
                  if (atk.id === 2) color = '#ef4444'; // Red
                  if (atk.id === 3) color = '#a855f7'; // Purple
                  if (atk.id === 4) color = '#3b82f6'; // Blue

                  return (
                    <motion.button
                      key={atk.id}
                      id={`trigger-attack-${atk.id}-btn`}
                      onClick={() => !isThis && !isOther && triggerAttack(atk.id)}
                      disabled={loading || status?.is_running === true}
                      whileHover={!status?.is_running ? { scale: 1.02 } : {}}
                      whileTap={!status?.is_running ? { scale: 0.98 } : {}}
                      className="flex flex-col gap-2 p-3 rounded-xl border text-left transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                      style={{
                        borderColor: isThis ? `${color}80` : `${color}20`,
                        background: isThis ? `${color}15` : 'rgba(255,255,255,0.02)',
                        boxShadow: isThis ? `0 0 20px ${color}20` : 'none',
                      }}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Activity className="w-4 h-4" style={{ color }} />
                          <span className="font-mono font-bold text-xs" style={{ color }}>
                            THREAT {atk.id}
                          </span>
                        </div>
                        {isThis ? (
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{ background: `${color}30`, color }}>ACTIVE</span>
                        ) : (
                          <span className="text-[10px] font-mono text-gray-500">{atk.default_duration_seconds}s</span>
                        )}
                      </div>
                      <span className="text-white text-sm font-semibold">{atk.name}</span>
                      <span className="text-gray-400 text-[11px] leading-relaxed">{atk.description}</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {atk.targets.slice(0, 4).map(t => (
                          <span key={t} className="px-1.5 py-0.5 bg-white/5 border border-white/10 rounded font-mono text-[9px] text-gray-400">{t}</span>
                        ))}
                        {atk.targets.length > 4 && (
                          <span className="px-1.5 py-0.5 bg-white/5 border border-white/10 rounded font-mono text-[9px] text-gray-400">+{atk.targets.length - 4} more</span>
                        )}
                      </div>
                    </motion.button>
                  );
                })}
              </div>

              {!status?.is_running && attacks.length === 0 && (
                <div className="text-center text-gray-600 text-xs font-mono py-4">
                  <Shield className="w-8 h-8 mx-auto mb-2 opacity-30" />
                  Threat engine offline — no connection to backend
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
