'use client';

import { useState, useEffect } from 'react';
import { ShieldAlert, Terminal, Eye, AlertTriangle, Zap, Flame, Fingerprint, RefreshCcw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { io } from 'socket.io-client';

const socket = io('http://localhost:8000');

export default function AlertsPage() {
  const [filter, setFilter] = useState('all');
  const [liveAlerts, setLiveAlerts] = useState<any[]>([]);
  const [selectedAlert, setSelectedAlert] = useState<any | null>(null);
  const [isResolving, setIsResolving] = useState(false);

  useEffect(() => {
    // Fetch initial alerts
    fetch('http://localhost:8000/api/alerts')
      .then(res => res.json())
      .then(data => setLiveAlerts(data))
      .catch(err => console.error('Failed to fetch alerts:', err));

    socket.on('new_alert', (alert) => {
      setLiveAlerts((prev) => [alert, ...prev]);
    });

    return () => {
      socket.off('new_alert');
    };
  }, []);

  const handleResolve = async () => {
    if (!selectedAlert) return;
    setIsResolving(true);
    try {
      const res = await fetch(`http://localhost:8000/api/alerts/${selectedAlert.id}/resolve`, {
        method: 'POST'
      });
      if (res.ok) {
        setLiveAlerts(prev => prev.filter(a => a.id !== selectedAlert.id));
        setSelectedAlert(null);
      }
    } catch (err) {
      console.error('Failed to resolve alert:', err);
    } finally {
      setIsResolving(false);
    }
  };

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
          {liveAlerts.filter(a => filter === 'all' || a.severity === filter).map((alert, idx) => (
            <div 
               key={alert.id || idx} 
               onClick={() => setSelectedAlert(alert)}
               className={cn(
                 "bg-[#111827] border rounded-xl p-4 transition-colors relative group overflow-hidden pl-5 cursor-pointer",
                 selectedAlert?.id === alert.id ? "border-[#3edcff] shadow-[0_0_15px_rgba(62,220,255,0.1)]" : "border-[#1e293b] hover:border-[#334155]"
               )}
            >
              
              {/* Severity Side Bar */}
              <div className={cn("absolute left-0 top-0 w-1.5 h-full", 
                alert.severity === 'critical' ? 'bg-red-500 shadow-[0_0_15px_#ef4444]' :
                alert.severity === 'high' ? 'bg-orange-500 shadow-[0_0_15px_#f97316]' :
                alert.severity === 'medium' ? 'bg-yellow-500' : 'bg-gray-500'
              )}></div>

              <div className="flex justify-between items-start mb-2">
                <div className="flex gap-3 items-center">
                  <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-[#1e293b] text-gray-300">{alert.id.split('-')[0]}..</span>
                  <span className="font-mono text-[#3edcff] font-bold tracking-tight text-sm flex items-center gap-1">
                    <Terminal size={14} /> {alert.device_id || alert.device}
                  </span>
                </div>
                <div className="flex gap-4 items-center">
                  <span className="text-xs font-mono text-gray-500">{new Date(alert.timestamp || alert.time).toLocaleTimeString()}</span>
                  <div className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-[#070b14] border border-[#1e293b]">
                    <span className={(alert.trust_score || alert.score) < 40 ? 'text-red-500 font-bold' : (alert.trust_score || alert.score) < 60 ? 'text-orange-500' : 'text-yellow-500'}>
                      {(alert.trust_score || alert.score).toFixed(1)}
                    </span>
                    <span className="text-gray-600">Trust</span>
                  </div>
                </div>
              </div>

              <h2 className="text-lg font-bold text-gray-200 mb-1 flex items-center gap-2 tracking-tight">
                {alert.severity === 'critical' ? <Flame size={18} className="text-red-500 animate-pulse" /> : <AlertTriangle size={18} className="text-orange-500" />}
                {alert.alert_type || alert.type}
              </h2>
              
              <p className="text-sm text-gray-400 mb-4">{alert.message}</p>
              
              <div className="flex justify-between items-center text-xs">
                 <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-[#1e293b]/50 border border-[#334155] text-gray-400 font-medium">
                    <Fingerprint size={12} className="text-[#3edcff]" />
                    Scores: <span className="text-white">VAE: {alert.vae_score?.toFixed(2)} | IF: {alert.if_score?.toFixed(2)} | LSTM: {alert.lstm_score?.toFixed(2)} | GNN: {alert.gnn_score?.toFixed(2)}</span>
                 </div>
              </div>
            </div>
          ))}
          {liveAlerts.length === 0 && (
             <div className="text-gray-500 p-8 border border-dashed border-[#1e293b] rounded-xl text-center flex flex-col items-center justify-center font-mono">
               <ShieldAlert className="mb-4 opacity-50" size={32} />
               No Active Alerts Found
             </div>
          )}
        </div>

        {/* Explainable AI Action Panel */}
        <div className="hidden xl:flex flex-col gap-4">
          <div className="bg-[#111827] border border-[#1e293b] rounded-xl p-6 shadow-xl sticky top-24">
            <h2 className="text-lg font-bold text-gray-200 flex items-center gap-2 mb-4 border-b border-[#1e293b] pb-2">
              <Eye className="text-[#3edcff]" /> Threat Context {selectedAlert && `(${selectedAlert.id})`}
            </h2>
            
            <p className="text-sm text-gray-400 mb-6 italic leading-relaxed">
              {selectedAlert ? 
                 `SHAP has analyzed ${selectedAlert.device} behavior against baseline parameters. High likelihood of mathematical drift associated with ${selectedAlert.type}. Action recommended.`
               : 
                 `Select an alert from the queue to run the SHAP Explainable AI explainer algorithm. SHAP will deconstruct the neural network's decision boundary.`
              }
            </p>
            
            <div className="border border-dashed border-[#1e293b] bg-[#070b14]/50 rounded-lg h-48 flex flex-col items-center justify-center text-gray-600 space-y-3 p-4 text-center">
              {selectedAlert ? (
                 <div className="text-left w-full h-full text-xs font-mono flex flex-col justify-between">
                    <div><span className="text-red-400">Bytes Out:</span> 9.2MB <span className="text-gray-500">(+340% Baseline)</span></div>
                    <div><span className="text-orange-400">Dst Port:</span> 443 <span className="text-gray-500">(Unseen internally)</span></div>
                    <div><span className="text-yellow-400">Conn. Duration:</span> 4h 12m <span className="text-gray-500">(Persistent)</span></div>
                    <div className="text-[#3edcff] font-bold mt-4 pt-2 border-t border-[#1e293b]">SHAP Confidence: 94.2%</div>
                 </div>
              ) : (
                 <>
                   <RefreshCcw size={24} className="animate-spin duration-[3000ms]" />
                   <span className="text-xs font-mono">Awaiting Alert Selection...</span>
                 </>
              )}
            </div>
            
            <button 
               disabled={!selectedAlert || isResolving}
               onClick={handleResolve}
               className={cn(
                 "w-full mt-6 py-2 border rounded-lg text-sm font-bold transition-all flex justify-center items-center gap-2",
                 selectedAlert 
                   ? "bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-500 border-emerald-500/50 hover:shadow-[0_0_15px_rgba(16,185,129,0.3)] shadow-inner" 
                   : "bg-[#070b14] text-gray-600 border-[#1e293b] cursor-not-allowed",
                 isResolving && "opacity-50"
               )}
            >
              {isResolving ? (
                <span className="flex items-center gap-2"><div className="w-4 h-4 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin"></div> Resolving Alert...</span>
              ) : (
                <><Zap size={16} /> Resolve Alert</>
              )}
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
