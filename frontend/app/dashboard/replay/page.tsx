'use client';

import { PlaySquare, FastForward, Play, Pause, Activity, RotateCcw, Crosshair } from 'lucide-react';
import { useState } from 'react';
import TrustScoreTimeline from '@/components/visualizations/TrustScoreTimeline';

export default function ReplayPage() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);

  const togglePlay = () => {
    setIsPlaying(!isPlaying);
  };

  const reset = () => {
    setIsPlaying(false);
    setProgress(0);
  };

  return (
    <div className="flex flex-col gap-6 max-w-[1600px] mx-auto fade-in">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4 border-b border-[#1e293b] pb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tighter mb-1 font-sans flex items-center gap-3">
            <PlaySquare className="text-purple-500" size={28} />
            Historical Attack Replay
          </h1>
          <p className="text-gray-400 text-sm">Forensic timeline scrub tool re-evaluating historical incident telemetry vectors.</p>
        </div>
        
        <div className="bg-[#111827] border border-[#1e293b] rounded-lg px-4 py-2 flex items-center gap-3 text-sm font-mono text-gray-500 shadow-md">
          <Crosshair size={16} className="text-[#3edcff]" />
          Incident: <span className="text-white font-bold">ALT-9399 (Lateral Movement)</span>
          <div className="mx-2 w-px h-4 bg-[#1e293b]"></div>
          Time: <span className="text-white">Mar 12, 04:22 UTC</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Main Playback Canvas & Transport Controls */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <div className="bg-[#111827] border border-[#1e293b] rounded-xl flex flex-col min-h-[400px] p-6 relative overflow-hidden group shadow-2xl">
            {/* The Graph Re-use */}
            <h3 className="font-semibold text-sm text-gray-300 mb-4 flex justify-between items-center">
               <span>Reconstructed Trust Trajectory</span>
               <span className="text-[#3edcff] text-xs px-2 py-0.5 border border-[#3edcff]/30 bg-[#3edcff]/10 rounded items-center flex gap-1">
                 <Activity size={12} /> Syncing
               </span>
            </h3>
            <div className="flex-1 w-full relative">
              <TrustScoreTimeline />
              
              {/* Fake Scrub Bar overlaying the graph */}
              <div 
                 className="absolute top-0 bottom-0 w-px bg-white/50 pointer-events-none transition-all duration-300 shadow-[0_0_10px_#fff]" 
                 style={{ left: `${30 + progress}%` }} 
              />
            </div>
          </div>

          {/* Transport Controls Bottom Bar */}
          <div className="bg-[#111827] border border-[#1e293b] rounded-xl p-4 flex flex-col gap-4">
             {/* Progress Bar */}
             <div className="flex items-center gap-4 text-xs font-mono text-gray-500 cursor-pointer group">
               <span>-01:30</span>
               <div className="flex-1 h-2 bg-[#070b14] rounded-full border border-[#1e293b] overflow-hidden relative">
                  <div 
                    className="h-full bg-gradient-to-r from-[#3edcff] to-purple-500 transition-all duration-300 relative group-hover:from-blue-400 group-hover:to-purple-400" 
                    style={{ width: `${progress}%` }} 
                  >
                     <div className="absolute right-0 top-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full shadow-lg scale-0 group-hover:scale-100 transition-transform"></div>
                  </div>
               </div>
               <span>+00:45</span>
             </div>
             
             {/* Buttons */}
             <div className="flex justify-between items-center px-4">
               <div className="text-xs text-gray-500 font-mono">Status: {isPlaying ? 'Replaying Incident...' : 'Paused'}</div>
               
               <div className="flex items-center gap-4">
                 <button onClick={reset} className="text-gray-500 hover:text-white transition-colors"><RotateCcw size={18} /></button>
                 <button className="text-gray-500 hover:text-white transition-colors"><FastForward size={20} className="rotate-180" /></button>
                 
                 <button 
                   onClick={togglePlay} 
                   className="w-12 h-12 bg-[#3edcff] text-[#070b14] rounded-full flex items-center justify-center hover:bg-blue-400 hover:scale-105 transition-all shadow-[0_0_15px_rgba(62,220,255,0.4)]"
                 >
                   {isPlaying ? <Pause size={24} /> : <Play size={24} className="ml-1" />}
                 </button>
                 
                 <button className="text-gray-500 hover:text-white transition-colors"><FastForward size={20} /></button>
               </div>
               
               <div className="text-xs text-purple-400 border border-purple-500/30 bg-purple-500/10 px-2 py-1 rounded">1.0x Speed</div>
             </div>
          </div>
        </div>

        {/* Forensic Snapshot Panel */}
        <div className="lg:col-span-1 border-l border-[#1e293b] pl-6 flex flex-col gap-4">
          <div className="text-sm font-bold border-b border-[#1e293b] pb-2 text-gray-300">Raw Flow Replay Capture</div>
          
          <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col gap-2">
            {[...Array(8)].map((_, i) => (
              <div key={i} className={`p-3 rounded-md text-xs font-mono border ${i === 3 ? 'bg-red-500/10 border-red-500/50 text-red-400' : 'bg-[#111827] border-[#1e293b] text-gray-400'}`}>
                <div className="flex justify-between mb-1">
                  <span>10.0.1.44:50932</span>
                  <span>-></span>
                  <span className={i===3 ? 'text-red-500 font-bold' : ''}>185.15.x.x:443</span>
                </div>
                <div className="flex justify-between text-gray-600">
                  <span>TCP</span>
                  <span>{1024 + Math.random() * 5000 | 0} B</span>
                  <span>{i===3 ? 'ANOMALY_SIG_T' : 'CLEAN'}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
