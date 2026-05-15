'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { ShieldCheck, ArrowRight, Activity, Terminal, Lock, Network, Code, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';

// Premium Video Background
const VideoBackground = () => {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10 bg-black">
      <video
        autoPlay
        loop
        muted
        playsInline
        className="absolute inset-0 w-full h-full object-cover opacity-60"
        src="/bg-video.mp4"
      />
      {/* Refined Overlay so video remains highly visible but text is readable */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/40 via-black/20 to-[#050505] opacity-90" />
    </div>
  );
};

const features = [
  { title: "Behavioral CUSUM Drift", desc: "Monitors statistical accumulation to detect slow-acting threats like persistent data exfiltration over days instead of seconds.", icon: Activity },
  { title: "GraphSAGE Lateral Maps", desc: "Constructs live topological maps using D3 and identifies malicious traversal between isolated hardware nodes.", icon: Network },
  { title: "Isolation Forests", desc: "Immediately detects structural anomalies in payload sizes and port destinations mathematically independent of known signatures.", icon: Lock },
  { title: "NLP Rule Compilers", desc: "Translates human-readable English policies into rigid firewall enforcement logic via BERT embeddings instantly.", icon: Terminal }
];

export default function Home() {
  const [apiStatus, setApiStatus] = useState<'loading' | 'online' | 'offline'>('loading');

  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'ok') setApiStatus('online');
        else setApiStatus('offline');
      })
      .catch(() => setApiStatus('offline'));
  }, []);

  return (
    <main className="relative min-h-screen font-sans text-slate-200 overflow-x-hidden flex flex-col selection:bg-indigo-500/30 pb-32 bg-transparent">
      <VideoBackground />

      {/* Navigation */}
      <nav className="w-full flex justify-between items-center px-6 py-8 md:px-12 z-50">
         <div className="flex items-center gap-2 font-semibold text-xl tracking-tight text-white">
            <ShieldCheck className="text-indigo-400" size={24} />
            DeviceDNA
         </div>
         <div className="flex items-center gap-8 text-sm font-medium">
            <a href="https://github.com/karthik5033/DeviceDNA" target="_blank" rel="noreferrer" className="text-slate-400 hover:text-white transition-colors hidden md:block">Documentation</a>
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-white/10 bg-white/5 backdrop-blur-md">
               <div className={cn("w-2 h-2 rounded-full",
                 apiStatus === 'online' ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]' : 
                 apiStatus === 'offline' ? 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]' : 
                 'bg-amber-500 animate-pulse'
               )} />
               <span className="text-xs uppercase tracking-widest text-slate-300">
                  {apiStatus}
               </span>
            </div>
         </div>
      </nav>

      {/* HERO SECTION */}
      <div className="flex-1 flex flex-col items-center justify-center px-4 md:px-6 z-10 pt-16 md:pt-24 pb-20">
        <motion.div 
           initial={{ opacity: 0, y: 20 }}
           animate={{ opacity: 1, y: 0 }}
           transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
           className="max-w-5xl w-full flex flex-col items-center text-center"
        >
          {/* Subtle Announcement Pill */}
          <div className="inline-flex items-center gap-3 px-3 py-1.5 rounded-full border border-indigo-500/20 bg-indigo-500/10 text-indigo-200 text-xs font-medium tracking-wide backdrop-blur-md mb-8 hover:bg-indigo-500/20 transition-colors cursor-pointer group">
             <span className="bg-indigo-500 text-white px-2 py-0.5 rounded-full text-[10px] font-bold tracking-wider">NEW</span>
             <span>Protocol V1.0 is now live</span>
             <ChevronRight size={14} className="text-indigo-400 group-hover:translate-x-0.5 transition-transform" />
          </div>

          {/* Hero Typography */}
          <h1 className="text-5xl md:text-7xl lg:text-[5.5rem] font-bold tracking-tight leading-[1.05] mb-6 text-white text-balance">
            Zero-Trust Security.
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 via-white to-slate-400">
               Automated & Intelligent.
            </span>
          </h1>

          <p className="text-lg md:text-xl text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed text-balance font-light">
            The multi-dimensional machine learning platform. Map lateral movement, detect statistical drift, and neutralize threats autonomously before the breach occurs.
          </p>
          
          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 w-full">
            <Link 
              href="/dashboard"
              className="group relative inline-flex items-center justify-center gap-2 font-medium text-black bg-white rounded-full px-8 py-4 transition-all w-full sm:w-auto hover:bg-slate-200 hover:scale-[1.02] active:scale-95 shadow-[0_0_30px_rgba(255,255,255,0.1)]"
            >
              Enter SOC Dashboard <ArrowRight size={18} className="transition-transform group-hover:translate-x-1" />
            </Link>

            <a 
              href="https://github.com/karthik5033/DeviceDNA" target="_blank" rel="noreferrer"
              className="inline-flex items-center justify-center gap-2 font-medium text-white bg-white/5 border border-white/10 hover:border-white/20 hover:bg-white/10 rounded-full px-8 py-4 transition-all w-full sm:w-auto backdrop-blur-md hover:scale-[1.02] active:scale-95"
            >
              <Code size={18} /> View Architecture
            </a>
          </div>
        </motion.div>

        {/* Sleek Dashboard Preview Window */}
        <motion.div 
           initial={{ opacity: 0, y: 40 }}
           animate={{ opacity: 1, y: 0 }}
           transition={{ duration: 1.2, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
           className="w-full flex justify-center mt-20 md:mt-32 px-4"
        >
           <div className="w-full max-w-5xl bg-[#0a0a0a]/80 backdrop-blur-2xl border border-white/10 rounded-2xl shadow-2xl flex flex-col overflow-hidden relative ring-1 ring-white/5">
               
               {/* Ambient top glow */}
               <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3/4 h-[1px] bg-gradient-to-r from-transparent via-indigo-500/50 to-transparent" />

               {/* Window Controls */}
               <div className="w-full h-12 border-b border-white/5 bg-white/[0.02] flex items-center px-4 gap-2 z-10">
                   <div className="flex gap-1.5">
                       <div className="w-2.5 h-2.5 rounded-full bg-slate-600/50" />
                       <div className="w-2.5 h-2.5 rounded-full bg-slate-600/50" />
                       <div className="w-2.5 h-2.5 rounded-full bg-slate-600/50" />
                   </div>
                   <div className="mx-auto text-[10px] font-mono text-slate-500 tracking-widest uppercase flex items-center gap-2">
                       <ShieldCheck size={12} /> Live Telemetry Feed
                   </div>
                   <div className="w-10" /> {/* Spacer for centering */}
               </div>
               
               {/* Refined Interface Mockup */}
               <div className="p-8 md:p-12 w-full relative grid grid-cols-1 md:grid-cols-3 gap-6 bg-gradient-to-b from-transparent to-[#050505]">
                   {/* Card 1 */}
                   <div className="col-span-1 bg-white/[0.02] border border-white/5 rounded-xl p-6 flex flex-col relative overflow-hidden group hover:bg-white/[0.04] transition-colors">
                      <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 blur-[40px] rounded-full group-hover:bg-indigo-500/20 transition-colors" />
                      <span className="text-slate-400 text-xs font-medium tracking-wider uppercase mb-8">Active Edge Nodes</span>
                      <div className="flex items-end gap-3 mt-auto">
                         <span className="text-4xl md:text-5xl font-light text-white tracking-tight">1,204</span>
                         <span className="text-emerald-400 text-sm font-medium mb-1.5">+12%</span>
                      </div>
                   </div>

                   {/* Card 2 (Terminal/Logs) */}
                   <div className="col-span-1 md:col-span-2 bg-[#000000] border border-white/5 rounded-xl p-6 flex flex-col font-mono text-xs text-slate-400 relative overflow-hidden shadow-inner">
                      <div className="flex justify-between items-center mb-4 border-b border-white/5 pb-2">
                         <span className="text-slate-500">system_logs.sh</span>
                         <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                      </div>
                      <div className="space-y-2">
                         <p><span className="text-indigo-400">→</span> Analyzing flow: MED-0007</p>
                         <p><span className="text-indigo-400">→</span> Extracting 14D vector...</p>
                         <p><span className="text-emerald-400">✓</span> VAE Reconstruction MSE: 0.12</p>
                         <p><span className="text-emerald-400">✓</span> Isolation Forest Score: 0.04</p>
                         <p className="text-slate-500">Waiting for next batch...</p>
                      </div>
                   </div>
               </div>
           </div>
        </motion.div>
      </div>

      {/* DETAILED SECTIONS */}
      <div className="w-full max-w-5xl mx-auto px-6 mt-32 z-10 relative">
        <div className="text-center mb-20">
          <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-6 text-white">The Intelligence Architecture.</h2>
          <p className="text-slate-400 max-w-2xl mx-auto text-lg leading-relaxed font-light">
            By rejecting rigid legacy IPS patterns, the DeviceDNA pipeline processes continuous high-throughput Kafka telemetry streams through an ensemble of complex models simultaneously.
          </p>
        </div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
          {features.map((feature, i) => (
            <div 
              key={i}
              className="bg-white/[0.02] border border-white/5 rounded-2xl p-8 hover:bg-white/[0.04] transition-colors duration-300"
            >
              <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center mb-6">
                 <feature.icon className="w-5 h-5 text-indigo-300" />
              </div>
              <h3 className="text-xl font-semibold mb-3 tracking-tight text-slate-100">{feature.title}</h3>
              <p className="text-slate-400 leading-relaxed text-sm font-light">{feature.desc}</p>
            </div>
          ))}
        </div>

        {/* Architecture Diagram */}
        <div className="mt-32 border border-white/5 rounded-3xl bg-[#050505]/50 backdrop-blur-xl p-8 lg:p-12 relative overflow-hidden ring-1 ring-white/5">
          <div className="absolute top-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-white/10 to-transparent" />
          
          <h2 className="text-xl font-semibold mb-12 text-center tracking-wide text-slate-200">End-to-End Pipeline</h2>
          
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 font-mono text-xs md:text-sm relative z-10">
            <div className="hidden md:block absolute top-1/2 left-[10%] w-[80%] h-[1px] bg-white/5 -z-10" />
            
            <div className="bg-[#0a0a0a] border border-white/10 rounded-xl p-6 text-center w-full md:w-[30%]">
              <span className="text-indigo-300 font-medium block mb-2">Apache Kafka</span>
              <span className="text-slate-500">High-throughput ingestion</span>
            </div>
            
            <div className="bg-white/[0.03] border border-indigo-500/30 rounded-xl p-8 text-center w-full md:w-[35%] relative shadow-[0_0_30px_rgba(99,102,241,0.05)]">
              <span className="text-white font-semibold block mb-3 text-base">FastAPI Core</span>
              <span className="text-indigo-200 font-sans block bg-indigo-500/10 rounded-full py-1 px-3 border border-indigo-500/20 inline-block text-xs">PyTorch • Scikit-Learn</span>
            </div>
            
            <div className="bg-[#0a0a0a] border border-white/10 rounded-xl p-6 text-center w-full md:w-[30%]">
              <span className="text-indigo-300 font-medium block mb-2">Next.js Edge</span>
              <span className="text-slate-500">WebSockets + D3</span>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
