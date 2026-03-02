'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { motion, useAnimation } from 'framer-motion';
import { ShieldCheck, ArrowRight, Activity, Terminal, Lock, Network } from 'lucide-react';
import { cn } from '@/lib/utils';

// Abstract Floating Network Node Background Component
const AnimatedBackground = () => {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none -z-10 bg-[#020617]">
      {/* Premium Dark Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_80%_at_50%_10%,#000_20%,transparent_100%)] opacity-30" />
      
      {/* High-Contrast Architectural Glowing Orbs */}
      <motion.div 
         animate={{ scale: [1, 1.2, 1], opacity: [0.1, 0.2, 0.1] }}
         transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
         className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-blue-600 blur-[150px]" 
      />
      <motion.div 
         animate={{ scale: [1, 1.3, 1], opacity: [0.1, 0.15, 0.1] }}
         transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 2 }}
         className="absolute top-[30%] right-[-20%] w-[40%] h-[60%] rounded-full bg-cyan-400 blur-[150px]" 
      />
      
      {/* Vertical Data Streams */}
      <div className="absolute top-0 left-1/4 w-[1px] h-full bg-gradient-to-b from-transparent via-[#3edcff]/50 to-transparent opacity-20" />
      <div className="absolute top-0 right-1/4 w-[1px] h-full bg-gradient-to-b from-transparent via-blue-500/50 to-transparent opacity-20" />
    </div>
  );
};

// Simulated Mini Data Streams for the Hero
const DataStream = ({ delay, x }: { delay: number, x: string }) => (
   <motion.div 
      initial={{ y: "-100%", opacity: 0 }}
      animate={{ y: "1500%", opacity: [0, 1, 0] }}
      transition={{ duration: 5, repeat: Infinity, delay, ease: "linear" }}
      className={`absolute top-0 w-[2px] h-24 bg-gradient-to-b from-transparent via-[#3edcff] to-transparent shadow-[0_0_15px_#3edcff]`}
      style={{ left: x }}
   />
);

const features = [
  { title: "Behavioral CUSUM Drift", desc: "Monitors statistical accumulation to detect slow-acting threats like persistent data exfiltration over days instead of seconds.", icon: Activity },
  { title: "GraphSAGE Lateral Maps", desc: "Constructs live topological maps using D3 and identifies malicious traversal between isolated hardware nodes.", icon: Network },
  { title: "Isolation Forests", desc: "Immediately detects structural anomalies in payload sizes and port destinations mathematically independent of known signatures.", icon: Lock },
  { title: "NLP Rule Compilers", desc: "Translates human-readable English policies into rigid firewall enforcement logic via BERT embeddings instantly.", icon: Terminal }
];

// Live Simulated Graph Node Background for the Mockup box
const LiveGraphMockup = () => {
    return (
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
            {Array.from({ length: 15 }).map((_, i) => (
                <motion.div
                    key={i}
                    initial={{ 
                        x: Math.random() * 800, 
                        y: Math.random() * 300,
                        scale: 0
                    }}
                    animate={{ 
                        x: Math.random() * 800, 
                        y: Math.random() * 300,
                        scale: [0, 1, 1, 0]
                    }}
                    transition={{ 
                        duration: Math.random() * 10 + 10, 
                        repeat: Infinity,
                        repeatType: "reverse",
                        ease: "linear"
                    }}
                    className={`absolute w-3 h-3 rounded-full ${Math.random() > 0.8 ? 'bg-red-500 shadow-[0_0_15px_#ef4444]' : 'bg-[#3edcff] shadow-[0_0_10px_#3edcff]'}`}
                >
                    <div className="absolute top-1/2 left-1/2 w-32 h-[1px] bg-gradient-to-r from-current to-transparent origin-left rotate-45 opacity-20" />
                </motion.div>
            ))}
        </div>
    );
}

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

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1, transition: { staggerChildren: 0.15, delayChildren: 0.1 } }
  };

  const itemVariants = {
    hidden: { opacity: 0, filter: "blur(10px)", y: 30 },
    visible: { opacity: 1, filter: "blur(0px)", y: 0, transition: { duration: 0.8, ease: [0.16, 1, 0.3, 1] } }
  };

  return (
    <main className="relative min-h-screen font-sans text-white overflow-x-hidden flex flex-col selection:bg-[#3edcff]/30 pb-24 bg-[#020617]">
      <AnimatedBackground />
      <DataStream delay={0.5} x="15%" />
      <DataStream delay={3.2} x="85%" />
      <DataStream delay={1.8} x="50%" />

      {/* Navigation Layer */}
      <nav className="w-full flex justify-between items-center p-6 lg:px-12 z-50 relative mix-blend-difference">
         <div className="flex items-center gap-2 font-bold text-xl tracking-tighter">
            <Lock className="text-[#3edcff]" size={20} />
            Device<span className="text-[#3edcff]">DNA</span>
         </div>
         <div className="flex items-center gap-6 text-sm font-medium">
            <a href="https://github.com/karthik5033/DeviceDNA" target="_blank" rel="noreferrer" className="text-gray-400 hover:text-white transition-colors hidden sm:block">Architecture</a>
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full border border-white/10 bg-black/50 backdrop-blur font-mono">
               <div className={cn("w-2 h-2 rounded-full",
                 apiStatus === 'online' ? 'bg-green-500 shadow-[0_0_8px_#22c55e]' : 
                 apiStatus === 'offline' ? 'bg-red-500 shadow-[0_0_8px_#ef4444]' : 
                 'bg-yellow-500 animate-pulse'
               )} />
               {apiStatus.toUpperCase()}
            </div>
         </div>
      </nav>

      {/* HERO SECTION */}
      <div className="min-h-[90vh] flex flex-col items-center justify-center px-4 z-10 relative mt-[-80px]">
        <motion.div 
           className="max-w-6xl w-full flex flex-col items-center text-center"
           variants={containerVariants}
           initial="hidden"
           animate="visible"
        >
          {/* Top Pill / Badge */}
          <motion.div variants={itemVariants} className="mb-10">
             <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-white/10 bg-white/5 text-gray-300 text-xs font-semibold tracking-wide backdrop-blur-md shadow-2xl">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#3edcff] opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-[#3edcff]"></span>
                </span>
                V1.0 Edge Machine Learning Protocol
             </div>
          </motion.div>

          {/* Hero Typography - Premium Layout */}
          <motion.h1 variants={itemVariants} className="text-5xl md:text-7xl lg:text-[6.5rem] font-bold tracking-tighter leading-[1] mb-8 text-balance">
            Zero-Trust Security,
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-br from-[#3edcff] via-blue-400 to-indigo-600 animate-gradient-x">
               Solved Automatically.
            </span>
          </motion.h1>

          <motion.p variants={itemVariants} className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mb-12 leading-relaxed text-balance">
            The multi-dimensional, unsupervised ML cyber platform. DeviceDNA maps lateral movement, detects CUSUM drift, and severs compromised hardware before the breach occurs.
          </motion.p>
          
          {/* CTAs */}
          <motion.div variants={itemVariants} className="flex flex-col sm:flex-row items-center justify-center gap-5 w-full relative">
            <Link 
              href="/dashboard"
              className="group relative inline-flex items-center justify-center gap-2 font-semibold text-black bg-white rounded-full px-8 py-4 transition-all overflow-hidden w-full sm:w-auto hover:bg-gray-100 shadow-[0_0_40px_rgba(255,255,255,0.2)] hover:shadow-[0_0_60px_rgba(255,255,255,0.3)] active:scale-95"
            >
              Enter SOC Dashboard <ArrowRight size={18} className="transition-transform group-hover:translate-x-1" />
            </Link>

            <a 
              href="https://github.com/karthik5033/DeviceDNA" target="_blank" rel="noreferrer"
              className="inline-flex items-center justify-center gap-2 font-semibold text-white bg-transparent border border-white/20 hover:border-white/50 hover:bg-white/5 rounded-full px-8 py-4 transition-all w-full sm:w-auto backdrop-blur-sm active:scale-95"
            >
              <Code size={18} /> Documentation
            </a>
          </motion.div>
        </motion.div>

        {/* Floating Abstract Application Window */}
        <motion.div 
           initial={{ y: 200, opacity: 0, scale: 0.9, rotateX: 25 }}
           animate={{ y: 0, opacity: 1, scale: 1, rotateX: 0 }}
           transition={{ duration: 1.2, delay: 0.8, type: "spring", stiffness: 40, damping: 20 }}
           style={{ perspective: "1500px" }}
           className="w-full flex justify-center mt-24 px-6 md:px-0"
        >
           <div className="w-full max-w-5xl h-64 md:h-96 bg-[#0B1221]/90 backdrop-blur-2xl border border-white/10 rounded-2xl shadow-[0_-30px_100px_rgba(62,220,255,0.15)] flex flex-col overflow-hidden relative group ring-1 ring-white/5">
               {/* Window Controls */}
               <div className="w-full h-12 bg-white/5 border-b border-white/10 flex items-center px-4 gap-2 z-10">
                   <div className="flex gap-2">
                       <div className="w-3 h-3 rounded-full bg-red-500/80" />
                       <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
                       <div className="w-3 h-3 rounded-full bg-green-500/80" />
                   </div>
                   <div className="ml-auto text-xs font-mono text-gray-500 flex items-center gap-2 opacity-50">
                       <ShieldCheck size={14} /> Global Threat Overview
                   </div>
               </div>
               
               {/* Live Background Graph Mockup */}
               <div className="flex-1 w-full relative">
                   <LiveGraphMockup />
                   <div className="absolute inset-0 bg-gradient-to-t from-[#020617] via-transparent to-transparent pointer-events-none" />
               </div>
           </div>
        </motion.div>
      </div>

      {/* DETAILED SECTIONS */}
      <div className="w-full max-w-6xl mx-auto px-6 mt-40 z-10 relative">
        <motion.div 
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.8 }}
          className="text-center mb-20"
        >
          <h2 className="text-3xl md:text-5xl font-bold tracking-tighter mb-6 text-balance">The Intelligence Architecture.</h2>
          <p className="text-gray-400 max-w-2xl mx-auto text-lg leading-relaxed text-balance">
            By rejecting rigid legacy IPS patterns, the DeviceDNA pipeline processes continuous high-throughput Kafka telemetry streams through an ensemble of complex models simultaneously.
          </p>
        </motion.div>

        {/* Feature Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {features.map((feature, i) => (
            <motion.div 
              key={i}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ duration: 0.5, delay: i * 0.1 }}
              className="bg-white/[0.02] backdrop-blur-lg border border-white/10 rounded-3xl p-8 hover:bg-white/[0.04] hover:border-white/20 transition-all duration-300 group relative overflow-hidden"
            >
              <div className="absolute top-0 right-0 w-48 h-48 bg-gradient-to-bl from-blue-500/10 to-transparent rounded-full blur-3xl group-hover:from-[#3edcff]/20 transition-all duration-500" />
              <div className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 flex items-center justify-center mb-6">
                 <feature.icon className="w-6 h-6 text-[#3edcff]" />
              </div>
              <h3 className="text-2xl font-bold mb-3 tracking-tight">{feature.title}</h3>
              <p className="text-gray-400 leading-relaxed text-sm md:text-base">{feature.desc}</p>
            </motion.div>
          ))}
        </div>

        {/* Architecture Diagram Fake */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mt-32 border border-white/10 rounded-3xl bg-white/[0.02] backdrop-blur-xl p-8 lg:p-16 relative overflow-hidden"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 via-blue-900/5 to-transparent pointer-events-none" />
          <h2 className="text-2xl md:text-3xl font-bold mb-12 text-center tracking-tight text-white"><ShieldCheck className="inline-block mr-2 mb-1 text-[#3edcff]" /> End-to-End Pipeline</h2>
          
          <div className="flex flex-col md:flex-row items-center justify-between gap-6 md:gap-4 font-mono text-sm relative">
            {/* Arrows behind */}
            <div className="hidden md:block absolute top-[40%] left-[10%] w-[80%] h-[2px] bg-gradient-to-r from-transparent via-[#334155]/50 to-transparent shadow-[0_0_10px_rgba(255,255,255,0.1)] -z-10" />
            
            <div className="bg-[#0f172a]/90 backdrop-blur border border-white/10 rounded-2xl p-6 text-center w-full md:w-[30%] shadow-lg">
              <span className="text-[#3edcff] font-bold block mb-2 text-lg tracking-wider">Apache Kafka</span>
              <span className="text-gray-500 text-xs">High-throughput packet ingestion</span>
            </div>
            
            <div className="bg-gradient-to-b from-[#1e293b] to-[#0f172a] border border-[#3edcff]/30 rounded-2xl p-8 text-center shadow-[0_0_40px_rgba(62,220,255,0.1)] w-full md:w-[35%] z-10 relative">
              <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3/4 h-[1px] bg-gradient-to-r from-transparent via-[#3edcff] to-transparent" />
              <span className="text-white font-bold block mb-3 text-xl tracking-tight">FastAPI Core</span>
              <span className="text-cyan-400/80 text-xs font-sans block bg-[#3edcff]/10 rounded-full py-1 px-2 border border-[#3edcff]/20">PyTorch • Scikit-Learn</span>
            </div>
            
            <div className="bg-[#0f172a]/90 backdrop-blur border border-white/10 rounded-2xl p-6 text-center w-full md:w-[30%] shadow-lg">
              <span className="text-indigo-400 font-bold block mb-2 text-lg tracking-wider">Next.js Edge</span>
              <span className="text-gray-500 text-xs">WebSockets + D3 Render</span>
            </div>
          </div>
        </motion.div>
      </div>
    </main>
  );
}
