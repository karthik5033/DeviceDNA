'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { motion, useAnimation } from 'framer-motion';
import { ShieldCheck, ArrowRight, Activity, Terminal, Lock, Network } from 'lucide-react';
import { cn } from '@/lib/utils';

// Abstract Floating Network Node Background Component
const AnimatedBackground = () => {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none -z-10 bg-[#070b14]">
      {/* Dynamic Grid */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_110%)] opacity-20" />
      
      {/* Glowing Orbs */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] rounded-full bg-blue-600/20 blur-[120px]" />
      <div className="absolute top-[20%] right-[-10%] w-[30%] h-[50%] rounded-full bg-cyan-500/10 blur-[120px]" />
      
      {/* Floating abstract network paths */}
      <svg className="absolute w-full h-full opacity-30" viewBox="0 0 100 100" preserveAspectRatio="none">
         <motion.path 
            d="M 0 50 Q 25 30 50 50 T 100 50" 
            fill="transparent" 
            stroke="url(#grad1)" 
            strokeWidth="0.1"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.5, y: [0, -2, 0] }}
            transition={{ duration: 4, repeat: Infinity, repeatType: "reverse", ease: "easeInOut" }}
         />
         <motion.path 
            d="M 0 80 Q 30 90 60 70 T 100 60" 
            fill="transparent" 
            stroke="url(#grad2)" 
            strokeWidth="0.05"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ pathLength: 1, opacity: 0.3, y: [0, 2, 0] }}
            transition={{ duration: 5, repeat: Infinity, repeatType: "reverse", ease: "easeInOut", delay: 1 }}
         />
         <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
               <stop offset="0%" stopColor="#3edcff" stopOpacity="0" />
               <stop offset="50%" stopColor="#3edcff" stopOpacity="1" />
               <stop offset="100%" stopColor="#3edcff" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="0%">
               <stop offset="0%" stopColor="#ef4444" stopOpacity="0" />
               <stop offset="50%" stopColor="#ef4444" stopOpacity="1" />
               <stop offset="100%" stopColor="#ef4444" stopOpacity="0" />
            </linearGradient>
         </defs>
      </svg>
    </div>
  );
};

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
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.2 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { type: 'spring', stiffness: 100, damping: 15 } }
  };

  return (
    <main className="relative min-h-screen font-sans text-white overflow-hidden flex flex-col selection:bg-[#3edcff]/30">
      <AnimatedBackground />

      {/* Navigation Layer */}
      <nav className="w-full flex justify-between items-center p-6 lg:px-12 z-50">
         <div className="flex items-center gap-2 font-bold text-xl tracking-tighter">
            <Lock className="text-[#3edcff]" size={20} />
            Device<span className="text-[#3edcff]">DNA</span>
         </div>
         <div className="flex items-center gap-4">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full border border-[#1e293b] bg-[#111827]/50 backdrop-blur text-xs font-mono">
               <div className={cn("w-2 h-2 rounded-full",
                 apiStatus === 'online' ? 'bg-green-500 shadow-[0_0_8px_#22c55e]' : 
                 apiStatus === 'offline' ? 'bg-red-500 shadow-[0_0_8px_#ef4444]' : 
                 'bg-yellow-500 animate-pulse'
               )} />
               API Status: {apiStatus.toUpperCase()}
            </div>
            <Link href="/dashboard" className="text-sm font-medium text-gray-300 hover:text-white transition-colors relative group">
               Enter SOC
               <span className="absolute -bottom-1 left-0 w-0 h-[1px] bg-[#3edcff] transition-all group-hover:w-full"></span>
            </Link>
         </div>
      </nav>

      <div className="flex-1 flex flex-col items-center justify-center -mt-16 px-4 z-10 relative">
        <motion.div 
           className="max-w-5xl w-full flex flex-col items-center text-center"
           variants={containerVariants}
           initial="hidden"
           animate="visible"
        >
          {/* Top Pill */}
          <motion.div variants={itemVariants} className="mb-8">
             <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-[#3edcff]/30 bg-[#3edcff]/5 text-[#3edcff] text-xs font-semibold tracking-wide backdrop-blur-md">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#3edcff] opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-[#3edcff]"></span>
                </span>
                DeviceDNA V1.0 ML Engine Live
             </div>
          </motion.div>

          {/* Hero Typography */}
          <motion.h1 variants={itemVariants} className="text-5xl md:text-7xl lg:text-[5.5rem] font-extrabold tracking-tighter leading-[1.1] mb-6">
            Securing the <span className="text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-500">Deep Edge.</span>
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#3edcff] via-blue-500 to-cyan-400">
               Autonomous IoT Defense.
            </span>
          </motion.h1>

          <motion.p variants={itemVariants} className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            The world's first multi-dimensional, unsupervised ML cyber platform. DeviceDNA maps lateral movement, detects CUSUM drift, and severs compromised hardware before the breach.
          </motion.p>
          
          {/* CTAs */}
          <motion.div variants={itemVariants} className="flex flex-col sm:flex-row items-center justify-center gap-4 w-full relative">
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[120%] h-[120%] bg-[#3edcff]/10 blur-3xl rounded-full -z-10" />
            
            <Link 
              href="/dashboard"
              className="group relative inline-flex items-center justify-center gap-2 font-bold text-black bg-white hover:bg-gray-100 rounded-full px-8 py-3.5 transition-all overflow-hidden w-full sm:w-auto hover:scale-105 active:scale-95"
            >
              <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/40 to-transparent -translate-x-[200%] group-hover:translate-x-[200%] transition-transform duration-700 ease-in-out" />
              Access SOC Dashboard <ArrowRight size={18} className="transition-transform group-hover:translate-x-1" />
            </Link>

            <a 
              href="https://github.com/karthik5033/DeviceDNA" target="_blank" rel="noreferrer"
              className="inline-flex items-center justify-center gap-2 font-semibold text-white bg-[#1e293b]/50 hover:bg-[#1e293b] border border-[#334155] rounded-full px-8 py-3.5 transition-all w-full sm:w-auto hover:text-[#3edcff] hover:border-[#3edcff]/50 backdrop-blur-sm"
            >
              <Terminal size={18} /> View Architecture
            </a>
          </motion.div>

        </motion.div>
      </div>

      {/* Floating 3D Mockup Perspective at the bottom */}
      <motion.div 
         initial={{ y: 150, opacity: 0, rotateX: 20 }}
         animate={{ y: 0, opacity: 1, rotateX: 0 }}
         transition={{ duration: 1, delay: 0.6, type: "spring", stiffness: 50 }}
         style={{ perspective: "1200px" }}
         className="w-full absolute bottom-0 left-0 flex justify-center translate-y-24"
      >
         <div className="w-[80%] max-w-6xl h-64 bg-gradient-to-t from-[#070b14] via-[#111827] to-[#1e293b] border-t border-l border-r border-[#334155] rounded-t-3xl shadow-[0_-20px_50px_rgba(62,220,255,0.05)] flex p-6 gap-6 relative overflow-hidden backdrop-blur-xl group">
             {/* Fake UI Elements inside the mockup */}
             <div className="w-1/4 h-full flex flex-col gap-4">
                <div className="w-full h-8 bg-black/20 rounded-md border border-[#334155]/50 flex items-center px-3"><div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" /></div>
                <div className="w-full h-24 bg-black/20 rounded-md border border-[#334155]/50 flex gap-2 p-2">
                   <div className="w-1/2 h-full bg-[#3edcff]/10 rounded" />
                   <div className="w-1/2 h-full bg-[#ef4444]/10 rounded" />
                </div>
             </div>
             <div className="flex-1 h-full bg-black/30 rounded-lg border border-[#334155]/50 relative overflow-hidden">
                <Network className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-[#334155] opacity-30 w-32 h-32 group-hover:scale-110 transition-transform duration-1000 group-hover:opacity-50" />
                <div className="absolute inset-0 bg-gradient-to-t from-[#070b14] to-transparent pointer-events-none" />
             </div>
         </div>
      </motion.div>
    </main>
  );
}
