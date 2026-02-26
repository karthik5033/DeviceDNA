'use client';
import Link from 'next/link';
import { useEffect, useState } from 'react';

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
    <main className="min-h-screen bg-[#070b14] text-white flex flex-col items-center justify-center p-24 font-ui">
      <div className="z-10 max-w-5xl w-full items-center flex flex-col justify-center">
        <h1 className="text-6xl font-bold text-center tracking-tight mb-4">
          Device<span className="text-[#3edcff]">DNA</span>
        </h1>
        <p className="text-xl text-center text-gray-400 mb-12 max-w-2xl mx-auto">
          AI-Powered IoT Cybersecurity Intelligence Platform.<br/>
          Detects drift, models behavior, and prevents breaches.
        </p>
        
        <div className="flex justify-center flex-wrap gap-6 mb-20">
          <Link 
            href="/dashboard"
            className="group rounded-lg border border-white/10 bg-white/5 px-6 py-5 transition-all hover:border-gray-500 hover:bg-white/10 text-center"
          >
            <h2 className="mb-3 text-2xl font-semibold">
              Enter SOC <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">-&gt;</span>
            </h2>
            <p className="m-0 max-w-[30ch] text-sm text-gray-400">
              Access the main IoT security dashboard and visualization center.
            </p>
          </Link>
        </div>

        <div className="text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-gray-800 bg-gray-900/50">
            <div className={`w-2 h-2 rounded-full ${
              apiStatus === 'online' ? 'bg-green-500 shadow-[0_0_10px_rgba(34,197,94,0.5)]' : 
              apiStatus === 'offline' ? 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]' : 
              'bg-yellow-500 animate-pulse'
            }`} />
            <span className="text-sm font-mono text-gray-300">
              API Status: {apiStatus.toUpperCase()}
            </span>
          </div>
        </div>
      </div>
    </main>
  );
}
