'use client';

import { useState } from 'react';
import { Search, BrainCircuit, Code, Play, CheckCircle2, ShieldOff, Sparkles } from 'lucide-react';

const MOCK_POLICIES = [
  { id: 'NLP-01', text: 'Alert if any device attempts to connect to an IP outside the subnet after midnight.', status: 'active', matchCount: 14, lastMatched: '2h ago', risk: 'High' },
  { id: 'NLP-02', text: 'Thermostats must never transmit more than 5MB of data in a 5 minute interval.', status: 'active', matchCount: 0, lastMatched: 'Never', risk: 'Critical' },
  { id: 'NLP-03', text: 'Block the IP Camera SIM-0012 from speaking to any other camera on the LAN.', status: 'inactive', matchCount: 3, lastMatched: '1d ago', risk: 'Medium' },
  { id: 'NLP-04', text: 'Flag if the NLP policy violation simulator scenario executes.', status: 'active', matchCount: 112, lastMatched: 'Just now', risk: 'High' }
];

export default function NLPPoliciesPage() {
  const [query, setQuery] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);
  const [translationResult, setTranslationResult] = useState<any>(null);

  const simulateTranslation = () => {
    if (!query) return;
    setIsTranslating(true);
    setTranslationResult(null);
    
    // Fake the BERT translation delay
    setTimeout(() => {
        setIsTranslating(false);
        setTranslationResult({
          intent: "BLOCK_BEHAVIOR",
          entities: {
              device_class: "Thermostat",
              action: "UPLOAD_BYTES",
              threshold: "5000000", // 5MB
              timeframe: "300s"
          },
          confidence: "98.4%",
          generatedRule: `IF dev_class == 'thermostat' AND upload_bytes_5m > 5000000 THEN ACTION=BLOCK_AND_ALERT`
        });
    }, 1500);
  }

  return (
    <div className="flex flex-col gap-6 max-w-[1600px] mx-auto fade-in h-[calc(100vh-100px)]">
      
      {/* Policy Engine Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tighter mb-1 font-sans flex items-center gap-3">
            <BrainCircuit className="text-[#3edcff]" size={28} />
            NLP Policy Engine
          </h1>
          <p className="text-gray-400 text-sm">Write human-readable rules. BERT transforms them into live network constraints.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 flex-1 overflow-hidden">
        
        {/* Left Side: The Interactive Prompt Pane */}
        <div className="flex flex-col gap-4 bg-[#111827] border border-[#1e293b] rounded-xl p-6 shadow-xl relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-[#3edcff] to-blue-600 shadow-[0_0_15px_#3edcff] pointer-events-none" />
          
          <h2 className="text-lg font-bold text-gray-200 flex items-center gap-2 mb-2">
            <Sparkles className="text-yellow-500" /> New Language Rule
          </h2>
          
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="E.g., Flag any device behaving like a camera that connects to a Russian IP..."
            className="w-full h-32 bg-[#070b14]/50 border border-[#334155] rounded-xl p-4 text-white text-md min-h-[120px] focus:outline-none focus:border-[#3edcff] focus:ring-1 focus:ring-[#3edcff] transition-all resize-none font-ui shadow-inner leading-relaxed"
          />

          <button 
             onClick={simulateTranslation}
             disabled={!query || isTranslating}
             className="ml-auto px-6 py-2 bg-[#3edcff] hover:bg-blue-500 text-[#070b14] rounded-lg font-bold shadow-[0_0_15px_#3edcff]/50 hover:shadow-[0_0_20px_#3edcff] transition-all duration-300 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
             {isTranslating ? <span className="flex items-center gap-2"><div className="w-4 h-4 border-2 border-[#070b14] border-t-transparent rounded-full animate-spin"></div> Translating...</span> : 'Translate to Ruleset'}
          </button>

          {/* Result Block */}
          <div className={`mt-6 transition-all duration-500 flex-1 overflow-y-auto ${translationResult ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4 pointer-events-none'}`}>
             {translationResult && (
               <div className="bg-[#070b14] border border-green-500/30 rounded-lg p-5">
                 <div className="flex justify-between items-center mb-4 border-b border-[#1e293b] pb-2">
                    <h3 className="text-green-500 font-bold flex items-center gap-2 tracking-tight">
                       <CheckCircle2 size={16} /> Translation Successful
                    </h3>
                    <span className="text-xs font-mono text-gray-500">Confidence: <span className="text-[#3edcff] font-bold">{translationResult.confidence}</span></span>
                 </div>
                 
                 <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                    <div className="bg-[#111827] border border-[#1e293b] p-3 rounded">
                      <span className="text-xs text-gray-400 block mb-1">Extracted Intent:</span>
                      <span className="font-mono text-red-400 font-bold">{translationResult.intent}</span>
                    </div>
                    <div className="bg-[#111827] border border-[#1e293b] p-3 rounded">
                      <span className="text-xs text-gray-400 block mb-1">Entities Recognized:</span>
                      <pre className="font-mono text-xs text-yellow-500">{JSON.stringify(translationResult.entities, null, 2)}</pre>
                    </div>
                 </div>

                 <div className="bg-gray-900 border border-gray-700 p-4 rounded-lg relative group">
                    <span className="absolute -top-3 left-4 bg-[#111827] px-2 text-xs text-gray-500 font-mono flex items-center gap-1"><Code size={12}/> Compiled Rule</span>
                    <code className="text-[#3edcff] font-mono text-sm leading-relaxed block overflow-x-auto whitespace-pre">
                      {translationResult.generatedRule}
                    </code>
                 </div>

                 <button className="w-full mt-4 py-2 bg-green-500/10 hover:bg-green-500/20 text-green-500 border border-green-500/50 rounded-lg text-sm font-bold transition-colors flex justify-center items-center gap-2">
                    <Play size={16} /> Deploy Policy to Live Tracking
                 </button>
               </div>
             )}
          </div>
        </div>

        {/* Right Side: Active Policy Ledger */}
        <div className="flex flex-col bg-[#111827] border border-[#1e293b] rounded-xl relative overflow-hidden h-full">
           <div className="p-4 border-b border-[#1e293b] flex justify-between items-center bg-[#111827] z-10 sticky top-0">
             <h2 className="text-lg font-bold text-gray-200">Active Rule Ledger</h2>
             
             <div className="relative text-gray-400">
               <Search className="absolute left-2.5 top-1/2 -translate-y-1/2" size={14} />
               <input type="text" placeholder="Filter policies..." className="bg-[#070b14] border border-[#334155] rounded text-xs py-1.5 pl-8 pr-3 text-white focus:outline-none w-48" />
             </div>
           </div>

           <div className="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar">
             {MOCK_POLICIES.map((policy) => (
               <div key={policy.id} className="bg-[#070b14] border border-[#1e293b] rounded-lg p-4 hover:border-[#334155] transition-colors relative">
                 <div className="flex justify-between items-start mb-2">
                   <div className="flex items-center gap-2">
                     <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-[#1e293b] text-gray-300">{policy.id}</span>
                     {policy.status === 'active' ? (
                       <span className="text-xs px-2 py-0.5 rounded text-green-500 border border-green-500/30 flex items-center gap-1"><div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"/> Tracking</span>
                     ) : (
                       <span className="text-xs px-2 py-0.5 rounded text-gray-500 border border-gray-600 flex items-center gap-1"><ShieldOff size={10} /> Inactive</span>
                     )}
                   </div>
                   
                   <span className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider ${
                     policy.risk === 'Critical' ? 'bg-red-500/10 text-red-500 border border-red-500/30' : 
                     policy.risk === 'High' ? 'bg-orange-500/10 text-orange-500 border border-orange-500/30' :
                     'bg-yellow-500/10 text-yellow-500 border border-yellow-500/30'
                   }`}>{policy.risk}</span>
                 </div>
                 
                 <p className="text-sm text-gray-300 italic mb-4 leading-relaxed font-serif pl-2 border-l-2 border-[#1e293b]">"{policy.text}"</p>
                 
                 <div className="flex justify-between items-center text-xs pt-3 border-t border-[#1e293b]/50">
                    <span className="text-gray-500 flex items-center gap-1">Matches: <span className="font-bold text-white font-mono">{policy.matchCount}</span></span>
                    <span className="text-gray-500 flex items-center gap-1">Last seen: <span className="font-mono">{policy.lastMatched}</span></span>
                 </div>
               </div>
             ))}
           </div>
        </div>

      </div>
    </div>
  );
}
