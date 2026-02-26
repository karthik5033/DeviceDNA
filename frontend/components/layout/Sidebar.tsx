'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from '@/lib/utils';
import { 
  LayoutDashboard, 
  Network, 
  ShieldAlert, 
  BrainCircuit,
  PlaySquare,
  ActivitySquare
} from 'lucide-react';

const routes = [
  { name: 'Overview', path: '/dashboard', icon: LayoutDashboard },
  { name: 'Network Topology', path: '/dashboard/topology', icon: Network },
  { name: 'Alert Queue', path: '/dashboard/alerts', icon: ShieldAlert },
  { name: 'NLP Policies', path: '/dashboard/policies', icon: BrainCircuit },
  { name: 'Attack Replay', path: '/dashboard/replay', icon: PlaySquare },
  { name: 'Predictive Risk', path: '/dashboard/predict', icon: ActivitySquare },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-64 border-r border-[#1e293b] bg-[#0b101e] hidden md:flex flex-col">
      <div className="h-16 flex items-center px-6 border-b border-[#1e293b]">
        <Link href="/" className="font-bold text-xl tracking-tighter">
          Device<span className="text-[#3edcff]">DNA</span><span className="text-gray-500 font-mono text-xs ml-2">SOC</span>
        </Link>
      </div>
      
      <nav className="flex-1 py-6 px-4 space-y-2">
        {routes.map((route) => {
          const isActive = pathname === route.path || (pathname.startsWith(route.path) && route.path !== '/dashboard');
          const Icon = route.icon;
          
          return (
            <Link 
              key={route.path} 
              href={route.path}
              className={cn(
                "flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors",
                isActive 
                  ? "bg-[#1e293b]/50 text-white border border-[#334155]" 
                  : "text-gray-400 hover:text-white hover:bg-[#1e293b]/30"
              )}
            >
              <Icon size={18} className={isActive ? "text-[#3edcff]" : "text-gray-500"} />
              {route.name}
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t border-[#1e293b]">
        <div className="bg-[#111827] border border-[#1e293b] rounded-md p-3">
          <div className="text-xs text-gray-400 uppercase tracking-wider font-semibold mb-2">Engine Status</div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-green-500 flex items-center gap-1.5"><div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"/> Tracking</span>
            <span className="font-mono text-gray-500">50 devices</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
