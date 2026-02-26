import { Bell, Search, User } from 'lucide-react';

export default function Header() {
  return (
    <header className="h-16 flex items-center justify-between px-6 border-b border-[#1e293b] bg-[#070b14]/90 backdrop-blur top-0 sticky z-50">
      
      {/* Search Bar */}
      <div className="relative w-64 md:w-96 text-gray-400">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2" size={18} />
        <input 
          type="text" 
          placeholder="Search devices, alerts, or queries (NLP)"
          className="w-full bg-[#111827] border border-[#1e293b] rounded-full py-1.5 pl-10 pr-4 text-sm text-white focus:outline-none focus:border-[#3edcff]/50 focus:ring-1 focus:ring-[#3edcff]/50 transition-all font-ui"
        />
      </div>

      {/* Toggles / User Actions */}
      <div className="flex items-center gap-4 text-gray-400">
        <button className="flex items-center gap-2 hover:text-white transition-colors relative">
          <Bell size={20} />
          <span className="absolute top-0 right-0 h-2.5 w-2.5 bg-red-500 rounded-full border border-[#070b14]"></span>
        </button>
        
        <div className="h-8 w-px bg-[#1e293b] mx-2"></div>
        
        <button className="flex items-center gap-2 hover:text-white transition-colors font-medium text-sm">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-[#3edcff] to-blue-600 flex items-center justify-center text-white">
            <User size={16} />
          </div>
          <span className="hidden lg:inline-block pr-2">Security Admin</span>
        </button>
      </div>

    </header>
  );
}
