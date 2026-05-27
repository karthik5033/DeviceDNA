'use client';

import Sidebar from '@/components/layout/Sidebar';
import Header from '@/components/layout/Header';
import { useState } from 'react';

export default function ResizableLayout({ children }: { children: React.ReactNode }) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
  };

  return (
    <div className="flex w-full h-screen overflow-hidden bg-[#070b14] text-white">
      {/* Left Sidebar (CSS Toggle) */}
      <div 
        className="transition-all duration-300 ease-in-out flex-shrink-0"
        style={{ width: isCollapsed ? '80px' : '260px' }}
      >
        <Sidebar isCollapsed={isCollapsed} toggleSidebar={toggleSidebar} />
      </div>
      
      {/* Main Content Area */}
      <div className="flex flex-col flex-1 h-full overflow-hidden relative">
        <Header toggleSidebar={toggleSidebar} isSidebarCollapsed={isCollapsed} />
        <main className="flex-1 overflow-y-auto p-6 scroll-smooth">
          {children}
        </main>
      </div>
    </div>
  );
}
