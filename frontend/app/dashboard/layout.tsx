import Sidebar from '@/components/layout/Sidebar';
import Header from '@/components/layout/Header';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-screen overflow-hidden bg-[#070b14] text-white">
      {/* Sidebar - Fix width and persist */}
      <Sidebar />

      {/* Main Content Area */}
      <div className="flex flex-col flex-1 overflow-hidden relative">
        <Header />
        
        {/* Scrollable Main Area */}
        <main className="flex-1 overflow-y-auto p-6 scroll-smooth">
          {children}
        </main>
      </div>
    </div>
  );
}
