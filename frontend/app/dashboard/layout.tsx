import ResizableLayout from '@/components/layout/ResizableLayout';

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <ResizableLayout>{children}</ResizableLayout>;
}
