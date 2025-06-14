
import React from 'react';
import { SidebarProvider } from './ui/sidebar';
import { AppSidebar } from './AppSidebar';
import { PlaygroundProvider } from '../contexts/PlaygroundContext';

export const AppLayout = ({ children }: { children: React.ReactNode }) => {
  return (
    <PlaygroundProvider>
      <SidebarProvider>
        <div className="min-h-screen w-full bg-slate-950 text-white flex has-[[data-sidebar-hidden=true]]:justify-center">
          <AppSidebar />
          <main className="flex-1 flex flex-col overflow-hidden">
            {children}
          </main>
        </div>
      </SidebarProvider>
    </PlaygroundProvider>
  );
};
