
import React from 'react';
import { AppSidebar } from './AppSidebar';

export const AppLayout = ({ children }: { children: React.ReactNode }) => {
  return (
    <div className="min-h-screen bg-gray-950 text-white flex">
      <AppSidebar />
      <main className="flex-1 overflow-y-auto">
        {children}
      </main>
    </div>
  );
};
