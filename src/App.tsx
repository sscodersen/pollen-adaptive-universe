
import React from "react";
import { TopNav } from "@/components/TopNav";
import { MainTabs } from "@/components/MainTabs";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { EnhancedAppProvider } from "@/contexts/EnhancedAppContext";
import { Toaster } from "@/components/ui/sonner";

function App() {
  return (
    <ErrorBoundary>
      <EnhancedAppProvider>
        <div className="min-h-screen w-full flex flex-col font-sans bg-background">
          <TopNav />
          <MainTabs />
          <Toaster />
        </div>
      </EnhancedAppProvider>
    </ErrorBoundary>
  );
}

export default App;
