import React, { useState } from "react";
import { EnhancedFeed } from "@/components/EnhancedFeed";
import { EnhancedExplore } from "@/components/EnhancedExplore";
import { Shop } from "@/components/Shop";
import Community from "@/pages/Community";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Toaster } from "@/components/ui/sonner";

function App() {
  const [currentScreen, setCurrentScreen] = useState<'feed' | 'explore' | 'shop' | 'community'>('feed');

  return (
    <ErrorBoundary>
      <div className="relative min-h-screen w-full overflow-x-hidden">
        {currentScreen === 'feed' && <EnhancedFeed onNavigate={setCurrentScreen} />}
        {currentScreen === 'explore' && <EnhancedExplore onNavigate={setCurrentScreen} />}
        {currentScreen === 'shop' && <Shop onNavigate={setCurrentScreen} />}
        {currentScreen === 'community' && <Community />}
        <Toaster />
      </div>
    </ErrorBoundary>
  );
}

export default App;
