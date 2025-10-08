import React, { useState } from "react";
import { Feed } from "@/components/Feed";
import { Explore } from "@/components/Explore";
import { Shop } from "@/components/Shop";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Toaster } from "@/components/ui/sonner";

function App() {
  const [currentScreen, setCurrentScreen] = useState<'feed' | 'explore' | 'shop'>('feed');

  return (
    <ErrorBoundary>
      <div className="relative min-h-screen w-full overflow-x-hidden">
        {currentScreen === 'feed' && <Feed onNavigate={setCurrentScreen} />}
        {currentScreen === 'explore' && <Explore onNavigate={setCurrentScreen} />}
        {currentScreen === 'shop' && <Shop onNavigate={setCurrentScreen} />}
        <Toaster />
      </div>
    </ErrorBoundary>
  );
}

export default App;
