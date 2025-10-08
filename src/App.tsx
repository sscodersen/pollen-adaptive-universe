import React, { useState } from "react";
import { Home } from "@/components/Home";
import { Search } from "@/components/Search";
import { Collections } from "@/components/Collections";
import { Shopping } from "@/components/Shopping";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Toaster } from "@/components/ui/sonner";

function App() {
  const [currentScreen, setCurrentScreen] = useState<'home' | 'search' | 'collections' | 'shopping'>('home');

  return (
    <ErrorBoundary>
      <div className="relative min-h-screen w-full overflow-x-hidden">
        {currentScreen === 'home' && <Home onNavigate={setCurrentScreen} />}
        {currentScreen === 'search' && <Search onNavigate={setCurrentScreen} />}
        {currentScreen === 'collections' && <Collections onNavigate={setCurrentScreen} />}
        {currentScreen === 'shopping' && <Shopping onNavigate={setCurrentScreen} />}
        <Toaster />
      </div>
    </ErrorBoundary>
  );
}

export default App;
