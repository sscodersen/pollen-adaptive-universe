import React, { useState } from "react";
import { SearchCanvas } from "@/components/search/SearchCanvas";
import { useSearchOrchestrator } from "@/hooks/useSearchOrchestrator";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Toaster } from "@/components/ui/sonner";
import { Button } from "@/components/ui/button";
import { WelcomeOnboarding } from "@/components/WelcomeOnboarding";
import { GlobalLoadingIndicator } from "@/components/GlobalLoadingIndicator";
import FeedbackSystem from "@/components/FeedbackSystem";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";

function App() {
  const { theme, setTheme } = useTheme();
  const [sessionId] = useState(`session-${Date.now()}-${Math.random().toString(36).substring(7)}`);
  
  const {
    searchQuery,
    setSearchQuery,
    results,
    isLoading,
    executeSearch
  } = useSearchOrchestrator(sessionId);

  return (
    <ErrorBoundary>
      <WelcomeOnboarding />
      <div className="relative min-h-screen w-full overflow-x-hidden">
        {/* Minimal Fixed Header */}
        <div className="fixed top-4 right-4 z-50 flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="rounded-full bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg shadow-lg hover:shadow-xl transition-all"
          >
            <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
          </Button>
        </div>

        {/* Main Search Canvas */}
        <SearchCanvas
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          onSearch={executeSearch}
          results={results}
          isLoading={isLoading}
        />

        {/* Global Components */}
        <GlobalLoadingIndicator />
        <Toaster />
        <FeedbackSystem />
      </div>
    </ErrorBoundary>
  );
}

export default App;
