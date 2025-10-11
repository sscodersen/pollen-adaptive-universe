import React, { useState } from "react";
import { EnhancedFeed } from "@/components/EnhancedFeed";
import { EnhancedExplore } from "@/components/EnhancedExplore";
import { Shop } from "@/components/Shop";
import Community from "@/pages/Community";
import HealthResearch from "@/pages/HealthResearch";
import AIEthicsForum from "@/pages/AIEthicsForum";
import AdminDashboard from "@/pages/AdminDashboard";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Toaster } from "@/components/ui/sonner";
import { Button } from "@/components/ui/button";
import { WelcomeOnboarding } from "@/components/WelcomeOnboarding";
import { HelpSupport } from "@/components/HelpSupport";
import { Home, Compass, ShoppingBag, Users, Activity, Shield, Moon, Sun, HelpCircle, Settings } from "lucide-react";
import { useTheme } from "next-themes";

function App() {
  const [currentScreen, setCurrentScreen] = useState<'feed' | 'explore' | 'shop' | 'community' | 'health' | 'ethics' | 'help' | 'admin'>('feed');
  const { theme, setTheme } = useTheme();

  return (
    <ErrorBoundary>
      <WelcomeOnboarding />
      <div className="relative min-h-screen w-full overflow-x-hidden">
        {/* Top Navigation Bar */}
        <nav className="fixed top-0 left-0 right-0 z-50 bg-white/95 dark:bg-gray-900/95 backdrop-blur-lg border-b border-gray-200 dark:border-gray-800">
          <div className="container mx-auto px-4">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold text-black dark:text-white">
                  Pollen Universe
                </h1>
              </div>
              <div className="flex items-center space-x-2">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
                  className="mr-2"
                >
                  <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
                  <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
                  <span className="sr-only">Toggle theme</span>
                </Button>
                <Button
                  variant={currentScreen === 'feed' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setCurrentScreen('feed')}
                >
                  <Home className="w-4 h-4 mr-2" />
                  Feed
                </Button>
                <Button
                  variant={currentScreen === 'explore' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setCurrentScreen('explore')}
                >
                  <Compass className="w-4 h-4 mr-2" />
                  Explore
                </Button>
                <Button
                  variant={currentScreen === 'shop' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setCurrentScreen('shop')}
                >
                  <ShoppingBag className="w-4 h-4 mr-2" />
                  Shop
                </Button>
                <Button
                  variant={currentScreen === 'community' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setCurrentScreen('community')}
                >
                  <Users className="w-4 h-4 mr-2" />
                  Community
                </Button>
                <Button
                  variant={currentScreen === 'health' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setCurrentScreen('health')}
                >
                  <Activity className="w-4 h-4 mr-2" />
                  Health
                </Button>
                <Button
                  variant={currentScreen === 'ethics' ? 'default' : 'ghost'}
                  size="sm"
                  onClick={() => setCurrentScreen('ethics')}
                >
                  <Shield className="w-4 h-4 mr-2" />
                  Ethics
                </Button>
                <Button
                  variant={currentScreen === 'help' ? 'default' : 'ghost'}
                  size="icon"
                  onClick={() => setCurrentScreen('help')}
                  title="Help & Support"
                >
                  <HelpCircle className="w-4 h-4" />
                </Button>
                <Button
                  variant={currentScreen === 'admin' ? 'default' : 'ghost'}
                  size="icon"
                  onClick={() => setCurrentScreen('admin')}
                  title="Admin Dashboard"
                >
                  <Settings className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <div className="pt-16 page-transition">
          {currentScreen === 'feed' && <EnhancedFeed onNavigate={setCurrentScreen} />}
          {currentScreen === 'explore' && <EnhancedExplore onNavigate={setCurrentScreen} />}
          {currentScreen === 'shop' && <Shop onNavigate={setCurrentScreen} />}
          {currentScreen === 'community' && <Community />}
          {currentScreen === 'health' && <HealthResearch />}
          {currentScreen === 'ethics' && <AIEthicsForum />}
          {currentScreen === 'help' && <HelpSupport />}
          {currentScreen === 'admin' && <AdminDashboard />}
        </div>

        <Toaster />
      </div>
    </ErrorBoundary>
  );
}

export default App;
