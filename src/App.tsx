import React, { useState } from "react";
import { EnhancedFeed } from "@/components/EnhancedFeed";
import { EnhancedExplore } from "@/components/EnhancedExplore";
import { Shop } from "@/components/Shop";
import Community from "@/pages/Community";
import HealthResearch from "@/pages/HealthResearch";
import AIEthicsForum from "@/pages/AIEthicsForum";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { Toaster } from "@/components/ui/sonner";
import { Button } from "@/components/ui/button";
import { Home, Compass, ShoppingBag, Users, Activity, Shield } from "lucide-react";

function App() {
  const [currentScreen, setCurrentScreen] = useState<'feed' | 'explore' | 'shop' | 'community' | 'health' | 'ethics'>('feed');

  return (
    <ErrorBoundary>
      <div className="relative min-h-screen w-full overflow-x-hidden">
        {/* Top Navigation Bar */}
        <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-lg border-b border-gray-200">
          <div className="container mx-auto px-4">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  Pollen Universe
                </h1>
              </div>
              <div className="flex items-center space-x-2">
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
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <div className="pt-16">
          {currentScreen === 'feed' && <EnhancedFeed onNavigate={setCurrentScreen} />}
          {currentScreen === 'explore' && <EnhancedExplore onNavigate={setCurrentScreen} />}
          {currentScreen === 'shop' && <Shop onNavigate={setCurrentScreen} />}
          {currentScreen === 'community' && <Community />}
          {currentScreen === 'health' && <HealthResearch />}
          {currentScreen === 'ethics' && <AIEthicsForum />}
        </div>

        <Toaster />
      </div>
    </ErrorBoundary>
  );
}

export default App;
