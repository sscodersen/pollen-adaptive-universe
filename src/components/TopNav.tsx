
import React, { useState } from 'react';
import { Bell, Settings, User, Sparkles, Menu } from 'lucide-react';
import { GlobalSearch } from './GlobalSearch';
import { useEnhancedApp } from '@/contexts/EnhancedAppContext';

export const TopNav = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { state } = useEnhancedApp();
  const currentUser = state.user;

  const handleSearchResult = (result: any) => {
    console.log('Search result selected:', result);
  };

  return (
    <header className="h-16 bg-slate-800/40 backdrop-blur-xl border-b border-slate-700/50 flex items-center justify-between px-4 md:px-6 relative z-10">
      <div className="flex items-center space-x-2 md:space-x-4 flex-1 max-w-3xl">
        <div className="hidden md:block flex-1">
          <GlobalSearch 
            onResultSelect={handleSearchResult}
            placeholder="Ask Pollen anything..."
          />
        </div>
        
        <button 
          className="md:hidden p-2 hover:bg-slate-700/50 rounded-lg transition-colors"
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
        >
          <Menu className="w-5 h-5 text-slate-300" />
        </button>
        
        <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-full border border-cyan-500/20">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-medium text-cyan-300">AI Active</span>
        </div>
      </div>
      
      <div className="flex items-center space-x-2 md:space-x-3">
        <button className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors relative">
          <Bell className="w-5 h-5 text-slate-300" />
          <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></div>
        </button>
        
        <button className="hidden md:block p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
          <Settings className="w-5 h-5 text-slate-300" />
        </button>
        
        <button className="flex items-center space-x-2 p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
          <div className="w-7 h-7 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full flex items-center justify-center ring-2 ring-cyan-400/20">
            <User className="w-4 h-4 text-white" />
          </div>
          <span className="text-sm text-slate-300 font-medium hidden sm:block">
            {currentUser?.username || 'You'}
          </span>
        </button>
      </div>

      {mobileMenuOpen && (
        <div className="absolute top-16 left-0 right-0 bg-slate-800/95 backdrop-blur-xl border-b border-slate-700/50 p-4 md:hidden z-50">
          <GlobalSearch 
            onResultSelect={(result) => {
              handleSearchResult(result);
              setMobileMenuOpen(false);
            }}
            placeholder="Ask Pollen anything..."
            autoFocus
          />
        </div>
      )}
    </header>
  );
};
