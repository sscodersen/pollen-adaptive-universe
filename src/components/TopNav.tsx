
import React, { useState } from 'react';
import { Bell, Settings, User, Search, Sparkles } from 'lucide-react';

export const TopNav = () => {
  const [searchFocused, setSearchFocused] = useState(false);

  return (
    <header className="h-16 bg-slate-800/40 backdrop-blur-xl border-b border-slate-700/50 flex items-center justify-between px-6 relative z-10">
      <div className="flex items-center space-x-4">
        <div className="relative">
          <Search className={`absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 transition-colors ${
            searchFocused ? 'text-cyan-400' : 'text-slate-400'
          }`} />
          <input
            type="text"
            placeholder="Ask Pollen anything..."
            onFocus={() => setSearchFocused(true)}
            onBlur={() => setSearchFocused(false)}
            className={`bg-slate-700/50 border rounded-lg pl-10 pr-4 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 transition-all w-80 ${
              searchFocused ? 'border-cyan-500/50 bg-slate-700/70' : 'border-slate-600/50'
            }`}
          />
        </div>
        
        <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-full border border-cyan-500/20">
          <Sparkles className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-medium text-cyan-300">AI Active</span>
        </div>
      </div>
      
      <div className="flex items-center space-x-3">
        <button className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors relative">
          <Bell className="w-5 h-5 text-slate-300" />
          <div className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></div>
        </button>
        
        <button className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
          <Settings className="w-5 h-5 text-slate-300" />
        </button>
        
        <button className="flex items-center space-x-2 p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
          <div className="w-7 h-7 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full flex items-center justify-center ring-2 ring-cyan-400/20">
            <User className="w-4 h-4 text-white" />
          </div>
          <span className="text-sm text-slate-300 font-medium hidden sm:block">You</span>
        </button>
      </div>
    </header>
  );
};
