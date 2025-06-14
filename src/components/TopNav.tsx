
import React from 'react';
import { Bell, Settings, User, Search } from 'lucide-react';

export const TopNav = () => {
  return (
    <header className="h-16 bg-slate-800/30 backdrop-blur-xl border-b border-slate-700/50 flex items-center justify-between px-6">
      <div className="flex items-center space-x-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
          <input
            type="text"
            placeholder="Ask Pollen anything..."
            className="bg-slate-700/50 border border-slate-600/50 rounded-lg pl-10 pr-4 py-2 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 w-80"
          />
        </div>
      </div>
      
      <div className="flex items-center space-x-4">
        <button className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
          <Bell className="w-5 h-5 text-slate-300" />
        </button>
        <button className="p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
          <Settings className="w-5 h-5 text-slate-300" />
        </button>
        <button className="flex items-center space-x-2 p-2 hover:bg-slate-700/50 rounded-lg transition-colors">
          <div className="w-6 h-6 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full flex items-center justify-center">
            <User className="w-4 h-4 text-white" />
          </div>
          <span className="text-sm text-slate-300">You</span>
        </button>
      </div>
    </header>
  );
};
