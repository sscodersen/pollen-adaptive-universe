
import React from 'react';
import { usePlayground } from '../contexts/PlaygroundContext';
import { ArrowLeft, Bot } from 'lucide-react';

export const AppSidebar = () => {
  const { tabs, activeTab, setActiveTab } = usePlayground();

  return (
    <div className="w-72 bg-black/25 backdrop-blur-2xl border-r border-white/10 flex flex-col flex-shrink-0">
      <div className="flex items-center space-x-4 p-6 border-b border-white/10 h-24">
        <div className="w-11 h-11 bg-gradient-to-br from-cyan-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg flex-shrink-0">
          <Bot className="w-6 h-6 text-white"/>
        </div>
        <div>
          <h1 className="text-xl font-bold">Pollen</h1>
          <p className="text-gray-400 text-xs">Adaptive Intelligence</p>
        </div>
      </div>
      <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`w-full flex items-center space-x-4 px-4 py-3 rounded-lg transition-all duration-200 text-left ${
              activeTab === tab.id
                ? 'bg-cyan-400/20 text-white font-semibold'
                : 'text-gray-300 hover:text-white hover:bg-white/10'
            }`}
          >
            <tab.icon className="w-5 h-5 flex-shrink-0" />
            <span className="flex-1 truncate">{tab.name}</span>
          </button>
        ))}
      </nav>
      <div className="p-4 mt-auto border-t border-white/10">
        <button className="w-full flex items-center space-x-4 px-4 py-3 rounded-lg text-gray-300 hover:text-white hover:bg-white/10 transition-colors">
          <ArrowLeft className="w-5 h-5" />
          <span className="font-medium">Sign Out</span>
        </button>
      </div>
    </div>
  );
};
