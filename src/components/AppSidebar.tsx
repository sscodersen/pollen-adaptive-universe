
import React from 'react';
import { usePlayground } from '../contexts/PlaygroundContext';
import { ArrowLeft, Menu, Settings, Users, Briefcase, Bot } from 'lucide-react';

export const AppSidebar = () => {
  const { tabs, activeTab, setActiveTab } = usePlayground();

  return (
    <div className="w-80 bg-gray-950/60 backdrop-blur-xl border-r border-white/10 flex flex-col flex-shrink-0">
      <div className="flex items-center justify-between p-6 border-b border-white/10 h-24">
        <div className="flex items-center space-x-4">
          <div className="w-10 h-10 bg-gradient-to-br from-cyan-500 to-purple-600 rounded-xl flex items-center justify-center shadow-lg">
            <Bot className="w-6 h-6 text-white"/>
          </div>
          <div>
            <h1 className="text-xl font-bold">Pollen</h1>
            <p className="text-gray-400 text-xs">Adaptive Intelligence</p>
          </div>
        </div>
        <button className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors">
          <Menu className="w-5 h-5" />
        </button>
      </div>
      <nav className="flex-1 px-4 py-6 space-y-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 text-left ${
              activeTab === tab.id
                ? 'bg-cyan-400/10 text-cyan-300 shadow-md'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`}
          >
            <tab.icon className="w-5 h-5" />
            <span className="font-medium">{tab.name}</span>
          </button>
        ))}
      </nav>
      <div className="p-6 border-t border-white/10">
        <button className="w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-colors">
          <ArrowLeft className="w-5 h-5" />
          <span className="font-medium">Sign Out</span>
        </button>
      </div>
    </div>
  );
};
