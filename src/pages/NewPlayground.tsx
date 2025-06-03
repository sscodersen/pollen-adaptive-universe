
import React, { useState, useEffect } from 'react';
import { Search, Sparkles, Play, Pause, Settings, User, Bell, Plus } from 'lucide-react';
import { SocialFeed } from '../components/SocialFeed';
import { NewsEngine } from '../components/NewsEngine';
import { EntertainmentHub } from '../components/EntertainmentHub';
import { PollenChat } from '../components/PollenChat';
import { pollenAI } from '../services/pollenAI';

const NewPlayground = () => {
  const [activeTab, setActiveTab] = useState('social');
  const [isGenerating, setIsGenerating] = useState(true);
  const [memoryStats, setMemoryStats] = useState<any>(null);

  useEffect(() => {
    const updateStats = () => {
      const stats = pollenAI.getMemoryStats();
      setMemoryStats(stats);
    };

    updateStats();
    const interval = setInterval(updateStats, 5000);
    return () => clearInterval(interval);
  }, []);

  const tabs = [
    { id: 'social', name: 'Social', component: SocialFeed },
    { id: 'news', name: 'News', component: NewsEngine },
    { id: 'entertainment', name: 'Entertainment', component: EntertainmentHub },
    { id: 'chat', name: 'Chat', component: null }
  ];

  const ActiveComponent = tabs.find(t => t.id === activeTab)?.component;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900">
      {/* Top Navigation */}
      <nav className="h-16 bg-gray-900/80 backdrop-blur-xl border-b border-gray-700/50 flex items-center justify-between px-6">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-400 to-purple-500 rounded-lg flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-white">Pollen</span>
            <span className="text-sm text-gray-400">AI Platform</span>
          </div>
        </div>

        {/* Search Bar */}
        <div className="flex-1 max-w-2xl mx-8">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Ask Pollen anything..."
              className="w-full bg-gray-800/50 border border-gray-600/50 rounded-xl pl-12 pr-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50"
            />
          </div>
        </div>

        {/* Right Controls */}
        <div className="flex items-center space-x-4">
          <button
            onClick={() => setIsGenerating(!isGenerating)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
              isGenerating
                ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                : 'bg-gray-700/50 text-gray-400 border border-gray-600/50'
            }`}
          >
            {isGenerating ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            <span className="text-sm">{isGenerating ? 'Generating' : 'Paused'}</span>
          </button>
          
          <button className="p-2 text-gray-400 hover:text-white transition-colors">
            <Bell className="w-5 h-5" />
          </button>
          
          <button className="p-2 text-gray-400 hover:text-white transition-colors">
            <Settings className="w-5 h-5" />
          </button>
          
          <button className="flex items-center space-x-2 p-2 text-gray-400 hover:text-white transition-colors">
            <User className="w-5 h-5" />
            <span className="text-sm">You</span>
          </button>
        </div>
      </nav>

      <div className="flex h-[calc(100vh-4rem)]">
        {/* Sidebar */}
        <div className="w-80 bg-gray-900/50 backdrop-blur-xl border-r border-gray-700/50 flex flex-col">
          {/* Tab Navigation */}
          <div className="p-6">
            <div className="flex space-x-1 bg-gray-800/50 rounded-xl p-1">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex-1 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                    activeTab === tab.id
                      ? 'bg-blue-500 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                  }`}
                >
                  {tab.name}
                </button>
              ))}
            </div>
          </div>

          {/* Memory & Stats */}
          <div className="flex-1 p-6 space-y-6">
            <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
              <h3 className="text-lg font-semibold text-white mb-3">Memory</h3>
              {memoryStats && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Short-term</span>
                    <span className="text-white">{memoryStats.shortTermSize}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Patterns</span>
                    <span className="text-white">{memoryStats.longTermPatterns}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Reasoning Tasks</span>
                    <span className="text-white">{memoryStats.reasoningTasks || 0}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">High Reward</span>
                    <span className="text-green-400">{memoryStats.highRewardTasks || 0}</span>
                  </div>
                </div>
              )}
            </div>

            <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl p-4 border border-blue-500/20">
              <div className="flex items-center space-x-2 mb-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium text-green-400">Learning Active</span>
              </div>
              <p className="text-xs text-gray-300">
                Pollen is continuously evolving through autonomous reasoning and pattern recognition.
              </p>
            </div>

            <button className="w-full flex items-center space-x-2 px-4 py-3 bg-blue-500/20 border border-blue-500/30 rounded-xl text-blue-400 hover:bg-blue-500/30 transition-colors">
              <Plus className="w-4 h-4" />
              <span className="text-sm font-medium">New Session</span>
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col bg-gray-900/30">
          {ActiveComponent ? (
            <ActiveComponent isGenerating={isGenerating} />
          ) : (
            <div className="flex-1 p-6">
              <PollenChat 
                mode={activeTab} 
                onModeChange={setActiveTab}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default NewPlayground;
