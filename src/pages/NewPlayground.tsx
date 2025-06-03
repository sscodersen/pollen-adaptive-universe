
import React, { useState, useEffect } from 'react';
import { Layout } from '../components/Layout';
import { PollenChat } from '../components/PollenChat';
import { PollenMemoryPanel } from '../components/PollenMemoryPanel';
import { FileUpload } from '../components/FileUpload';
import { SocialFeed } from '../components/SocialFeed';
import { NewsEngine } from '../components/NewsEngine';
import { EntertainmentHub } from '../components/EntertainmentHub';
import { Brain, Code, Sparkles, BarChart3, Globe, Film, Users, Search } from 'lucide-react';

const NewPlayground = () => {
  const [activeMode, setActiveMode] = useState('social');
  const [isGenerating, setIsGenerating] = useState(true);

  const handleFileProcessed = (file: any) => {
    console.log('File processed:', file);
  };

  const modes = [
    { id: 'social', name: 'Social Feed', icon: Users, component: SocialFeed },
    { id: 'news', name: 'News Engine', icon: Globe, component: NewsEngine },
    { id: 'entertainment', name: 'Entertainment', icon: Film, component: EntertainmentHub },
    { id: 'chat', name: 'Chat & Reasoning', icon: Brain, component: null },
    { id: 'code', name: 'Code Assistant', icon: Code, component: null },
    { id: 'creative', name: 'Creative Studio', icon: Sparkles, component: null },
    { id: 'analysis', name: 'Analysis & Insights', icon: BarChart3, component: null }
  ];

  const ActiveComponent = modes.find(m => m.id === activeMode)?.component;

  return (
    <Layout>
      <div className="h-full flex bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
        {/* Left Sidebar */}
        <div className="w-80 p-6 space-y-6 border-r border-white/10 bg-black/20 backdrop-blur-xl">
          <div>
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-12 h-12 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-xl flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Pollen</h2>
                <p className="text-sm text-white/60">Self-Evolving AI Platform</p>
              </div>
            </div>
            
            {/* Mode Navigation */}
            <div className="space-y-2 mb-6">
              {modes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setActiveMode(mode.id)}
                  className={`w-full flex items-center space-x-3 p-3 rounded-xl transition-all ${
                    activeMode === mode.id
                      ? 'bg-white/20 border border-white/20 text-white'
                      : 'bg-white/5 border border-white/10 text-white/60 hover:bg-white/10 hover:text-white'
                  }`}
                >
                  <mode.icon className="w-5 h-5" />
                  <span className="font-medium">{mode.name}</span>
                  {(mode.id === 'social' || mode.id === 'news' || mode.id === 'entertainment') && (
                    <div className="ml-auto w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Memory Panel */}
          <PollenMemoryPanel />

          {/* File Upload */}
          <div>
            <h3 className="text-sm font-medium text-white/80 mb-3">File Analysis</h3>
            <FileUpload onFileProcessed={handleFileProcessed} />
          </div>

          {/* Generation Status */}
          <div className="bg-white/10 rounded-xl p-4 border border-white/10">
            <div className="flex items-center space-x-2 mb-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-sm font-medium text-white">Continuous Generation</span>
            </div>
            <p className="text-xs text-white/60">
              Pollen is autonomously generating content, news, and entertainment based on reasoning patterns.
            </p>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col">
          {ActiveComponent ? (
            <ActiveComponent />
          ) : (
            <div className="flex-1 p-6">
              <PollenChat 
                mode={activeMode} 
                onModeChange={setActiveMode}
              />
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
};

export default NewPlayground;
