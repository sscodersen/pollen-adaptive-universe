
import React, { useState, useEffect } from 'react';
import { ArrowLeft, Menu, Users, ShoppingBag, Play, Search, Bot, Globe, BarChart3, Target, Briefcase, TrendingUp, Zap } from 'lucide-react';
import { UnifiedHeader } from '../components/UnifiedHeader';
import { IntelligentActivityFeed } from '../components/IntelligentActivityFeed';
import { SystemMetricsDashboard } from '../components/SystemMetricsDashboard';
import { CrossDomainInsights } from '../components/CrossDomainInsights';
import { SocialFeed } from '../components/SocialFeed';
import { EntertainmentHub } from '../components/EntertainmentHub';
import { NewsEngine } from '../components/NewsEngine';
import { TaskAutomation } from '../components/TaskAutomation';
import { ShopHub } from '../components/ShopHub';
import { AnalyticsDashboard } from '../components/AnalyticsDashboard';
import { pollenAI } from '../services/pollenAI';

const NewPlayground = () => {
  const [activeTab, setActiveTab] = useState('Intelligence');
  const [isGenerating, setIsGenerating] = useState(false);
  const [aiStatus, setAiStatus] = useState('ready');
  const [platformMetrics, setPlatformMetrics] = useState({
    crossDomainConnections: 127,
    significanceScore: 8.9,
    globalUsers: 24891,
    aiOptimizations: 156,
    intelligenceSynergy: 94.3
  });

  const tabs = [
    { id: 'Intelligence', name: 'Intelligence Hub', icon: Bot, description: 'AI insights & cross-domain intelligence', color: 'from-purple-500 to-pink-500' },
    { id: 'Metrics', name: 'System Metrics', icon: BarChart3, description: 'Real-time platform analytics', color: 'from-cyan-500 to-blue-500' },
    { id: 'Social', name: 'Social Intelligence', icon: Users, description: 'AI-curated social insights', color: 'from-orange-500 to-red-500' },
    { id: 'Entertainment', name: 'Content Studio', icon: Play, description: 'AI-generated entertainment', color: 'from-purple-500 to-pink-500' },
    { id: 'Search', name: 'News Intelligence', icon: Search, description: 'Real-time news analysis', color: 'from-cyan-500 to-blue-500' },
    { id: 'Shop', name: 'Smart Commerce', icon: ShoppingBag, description: 'Intelligent shopping assistant', color: 'from-green-500 to-emerald-500' },
    { id: 'Automation', name: 'Task Automation', icon: Target, description: 'Automated workflow intelligence', color: 'from-violet-500 to-purple-500' },
    { id: 'Analytics', name: 'Deep Analytics', icon: BarChart3, description: 'Advanced insights dashboard', color: 'from-blue-500 to-indigo-500' }
  ];

  useEffect(() => {
    initializePlatform();
    
    const statusInterval = setInterval(() => {
      updateSystemStatus();
      updatePlatformMetrics();
    }, 5000);

    return () => {
      clearInterval(statusInterval);
    };
  }, []);

  const updatePlatformMetrics = () => {
    setPlatformMetrics(prev => ({
      crossDomainConnections: prev.crossDomainConnections + Math.floor(Math.random() * 3),
      significanceScore: Math.min(10, Math.max(8, prev.significanceScore + (Math.random() * 0.2 - 0.1))),
      globalUsers: prev.globalUsers + Math.floor(Math.random() * 20 - 10),
      aiOptimizations: prev.aiOptimizations + Math.floor(Math.random() * 5),
      intelligenceSynergy: Math.min(99.9, Math.max(90, prev.intelligenceSynergy + (Math.random() * 0.3 - 0.15)))
    }));
  };

  const updateSystemStatus = () => {
    const stats = pollenAI.getMemoryStats();
    setAiStatus(stats.isLearning ? 'learning' : 'ready');
  };

  const initializePlatform = () => {
    setIsGenerating(true);
    setTimeout(() => {
      setIsGenerating(false);
    }, 2000);
  };

  const renderTabContent = () => {
    const commonProps = { isGenerating };
    
    switch (activeTab) {
      case 'Intelligence':
        return (
          <div className="space-y-8">
            <IntelligentActivityFeed />
            <CrossDomainInsights />
          </div>
        );
      case 'Metrics':
        return <SystemMetricsDashboard />;
      case 'Social':
        return <SocialFeed {...commonProps} />;
      case 'Entertainment':
        return <EntertainmentHub {...commonProps} />;
      case 'Search':
        return <NewsEngine {...commonProps} />;
      case 'Shop':
        return <ShopHub />;
      case 'Automation':
        return <TaskAutomation {...commonProps} />;
      case 'Analytics':
        return <AnalyticsDashboard />;
      default:
        return (
          <div className="space-y-8">
            <IntelligentActivityFeed />
            <CrossDomainInsights />
          </div>
        );
    }
  };

  const activeTabInfo = tabs.find(tab => tab.id === activeTab);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-gray-950 to-blue-1000 text-white">
      {/* Enhanced Unified Header */}
      <UnifiedHeader 
        title="Pollen Intelligence Platform"
        subtitle="Operating System for Digital Life • Multi-Domain AI • Real-Time Cross-Domain Intelligence"
        activeFeatures={['ai', 'learning', 'optimized']}
      />

      {/* Enhanced Navigation with Better Visual Hierarchy */}
      <div className="border-b border-gray-800/60 bg-gray-900/40 backdrop-blur-sm sticky top-0 z-40">
        <div className="flex space-x-2 px-6 py-4 overflow-x-auto scrollbar-thin">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`group flex items-center space-x-3 px-6 py-4 rounded-xl transition-all duration-300 whitespace-nowrap font-medium min-w-fit relative ${
                  isActive
                    ? `bg-gradient-to-r ${tab.color} bg-opacity-20 border border-opacity-50 text-white shadow-lg backdrop-blur-sm scale-105`
                    : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/50 border border-transparent hover:scale-102'
                }`}
                style={isActive ? { 
                  borderColor: tab.color.includes('purple') ? '#a855f7' : 
                              tab.color.includes('cyan') ? '#06b6d4' : 
                              tab.color.includes('orange') ? '#ea580c' : 
                              tab.color.includes('green') ? '#059669' : '#8b5cf6'
                } : {}}
              >
                <tab.icon className={`w-5 h-5 transition-colors ${
                  isActive ? 'text-white' : 'group-hover:text-white'
                }`} />
                <div className="text-left">
                  <div className="font-semibold text-sm">{tab.name}</div>
                  <div className="text-xs opacity-70">{tab.description}</div>
                </div>
                {isActive && (
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse ml-2"></div>
                )}
              </button>
            );
          })}
        </div>

        {/* Enhanced Platform Intelligence Bar */}
        <div className="px-6 pb-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <Globe className="w-3 h-3 text-cyan-400 animate-pulse" />
                <span>{platformMetrics.crossDomainConnections} cross-domain connections</span>
              </div>
              <div className="flex items-center space-x-2">
                <TrendingUp className="w-3 h-3 text-green-400" />
                <span>Significance: {platformMetrics.significanceScore.toFixed(1)}/10</span>
              </div>
              <div className="flex items-center space-x-2">
                <Users className="w-3 h-3 text-purple-400" />
                <span>{platformMetrics.globalUsers.toLocaleString()} global users</span>
              </div>
              <div className="flex items-center space-x-2">
                <Zap className="w-3 h-3 text-orange-400 animate-pulse" />
                <span>{platformMetrics.aiOptimizations} AI optimizations</span>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="px-3 py-1 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-full text-cyan-300 border border-cyan-500/20 text-xs font-medium">
                Intelligence Synergy: {platformMetrics.intelligenceSynergy.toFixed(1)}%
              </div>
              <div className="px-3 py-1 bg-gradient-to-r from-green-500/10 to-emerald-500/10 rounded-full text-green-300 border border-green-500/20 text-xs font-medium">
                Operating System for Digital Life • Production Ready
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Content Area with Better Spacing and Loading States */}
      <div className="relative">
        {/* Active Tab Indicator */}
        {activeTabInfo && (
          <div className={`absolute top-0 left-0 right-0 bg-gradient-to-r ${activeTabInfo.color} opacity-10 h-px`}></div>
        )}
        
        <div className="p-6 min-h-screen">
          {isGenerating ? (
            <div className="flex items-center justify-center h-64">
              <div className="text-center">
                <div className="w-16 h-16 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl flex items-center justify-center mx-auto mb-4 animate-pulse">
                  <Bot className="w-8 h-8 text-white animate-spin" />
                </div>
                <div className="text-lg font-semibold text-white mb-2">Initializing Intelligence Platform</div>
                <div className="text-sm text-slate-400">Loading cross-domain AI systems...</div>
              </div>
            </div>
          ) : (
            <div className="animate-fade-in">
              {renderTabContent()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default NewPlayground;
