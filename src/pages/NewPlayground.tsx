
import React, { useState } from 'react';
import { ArrowLeft, Menu, Users, ShoppingBag, Play, Search, Bot, Globe, BarChart3, Target, Briefcase, TrendingUp, Zap } from 'lucide-react';
import { UnifiedHeader } from '../components/UnifiedHeader';
import { IntelligentActivityFeed } from '../components/IntelligentActivityFeed';
import { IntelligenceDashboard } from '../components/IntelligenceDashboard';
import { CrossDomainInsights } from '../components/CrossDomainInsights';
import { SocialFeed } from '../components/SocialFeed';
import { EntertainmentHub } from '../components/EntertainmentHub';
import { NewsEngine } from '../components/NewsEngine';
import { TaskAutomation } from '../components/TaskAutomation';
import { ShopHub } from '../components/ShopHub';
import { AnalyticsDashboard } from '../components/AnalyticsDashboard';
import { usePlatformState } from '../hooks/usePlatformState';
import { platformTheme } from '../lib/platformTheme';

const NewPlayground = () => {
  const [activeTab, setActiveTab] = useState('Intelligence');
  const { metrics, aiStatus, isGenerating, updateGeneratingState } = usePlatformState();

  const tabs = [
    { id: 'Intelligence', name: 'Intelligence Hub', icon: Bot, description: 'AI insights & cross-domain intelligence', color: platformTheme.colors.primary.purple },
    { id: 'Dashboard', name: 'Intelligence Dashboard', icon: BarChart3, description: 'Real-time AI metrics & system health', color: platformTheme.colors.primary.cyan },
    { id: 'Social', name: 'Social Intelligence', icon: Users, description: 'AI-curated social insights', color: platformTheme.colors.primary.orange },
    { id: 'Entertainment', name: 'Content Studio', icon: Play, description: 'AI-generated entertainment', color: platformTheme.colors.primary.purple },
    { id: 'Search', name: 'News Intelligence', icon: Search, description: 'Real-time news analysis', color: platformTheme.colors.primary.cyan },
    { id: 'Shop', name: 'Smart Commerce', icon: ShoppingBag, description: 'Intelligent shopping assistant', color: platformTheme.colors.primary.green },
    { id: 'Automation', name: 'Task Automation', icon: Target, description: 'Automated workflow intelligence', color: platformTheme.colors.primary.violet },
    { id: 'Analytics', name: 'Deep Analytics', icon: BarChart3, description: 'Advanced insights dashboard', color: platformTheme.colors.primary.cyan }
  ];

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
      case 'Dashboard':
        return <IntelligenceDashboard />;
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
      {/* Unified Header */}
      <UnifiedHeader 
        title="Pollen Intelligence Platform"
        subtitle="Operating System for Digital Life • Multi-Domain AI • Real-Time Cross-Domain Intelligence"
        activeFeatures={['ai', 'learning', 'optimized']}
      />

      {/* Enhanced Navigation */}
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

        {/* Platform Intelligence Bar */}
        <div className="px-6 pb-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <div className="flex items-center space-x-6">
              <div className="flex items-center space-x-2">
                <Globe className="w-3 h-3 text-cyan-400 animate-pulse" />
                <span>{metrics.crossDomainConnections} cross-domain connections</span>
              </div>
              <div className="flex items-center space-x-2">
                <TrendingUp className="w-3 h-3 text-green-400" />
                <span>Significance: {metrics.significanceScore.toFixed(1)}/10</span>
              </div>
              <div className="flex items-center space-x-2">
                <Users className="w-3 h-3 text-purple-400" />
                <span>{metrics.globalUsers.toLocaleString()} global users</span>
              </div>
              <div className="flex items-center space-x-2">
                <Zap className="w-3 h-3 text-orange-400 animate-pulse" />
                <span>{metrics.aiOptimizations} AI optimizations</span>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="px-3 py-1 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded-full text-cyan-300 border border-cyan-500/20 text-xs font-medium">
                Intelligence Synergy: {metrics.intelligenceSynergy.toFixed(1)}%
              </div>
              <div className="px-3 py-1 bg-gradient-to-r from-green-500/10 to-emerald-500/10 rounded-full text-green-300 border border-green-500/20 text-xs font-medium">
                Operating System for Digital Life • Production Ready
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="relative">
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
