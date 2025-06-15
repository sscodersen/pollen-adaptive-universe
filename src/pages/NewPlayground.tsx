import React, { useState, useEffect } from 'react';
import { ArrowLeft, Menu, Users, ShoppingBag, Play, Search, Bot, Globe, BarChart3, Target, Briefcase } from 'lucide-react';
import { UnifiedHeader } from '../components/UnifiedHeader';
import { ActivityFeed } from '../components/ActivityFeed';
import { SocialFeed } from '../components/SocialFeed';
import { EntertainmentHub } from '../components/EntertainmentHub';
import { NewsEngine } from '../components/NewsEngine';
import { CommunityHub } from '../components/CommunityHub';
import { TaskAutomation } from '../components/TaskAutomation';
import { ShopHub } from '../components/ShopHub';
import { AnalyticsDashboard } from '../components/AnalyticsDashboard';
import { pollenAI } from '../services/pollenAI';

const NewPlayground = () => {
  const [activeTab, setActiveTab] = useState('Social');
  const [activities, setActivities] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [aiStatus, setAiStatus] = useState('ready');
  const [platformMetrics, setPlatformMetrics] = useState({
    crossDomainConnections: 127,
    significanceScore: 8.9,
    globalUsers: 24891,
    aiOptimizations: 156
  });

  const tabs = [
    { id: 'Social', name: 'Social Intelligence', icon: Users, description: 'AI-curated social insights' },
    { id: 'Entertainment', name: 'Content Studio', icon: Play, description: 'AI-generated entertainment' },
    { id: 'Search', name: 'News Intelligence', icon: Search, description: 'Real-time news analysis' },
    { id: 'Shop', name: 'Smart Commerce', icon: ShoppingBag, description: 'Intelligent shopping' },
    { id: 'Automation', name: 'Task Automation', icon: Bot, description: 'Automated workflows' },
    { id: 'Community', name: 'Global Network', icon: Globe, description: 'Connected intelligence' },
    { id: 'Analytics', name: 'Platform Analytics', icon: BarChart3, description: 'Performance insights' },
    { id: 'Ads', name: 'Ad Intelligence', icon: Target, description: 'Smart advertising' },
    { id: 'Workspace', name: 'Digital Workspace', icon: Briefcase, description: 'Productivity hub' }
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
      aiOptimizations: prev.aiOptimizations + Math.floor(Math.random() * 5)
    }));
  };

  const updateSystemStatus = () => {
    const stats = pollenAI.getMemoryStats();
    setAiStatus(stats.isLearning ? 'learning' : 'ready');
  };

  const initializePlatform = () => {
    setIsGenerating(true);
    
    // Enhanced initial activities with deeper cross-domain intelligence
    const initialActivities = [
      {
        id: '1',
        type: 'ai_insight',
        user: {
          name: 'Pollen Intelligence Core',
          avatar: 'bg-gradient-to-r from-cyan-500 to-purple-500',
          initial: 'P'
        },
        action: 'synthesized cross-domain intelligence and generated',
        target: 'next-generation platform insights',
        content: `Successfully integrated ${platformMetrics.crossDomainConnections.toLocaleString()} cross-domain connections across all platform domains. Our enhanced 7-factor significance algorithm now processes real-time data with 98.9% accuracy. Cross-domain pattern recognition has identified 47 emerging trends with predictive confidence above 96%, enabling breakthrough discovery acceleration across news, entertainment, commerce, workspace, and social platforms.`,
        timestamp: '2m',
        aiGenerated: true,
        confidence: 0.989
      },
      {
        id: '2',
        type: 'system',
        user: {
          name: 'Platform Intelligence Engine',
          avatar: 'bg-gradient-to-r from-green-500 to-blue-500',
          initial: 'P'
        },
        action: 'optimized unified content generation algorithms with',
        target: 'multi-domain significance scoring and cross-platform intelligence',
        content: `Platform algorithms now generate personalized content across all 8 domains with 97% relevance accuracy. New significance scoring ensures only high-impact content reaches users. Cross-domain intelligence detects connections between news trends and entertainment preferences, workspace productivity patterns and social engagement, commerce insights and automation opportunities.`,
        timestamp: '18m'
      },
      {
        id: '3',
        type: 'community',
        user: {
          name: 'Global Intelligence Network',
          avatar: 'bg-gradient-to-r from-purple-500 to-pink-500',
          initial: 'G'
        },
        action: 'established distributed intelligence connections with',
        target: 'enhanced real-time collaboration and workspace optimization',
        content: `Successfully integrated with 18,500+ verified intelligence sources worldwide. Real-time significance analysis enables breakthrough discovery acceleration and unbiased information synthesis. Cross-domain collaboration tools now connect teams working on related challenges across different domains, increasing collective intelligence and productivity.`,
        timestamp: '1h'
      }
    ];
    
    setActivities(initialActivities);
    setIsGenerating(false);
  };

  const renderTabContent = () => {
    const commonProps = { isGenerating };
    
    switch (activeTab) {
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
      case 'Community':
        return <CommunityHub activities={activities} {...commonProps} />;
      case 'Analytics':
        return <AnalyticsDashboard />;
      case 'Ads':
        return <div className="p-6 text-center">
          <h2 className="text-2xl font-bold text-cyan-300 mb-4">Ad Intelligence Studio</h2>
          <p className="text-slate-400">Navigate to <a href="/ads" className="text-cyan-400 hover:underline">/ads</a> for the full experience</p>
        </div>;
      case 'Workspace':
        return <div className="p-6 text-center">
          <h2 className="text-2xl font-bold text-purple-300 mb-4">Digital Workspace</h2>
          <p className="text-slate-400">Navigate to <a href="/workspace" className="text-purple-400 hover:underline">/workspace</a> for the full experience</p>
        </div>;
      default:
        return <SocialFeed {...commonProps} />;
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

      {/* Enhanced Navigation with Cross-Domain Intelligence Indicators */}
      <div className="border-b border-gray-800/60 bg-gray-900/40 backdrop-blur-sm">
        <div className="flex space-x-1 px-6 py-4 overflow-x-auto">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            const hasConnections = platformMetrics.crossDomainConnections > 0;
            
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`group flex items-center space-x-3 px-6 py-4 rounded-xl transition-all duration-300 whitespace-nowrap font-medium min-w-fit relative ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 text-cyan-300 shadow-lg backdrop-blur-sm'
                    : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/50 border border-transparent'
                }`}
              >
                <tab.icon className={`w-5 h-5 transition-colors ${
                  isActive ? 'text-cyan-400' : 'group-hover:text-white'
                }`} />
                <div className="text-left">
                  <div className="font-semibold">{tab.name}</div>
                  <div className="text-xs opacity-70">{tab.description}</div>
                </div>
                {isActive && (
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse ml-2"></div>
                )}
                {hasConnections && !isActive && (
                  <div className="absolute -top-1 -right-1 w-2 h-2 bg-purple-400 rounded-full animate-pulse"></div>
                )}
              </button>
            );
          })}
        </div>

        {/* Platform Metrics Bar */}
        <div className="px-6 pb-4">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-1">
                <Globe className="w-3 h-3 text-cyan-400" />
                <span>{platformMetrics.crossDomainConnections} cross-domain connections</span>
              </div>
              <div className="flex items-center space-x-1">
                <TrendingUp className="w-3 h-3 text-green-400" />
                <span>Significance: {platformMetrics.significanceScore.toFixed(1)}/10</span>
              </div>
              <div className="flex items-center space-x-1">
                <Users className="w-3 h-3 text-purple-400" />
                <span>{platformMetrics.globalUsers.toLocaleString()} global users</span>
              </div>
              <div className="flex items-center space-x-1">
                <Zap className="w-3 h-3 text-orange-400" />
                <span>{platformMetrics.aiOptimizations} AI optimizations</span>
              </div>
            </div>
            <div className="px-2 py-1 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 rounded text-cyan-300 border border-cyan-500/20">
              Real-time Intelligence Active
            </div>
          </div>
        </div>
      </div>

      {/* Enhanced Content Area with Intelligence Overlay */}
      <div className="relative">
        {/* Active Tab Indicator */}
        {activeTabInfo && (
          <div className="absolute top-0 left-0 right-0 bg-gradient-to-r from-transparent via-cyan-500/5 to-transparent h-px"></div>
        )}
        
        {renderTabContent()}
      </div>
    </div>
  );
};

export default NewPlayground;
