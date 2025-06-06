
import React, { useState, useEffect } from 'react';
import { ArrowLeft, Menu, Users, ShoppingBag, Play, Search, Bot, Globe, BarChart3 } from 'lucide-react';
import { ActivityFeed } from '../components/ActivityFeed';
import { SocialFeed } from '../components/SocialFeed';
import { EntertainmentHub } from '../components/EntertainmentHub';
import { NewsEngine } from '../components/NewsEngine';
import { CommunityHub } from '../components/CommunityHub';
import { TaskAutomation } from '../components/TaskAutomation';
import { ShopHub } from '../components/ShopHub';
import { AnalyticsDashboard } from '../components/AnalyticsDashboard';
import { webScrapingService } from '../services/webScrapingService';
import { pollenAI } from '../services/pollenAI';

const NewPlayground = () => {
  const [activeTab, setActiveTab] = useState('Social');
  const [activities, setActivities] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [aiStatus, setAiStatus] = useState('ready');

  const tabs = [
    { id: 'Social', name: 'Social Intelligence', icon: Users },
    { id: 'Entertainment', name: 'Entertainment', icon: Play },
    { id: 'Search', name: 'News Engine', icon: Search },
    { id: 'Shop', name: 'Smart Shopping', icon: ShoppingBag },
    { id: 'Automation', name: 'Task Automation', icon: Bot },
    { id: 'Community', name: 'Community Hub', icon: Globe },
    { id: 'Analytics', name: 'Analytics', icon: BarChart3 }
  ];

  useEffect(() => {
    initializePlatform();
    
    const statusInterval = setInterval(() => {
      const stats = pollenAI.getMemoryStats();
      setAiStatus(stats.isLearning ? 'learning' : 'ready');
    }, 5000);

    return () => {
      clearInterval(statusInterval);
    };
  }, []);

  const initializePlatform = async () => {
    // Generate initial activities for community hub
    const initialActivities = [
      {
        id: '1',
        type: 'ai_insight',
        user: {
          name: 'Pollen AI',
          avatar: 'bg-gradient-to-r from-cyan-500 to-purple-500',
          initial: 'P'
        },
        action: 'analyzed global patterns and generated',
        target: 'high-significance content across all domains',
        content: "Successfully identified and curated 247 high-impact developments across technology, science, business, and social innovation with comprehensive significance scoring and real-time trend analysis.",
        timestamp: '2m',
        aiGenerated: true,
        confidence: 0.97
      },
      {
        id: '2',
        type: 'system',
        user: {
          name: 'Content Engine',
          avatar: 'bg-gradient-to-r from-green-500 to-blue-500',
          initial: 'C'
        },
        action: 'integrated multi-domain content generation with',
        target: 'significance algorithm optimization',
        content: "Platform now generates personalized content across social, news, entertainment, and shopping domains with 95% relevance accuracy and real-time adaptation to user preferences.",
        timestamp: '15m'
      },
      {
        id: '3',
        type: 'community',
        user: {
          name: 'Global Network',
          avatar: 'bg-gradient-to-r from-purple-500 to-pink-500',
          initial: 'G'
        },
        action: 'connected with distributed intelligence network',
        target: 'collaborative insight generation',
        content: "Successfully established connections with 12,000+ content sources and analysis engines worldwide, enabling comprehensive perspective synthesis and breakthrough discovery acceleration.",
        timestamp: '1h'
      }
    ];
    
    setActivities(initialActivities);

    // Initialize web scraping services
    try {
      await webScrapingService.scrapeContent('news', 5);
      await webScrapingService.scrapeContent('shop', 5);
      await webScrapingService.scrapeContent('entertainment', 5);
    } catch (error) {
      console.error('Failed to initialize web scraping:', error);
    }
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'Social':
        return <SocialFeed isGenerating={true} />;
      case 'Entertainment':
        return <EntertainmentHub isGenerating={true} />;
      case 'Search':
        return <NewsEngine isGenerating={true} />;
      case 'Shop':
        return <ShopHub isGenerating={true} />;
      case 'Automation':
        return <TaskAutomation isGenerating={true} />;
      case 'Community':
        return <CommunityHub activities={activities} isGenerating={isGenerating} />;
      case 'Analytics':
        return <AnalyticsDashboard />;
      default:
        return <SocialFeed isGenerating={true} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Enhanced Header */}
      <div className="flex items-center justify-between p-6 border-b border-gray-800 bg-gradient-to-r from-gray-900 to-gray-800">
        <div className="flex items-center space-x-6">
          <button className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex items-center space-x-4">
            <div className="w-10 h-10 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl flex items-center justify-center">
              <span className="text-white font-bold text-lg">P</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold">Pollen Intelligence Platform</h1>
              <p className="text-gray-400 text-sm">AI-Powered • Multi-Domain • Real-Time Analysis</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 px-4 py-2 rounded-full text-sm ${
              aiStatus === 'learning' 
                ? 'bg-yellow-500/20 text-yellow-400' 
                : 'bg-green-500/20 text-green-400'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                aiStatus === 'learning' ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'
              }`}></div>
              <span>AI {aiStatus === 'learning' ? 'Learning' : 'Active'}</span>
            </div>
            <div className="px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-full text-sm font-medium">
              Production Ready
            </div>
          </div>
        </div>
        
        <button className="p-3 hover:bg-gray-800 rounded-lg transition-colors">
          <Menu className="w-6 h-6" />
        </button>
      </div>

      {/* Enhanced Feature Tabs */}
      <div className="flex space-x-1 px-6 py-4 border-b border-gray-800 overflow-x-auto bg-gray-900/50">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-3 px-6 py-3 rounded-xl transition-all duration-200 whitespace-nowrap font-medium ${
              activeTab === tab.id
                ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 text-cyan-300 shadow-lg'
                : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/70'
            }`}
          >
            <tab.icon className="w-5 h-5" />
            <span>{tab.name}</span>
          </button>
        ))}
      </div>

      {/* Dynamic Content */}
      <div className="max-w-full">
        {renderTabContent()}
      </div>
    </div>
  );
};

export default NewPlayground;
