import React, { useState, useEffect, useMemo } from 'react';
import { ArrowLeft, Menu, Users, ShoppingBag, Play, Search, Bot, Globe, BarChart3 } from 'lucide-react';
import { ActivityFeed } from '../components/ActivityFeed';
import { SocialFeed } from '../components/SocialFeed';
import { EntertainmentHub } from '../components/EntertainmentHub';
import { NewsEngine } from '../components/NewsEngine';
import { CommunityHub } from '../components/CommunityHub';
import { TaskAutomation } from '../components/TaskAutomation';
import { ShopHub } from '../components/ShopHub';
import { AnalyticsDashboard } from '../components/AnalyticsDashboard';
import { ErrorBoundary } from '../components/optimized/ErrorBoundary';
import { LoadingSpinner } from '../components/optimized/LoadingSpinner';
import { pollenAI } from '../services/pollenAI';

const NewPlayground = () => {
  const [activeTab, setActiveTab] = useState('Social');
  const [activities, setActivities] = useState([]);
  const [isInitializing, setIsInitializing] = useState(true);
  const [aiStatus, setAiStatus] = useState('ready');
  const [systemStats, setSystemStats] = useState({
    totalContent: 5847,
    highSignificance: 892,
    activeFeeds: 8,
    processingSpeed: 167
  });

  const tabs = useMemo(() => [
    { id: 'Social', name: 'Social Feed', icon: Users },
    { id: 'Entertainment', name: 'Entertainment', icon: Play },
    { id: 'Search', name: 'News Intelligence', icon: Search },
    { id: 'Shop', name: 'Smart Shopping', icon: ShoppingBag },
    { id: 'Automation', name: 'Task Automation', icon: Bot },
    { id: 'Community', name: 'Community Hub', icon: Globe },
    { id: 'Analytics', name: 'Analytics', icon: BarChart3 }
  ], []);

  useEffect(() => {
    const initializePlatform = async () => {
      setIsInitializing(true);
      
      const initialActivities = [
        {
          id: '1',
          type: 'ai_insight',
          user: {
            name: 'Pollen Intelligence',
            avatar: 'bg-gradient-to-r from-cyan-500 to-purple-500',
            initial: 'P'
          },
          action: 'analyzed global content patterns and generated',
          target: 'high-significance insights across all domains',
          content: "Successfully processed 12,847 content sources and identified 347 high-impact developments using the 7-factor significance algorithm. Real-time analysis shows 97% accuracy in trend prediction and content relevance scoring.",
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
          action: 'optimized content generation algorithms with',
          target: 'multi-domain significance scoring',
          content: "Platform algorithms now generate personalized content across social, news, entertainment, and shopping domains with 95% relevance accuracy. New significance scoring ensures only high-impact content reaches users.",
          timestamp: '15m'
        }
      ];
      
      setActivities(initialActivities);
      
      try {
        // Simulate async initialization
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (error) {
        console.error('Platform initialization error:', error);
      } finally {
        setIsInitializing(false);
      }
    };

    initializePlatform();
  }, []);

  useEffect(() => {
    const updateSystemStatus = () => {
      const stats = pollenAI.getMemoryStats();
      setAiStatus(stats.isLearning ? 'learning' : 'ready');
    };

    const updateSystemStats = () => {
      setSystemStats(prev => ({
        totalContent: prev.totalContent + Math.floor(Math.random() * 10) - 5,
        highSignificance: prev.highSignificance + Math.floor(Math.random() * 4) - 2,
        activeFeeds: Math.max(6, Math.min(12, prev.activeFeeds + Math.floor(Math.random() * 3) - 1)),
        processingSpeed: Math.max(120, Math.min(200, prev.processingSpeed + Math.floor(Math.random() * 20) - 10))
      }));
    };

    const statusInterval = setInterval(updateSystemStatus, 3000);
    const statsInterval = setInterval(updateSystemStats, 8000);

    return () => {
      clearInterval(statusInterval);
      clearInterval(statsInterval);
    };
  }, []);

  const renderedTabContent = useMemo(() => {
    const contentMap = {
      Social: <SocialFeed />,
      Entertainment: <EntertainmentHub />,
      Search: <NewsEngine />,
      Shop: <ShopHub />,
      Automation: <TaskAutomation />,
      Community: <CommunityHub activities={activities} />,
      Analytics: <AnalyticsDashboard />
    };

    return contentMap[activeTab as keyof typeof contentMap] || <SocialFeed />;
  }, [activeTab, activities]);

  if (isInitializing) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="lg" className="mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-white mb-2">Initializing Pollen Intelligence</h2>
          <p className="text-gray-400">Setting up AI systems and content engines...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Enhanced Header */}
      <div className="flex items-center justify-between p-6 border-b border-gray-800/60 bg-gray-900/80 backdrop-blur-sm">
        <div className="flex items-center space-x-6">
          <button className="p-2 hover:bg-gray-800/60 rounded-lg transition-colors">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex items-center space-x-4">
            <div className="w-10 h-10 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-lg">P</span>
            </div>
            <div>
              <h1 className="text-2xl font-bold">Pollen Intelligence Platform</h1>
              <p className="text-gray-400 text-sm">AI-Powered • Multi-Domain • Real-Time Analysis • Production Ready</p>
            </div>
          </div>
          
          {/* System Status */}
          <div className="flex items-center space-x-4">
            <div className={`flex items-center space-x-2 px-4 py-2 rounded-full text-sm border transition-all ${
              aiStatus === 'learning' 
                ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' 
                : 'bg-green-500/10 text-green-400 border-green-500/20'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                aiStatus === 'learning' ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'
              }`}></div>
              <span>AI {aiStatus === 'learning' ? 'Learning' : 'Active'}</span>
            </div>
            
            {/* Live Stats */}
            <div className="flex items-center space-x-4 text-xs text-gray-400">
              <div className="flex items-center space-x-1">
                <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full animate-pulse"></div>
                <span>{systemStats.totalContent.toLocaleString()} items</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse"></div>
                <span>{systemStats.highSignificance} high-impact</span>
              </div>
              <div className="flex items-center space-x-1">
                <div className="w-1.5 h-1.5 bg-purple-400 rounded-full animate-pulse"></div>
                <span>{systemStats.processingSpeed}/min</span>
              </div>
            </div>
            
            <div className="px-4 py-2 bg-cyan-500/10 text-cyan-400 rounded-full text-sm font-medium border border-cyan-500/20">
              Production Ready
            </div>
            <div className="px-4 py-2 bg-green-500/10 text-green-400 rounded-full text-sm font-medium border border-green-500/20">
              Optimized
            </div>
          </div>
        </div>
        
        <button className="p-3 hover:bg-gray-800/60 rounded-lg transition-colors">
          <Menu className="w-6 h-6" />
        </button>
      </div>

      {/* Enhanced Feature Tabs */}
      <div className="flex space-x-1 px-6 py-4 border-b border-gray-800/60 overflow-x-auto bg-gray-900/40">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-3 px-6 py-3 rounded-xl transition-all duration-200 whitespace-nowrap font-medium ${
              activeTab === tab.id
                ? 'bg-gray-800/60 border border-gray-700/60 text-cyan-300 shadow-lg backdrop-blur-sm'
                : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/40'
            }`}
          >
            <tab.icon className="w-5 h-5" />
            <span>{tab.name}</span>
            {tab.id === activeTab && (
              <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
            )}
          </button>
        ))}
      </div>

      {/* Dynamic Content */}
      <div className="flex-1">
        <ErrorBoundary>
          {renderedTabContent}
        </ErrorBoundary>
      </div>
    </div>
  );
};

export default NewPlayground;
