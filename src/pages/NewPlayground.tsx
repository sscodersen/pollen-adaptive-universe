
import React, { useState, useEffect } from 'react';
import { ArrowLeft, Menu, Users, ShoppingBag, Play, Search, Bot, Globe, BarChart3, Settings, User, LogOut } from 'lucide-react';
import { ActivityFeed } from '../components/ActivityFeed';
import { SocialFeed } from '../components/SocialFeed';
import { EntertainmentHub } from '../components/EntertainmentHub';
import { NewsEngine } from '../components/NewsEngine';
import { CommunityHub } from '../components/CommunityHub';
import { TaskAutomation } from '../components/TaskAutomation';
import { ShopHub } from '../components/ShopHub';
import { AnalyticsDashboard } from '../components/AnalyticsDashboard';
import { WorkflowBuilder } from '../components/WorkflowBuilder';
import { AuthModal } from '../components/AuthModal';
import { backendService, type AuthState } from '../services/backendService';
import { webScrapingService } from '../services/webScrapingService';
import { pollenAI } from '../services/pollenAI';

const NewPlayground = () => {
  const [activeTab, setActiveTab] = useState('Social');
  const [activities, setActivities] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [aiStatus, setAiStatus] = useState('ready');
  const [authState, setAuthState] = useState<AuthState>({ isAuthenticated: false, user: null, token: null });
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [showUserMenu, setShowUserMenu] = useState(false);

  const tabs = [
    { id: 'Social', name: 'Social', icon: Users },
    { id: 'Entertainment', name: 'Entertainment', icon: Play },
    { id: 'Search', name: 'Search & News', icon: Search },
    { id: 'Shop', name: 'Shop', icon: ShoppingBag },
    { id: 'Automation', name: 'Task Automation', icon: Bot },
    { id: 'Workflows', name: 'Workflow Builder', icon: Settings },
    { id: 'Community', name: 'Community', icon: Globe },
    { id: 'Analytics', name: 'Analytics', icon: BarChart3 }
  ];

  useEffect(() => {
    initializePlatform();
    
    const statusInterval = setInterval(() => {
      const stats = pollenAI.getMemoryStats();
      setAiStatus(stats.isLearning ? 'learning' : 'ready');
    }, 5000);

    // Listen for real-time updates
    const handlePollenUpdate = (event: any) => {
      console.log('Real-time update:', event.detail);
      // Handle real-time updates from WebSocket
    };

    window.addEventListener('pollenUpdate', handlePollenUpdate);

    return () => {
      clearInterval(statusInterval);
      window.removeEventListener('pollenUpdate', handlePollenUpdate);
    };
  }, []);

  const initializePlatform = async () => {
    // Initialize auth state
    const currentAuthState = backendService.getAuthState();
    setAuthState(currentAuthState);

    // Generate initial activities
    const initialActivities = [
      {
        id: '1',
        type: 'ai_insight',
        user: {
          name: 'Pollen AI',
          avatar: 'bg-gradient-to-r from-cyan-500 to-purple-500',
          initial: 'P'
        },
        action: 'analyzed global trends and generated',
        target: 'high-significance content',
        content: "Identified 12 emerging patterns across technology, science, and social innovation with actionable insights for immediate implementation.",
        timestamp: '2m',
        aiGenerated: true,
        confidence: 0.96
      },
      {
        id: '2',
        type: 'community',
        user: {
          name: currentAuthState.user?.username || 'Anonymous User',
          avatar: currentAuthState.user?.avatar || 'bg-blue-500',
          initial: currentAuthState.user?.username?.[0] || 'A'
        },
        action: 'integrated real-time web scraping with',
        target: 'significance algorithm',
        content: "Platform now analyzes and ranks content from 500+ sources in real-time, ensuring only the most impactful information reaches users.",
        timestamp: '15m'
      },
      {
        id: '3',
        type: 'system',
        user: {
          name: 'System',
          avatar: 'bg-green-500',
          initial: 'S'
        },
        action: 'optimized content generation pipeline',
        target: 'performance boost',
        content: "Implemented advanced caching and significance filtering. Content relevance improved by 340% with 60% faster load times.",
        timestamp: '1h'
      }
    ];
    
    setActivities(initialActivities);

    // Initialize web scraping
    try {
      await webScrapingService.scrapeContent('news', 5);
      await webScrapingService.scrapeContent('shop', 5);
      await webScrapingService.scrapeContent('entertainment', 5);
    } catch (error) {
      console.error('Failed to initialize web scraping:', error);
    }
  };

  const handleAuthSuccess = (newAuthState: AuthState) => {
    setAuthState(newAuthState);
    setShowAuthModal(false);
    
    // Refresh activities with user context
    initializePlatform();
  };

  const handleLogout = () => {
    backendService.logout();
    setAuthState({ isAuthenticated: false, user: null, token: null });
    setShowUserMenu(false);
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
      case 'Workflows':
        return <WorkflowBuilder />;
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
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <div className="flex items-center space-x-4">
          <button className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex items-center space-x-3">
            <h1 className="text-xl font-semibold">Pollen Platform</h1>
            <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs ${
              aiStatus === 'learning' 
                ? 'bg-yellow-500/20 text-yellow-400' 
                : 'bg-green-500/20 text-green-400'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                aiStatus === 'learning' ? 'bg-yellow-400 animate-pulse' : 'bg-green-400'
              }`}></div>
              <span>AI {aiStatus === 'learning' ? 'Learning' : 'Ready'}</span>
            </div>
            <div className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-xs">
              Production Ready
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          {authState.isAuthenticated ? (
            <div className="relative">
              <button
                onClick={() => setShowUserMenu(!showUserMenu)}
                className="flex items-center space-x-3 p-2 hover:bg-gray-800 rounded-lg transition-colors"
              >
                <div className={`w-8 h-8 ${authState.user?.avatar} rounded-full flex items-center justify-center text-white font-medium`}>
                  {authState.user?.username?.[0] || 'U'}
                </div>
                <span className="text-sm font-medium">{authState.user?.username}</span>
              </button>
              
              {showUserMenu && (
                <div className="absolute right-0 top-full mt-2 w-48 bg-gray-800 rounded-lg border border-gray-700 shadow-lg z-50">
                  <div className="p-3 border-b border-gray-700">
                    <p className="text-sm font-medium text-white">{authState.user?.username}</p>
                    <p className="text-xs text-gray-400">{authState.user?.email}</p>
                  </div>
                  <div className="p-2">
                    <button
                      onClick={handleLogout}
                      className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-red-400 hover:bg-gray-700 rounded transition-colors"
                    >
                      <LogOut className="w-4 h-4" />
                      <span>Sign Out</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <button
              onClick={() => setShowAuthModal(true)}
              className="flex items-center space-x-2 px-4 py-2 bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 rounded-lg transition-all"
            >
              <User className="w-4 h-4" />
              <span>Sign In</span>
            </button>
          )}
          
          <button className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
            <Menu className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Feature Tabs */}
      <div className="flex space-x-2 px-4 py-3 border-b border-gray-800 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 whitespace-nowrap ${
              activeTab === tab.id
                ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 text-cyan-300'
                : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/50'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            <span className="font-medium">{tab.name}</span>
          </button>
        ))}
      </div>

      {/* Dynamic Content */}
      <div className="max-w-full">
        {renderTabContent()}
      </div>

      {/* Auth Modal */}
      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onAuthSuccess={handleAuthSuccess}
      />
    </div>
  );
};

export default NewPlayground;
