
import React, { useState, useEffect } from 'react';
import { ArrowLeft, Menu, Users, ShoppingBag, Play, Search, Bot, Globe } from 'lucide-react';
import { ActivityFeed } from '../components/ActivityFeed';
import { SocialFeed } from '../components/SocialFeed';
import { EntertainmentHub } from '../components/EntertainmentHub';
import { NewsEngine } from '../components/NewsEngine';
import { CommunityHub } from '../components/CommunityHub';
import { TaskAutomation } from '../components/TaskAutomation';
import { ShopHub } from '../components/ShopHub';
import { pollenAI } from '../services/pollenAI';

const NewPlayground = () => {
  const [activeTab, setActiveTab] = useState('Social');
  const [activities, setActivities] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [aiStatus, setAiStatus] = useState('ready');

  const tabs = [
    { id: 'Social', name: 'Social', icon: Users },
    { id: 'Entertainment', name: 'Entertainment', icon: Play },
    { id: 'Search', name: 'Search & News', icon: Search },
    { id: 'Shop', name: 'Shop', icon: ShoppingBag },
    { id: 'Automation', name: 'Task Automation', icon: Bot },
    { id: 'Community', name: 'Community', icon: Globe }
  ];

  useEffect(() => {
    generateInitialContent();
    
    const statusInterval = setInterval(() => {
      const stats = pollenAI.getMemoryStats();
      setAiStatus(stats.isLearning ? 'learning' : 'ready');
    }, 5000);

    return () => clearInterval(statusInterval);
  }, []);

  const generateInitialContent = async () => {
    const initialActivities = [
      {
        id: '1',
        type: 'ai_insight',
        user: {
          name: 'Pollen AI',
          avatar: 'bg-gradient-to-r from-cyan-500 to-purple-500',
          initial: 'P'
        },
        action: 'generated insights for',
        target: 'Social Feed',
        content: "Analyzed trending patterns and generated personalized content recommendations.",
        timestamp: '2m',
        aiGenerated: true,
        confidence: 0.94
      },
      {
        id: '2',
        type: 'community',
        user: {
          name: 'Sarah Chen',
          avatar: 'bg-blue-500',
          initial: 'S'
        },
        action: 'shared a workflow automation in',
        target: 'Task Automation',
        content: "Created an automated data processing pipeline that reduced manual work by 80%.",
        timestamp: '15m'
      }
    ];
    
    setActivities(initialActivities);
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
          </div>
        </div>
        <button className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
          <Menu className="w-5 h-5" />
        </button>
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
    </div>
  );
};

export default NewPlayground;
