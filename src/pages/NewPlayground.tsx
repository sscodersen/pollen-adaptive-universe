
import React, { useState, useEffect } from 'react';
import { ArrowLeft, Menu, Paperclip, Send, MoreHorizontal, Download, Heart, MessageCircle, Share2, Users, Brain, Zap } from 'lucide-react';
import { ActivityFeed } from '../components/ActivityFeed';
import { PersonalWorkspace } from '../components/PersonalWorkspace';
import { TeamWorkspace } from '../components/TeamWorkspace';
import { CommunityHub } from '../components/CommunityHub';
import { pollenAI } from '../services/pollenAI';

const NewPlayground = () => {
  const [activeTab, setActiveTab] = useState('Personal');
  const [activities, setActivities] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [aiStatus, setAiStatus] = useState('ready');

  const tabs = [
    { id: 'Personal', name: 'Personal', count: null, icon: Users },
    { id: 'Team', name: 'Team', count: 3, icon: Brain },
    { id: 'Community', name: 'Community', count: null, icon: Zap }
  ];

  useEffect(() => {
    // Initialize with sample content
    generateInitialContent();
    
    // Set up AI status monitoring
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
        target: 'Personal Workspace',
        content: "Based on your recent activity patterns, I've identified 3 optimization opportunities for your workflow.",
        timestamp: '2m',
        aiGenerated: true,
        confidence: 0.92
      },
      {
        id: '2',
        type: 'collaboration',
        user: {
          name: 'Sarah Chen',
          avatar: 'bg-blue-500',
          initial: 'S'
        },
        action: 'shared a design system with',
        target: 'Team Workspace',
        content: "Updated the component library with new accessibility guidelines and dark mode variants.",
        timestamp: '1h',
        teamActivity: true
      }
    ];
    
    setActivities(initialActivities);
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'Personal':
        return <PersonalWorkspace activities={activities} isGenerating={isGenerating} />;
      case 'Team':
        return <TeamWorkspace activities={activities} isGenerating={isGenerating} />;
      case 'Community':
        return <CommunityHub activities={activities} isGenerating={isGenerating} />;
      default:
        return <PersonalWorkspace activities={activities} isGenerating={isGenerating} />;
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

      {/* Tabs */}
      <div className="flex space-x-6 px-4 py-3 border-b border-gray-800">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center space-x-3 pb-2 px-3 py-2 rounded-lg transition-all duration-200 ${
              activeTab === tab.id
                ? 'bg-gradient-to-r from-cyan-500/20 to-purple-500/20 border border-cyan-500/30 text-cyan-300'
                : 'text-gray-400 hover:text-gray-300 hover:bg-gray-800/50'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            <span className="font-medium">{tab.name}</span>
            {tab.count && (
              <span className="bg-red-500 text-white text-xs px-1.5 py-0.5 rounded-full min-w-[20px] text-center">
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Dynamic Content */}
      <div className="max-w-6xl mx-auto">
        {renderTabContent()}
      </div>
    </div>
  );
};

export default NewPlayground;
