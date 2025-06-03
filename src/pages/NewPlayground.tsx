
import React, { useState, useEffect } from 'react';
import { ArrowLeft, Menu, Paperclip, Send, MoreHorizontal, Download, Heart, MessageCircle, Share2 } from 'lucide-react';
import { ActivityFeed } from '../components/ActivityFeed';
import { pollenAI } from '../services/pollenAI';

const NewPlayground = () => {
  const [activeTab, setActiveTab] = useState('All Workspace');
  const [commentText, setCommentText] = useState('');
  const [activities, setActivities] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);

  const tabs = [
    { id: 'All Workspace', name: 'All Workspace', count: null },
    { id: 'Personal', name: 'Personal', count: null },
    { id: 'Team', name: 'Team', count: 3 },
    { id: 'Community', name: 'Community', count: null }
  ];

  useEffect(() => {
    // Generate initial content
    generateInitialContent();
    
    // Set up slower generation interval (every 45 seconds)
    const interval = setInterval(() => {
      if (!isGenerating) {
        generateNewActivity();
      }
    }, 45000);

    return () => clearInterval(interval);
  }, [isGenerating]);

  const generateInitialContent = async () => {
    const initialActivities = [
      {
        id: '1',
        type: 'comment',
        user: {
          name: 'Benjamin',
          avatar: 'bg-green-500',
          initial: 'B'
        },
        action: 'commented on',
        target: 'Sora UI Kit',
        content: "What a good design! I like how you dealt with the spacing. Where can I get this file?",
        timestamp: '1h',
        replies: [
          {
            user: { name: 'Benjamin', avatar: 'bg-blue-500', initial: 'B' },
            content: 'here is the link supaui.com/download 👍',
            timestamp: '45m'
          }
        ],
        hasNewReply: true
      },
      {
        id: '2',
        type: 'generation',
        user: {
          name: 'Jacob',
          avatar: 'bg-blue-500',
          initial: 'J'
        },
        action: 'generated a new image on',
        target: 'Midjourney',
        content: null,
        image: '/lovable-uploads/f306d6f2-5915-45e7-b800-05894257a2c7.png',
        timestamp: '8h',
        actions: ['Download', 'Use image'],
        likes: []
      }
    ];
    
    setActivities(initialActivities);
  };

  const generateNewActivity = async () => {
    setIsGenerating(true);
    
    try {
      const modes = ['social', 'news', 'entertainment'];
      const randomMode = modes[Math.floor(Math.random() * modes.length)];
      
      const response = await pollenAI.generate(
        `Generate ${randomMode} content for the activity feed`,
        randomMode
      );
      
      const newActivity = {
        id: Date.now().toString(),
        type: randomMode === 'social' ? 'comment' : 'generation',
        user: {
          name: getRandomName(),
          avatar: getRandomAvatar(),
          initial: getRandomName()[0]
        },
        action: randomMode === 'social' ? 'shared' : 'generated content on',
        target: getRandomTarget(randomMode),
        content: response.content,
        timestamp: 'now',
        likes: []
      };
      
      setActivities(prev => [newActivity, ...prev.slice(0, 9)]);
    } catch (error) {
      console.error('Failed to generate activity:', error);
    }
    
    setIsGenerating(false);
  };

  const getRandomName = () => {
    const names = ['Alex', 'Sam', 'Jordan', 'Casey', 'Riley', 'Taylor', 'Morgan', 'Avery'];
    return names[Math.floor(Math.random() * names.length)];
  };

  const getRandomAvatar = () => {
    const colors = ['bg-red-500', 'bg-blue-500', 'bg-green-500', 'bg-purple-500', 'bg-orange-500', 'bg-pink-500'];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  const getRandomTarget = (mode) => {
    const targets = {
      social: ['Design System', 'UI Kit', 'Component Library'],
      news: ['TechCrunch', 'Wired', 'The Verge'],
      entertainment: ['Midjourney', 'RunwayML', 'Creative Studio']
    };
    const modeTargets = targets[mode] || targets.social;
    return modeTargets[Math.floor(Math.random() * modeTargets.length)];
  };

  const handleSubmitComment = () => {
    if (!commentText.trim()) return;
    // Handle comment submission
    setCommentText('');
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <div className="flex items-center space-x-4">
          <button className="p-2 hover:bg-gray-800 rounded-lg transition-colors">
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h1 className="text-xl font-semibold">Activity</h1>
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
            className={`flex items-center space-x-2 pb-2 transition-colors ${
              activeTab === tab.id
                ? 'text-white border-b-2 border-white'
                : 'text-gray-400 hover:text-gray-300'
            }`}
          >
            <span>{tab.name}</span>
            {tab.count && (
              <span className="bg-red-500 text-white text-xs px-1.5 py-0.5 rounded-full min-w-[20px] text-center">
                {tab.count}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Activity Feed */}
      <div className="max-w-2xl mx-auto">
        <ActivityFeed activities={activities} isGenerating={isGenerating} />
      </div>
    </div>
  );
};

export default NewPlayground;
