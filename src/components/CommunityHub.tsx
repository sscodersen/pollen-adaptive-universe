
import React, { useState, useEffect } from 'react';
import { Users, TrendingUp, MessageCircle, Heart, Zap, Globe } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';

interface CommunityHubProps {
  activities: any[];
  isGenerating: boolean;
}

export const CommunityHub = ({ activities, isGenerating }: CommunityHubProps) => {
  const [communityStats, setCommunityStats] = useState({
    totalMembers: 12847,
    activeToday: 1205,
    postsToday: 324,
    engagement: 89
  });
  const [trendingTopics, setTrendingTopics] = useState([]);

  useEffect(() => {
    generateCommunityContent();
  }, []);

  const generateCommunityContent = async () => {
    try {
      const topics = [
        'AI Ethics Discussion',
        'Future of Work',
        'Sustainable Technology',
        'Creative AI Applications',
        'Human-AI Collaboration'
      ];
      setTrendingTopics(topics);
    } catch (error) {
      console.error('Failed to generate community content:', error);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Community Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center space-x-3">
            <Users className="w-8 h-8 text-blue-400" />
            <div>
              <p className="text-2xl font-bold text-white">{communityStats.totalMembers.toLocaleString()}</p>
              <p className="text-sm text-gray-400">Total Members</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center space-x-3">
            <Zap className="w-8 h-8 text-green-400" />
            <div>
              <p className="text-2xl font-bold text-white">{communityStats.activeToday.toLocaleString()}</p>
              <p className="text-sm text-gray-400">Active Today</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center space-x-3">
            <MessageCircle className="w-8 h-8 text-purple-400" />
            <div>
              <p className="text-2xl font-bold text-white">{communityStats.postsToday}</p>
              <p className="text-sm text-gray-400">Posts Today</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center space-x-3">
            <Heart className="w-8 h-8 text-red-400" />
            <div>
              <p className="text-2xl font-bold text-white">{communityStats.engagement}%</p>
              <p className="text-sm text-gray-400">Engagement</p>
            </div>
          </div>
        </div>
      </div>

      {/* Trending Topics */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
          <TrendingUp className="w-5 h-5 text-orange-400" />
          <span>Trending Topics</span>
        </h3>
        <div className="space-y-3">
          {trendingTopics.map((topic, index) => (
            <div key={index} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
              <span className="text-white font-medium">{topic}</span>
              <div className="flex items-center space-x-2 text-sm text-gray-400">
                <MessageCircle className="w-4 h-4" />
                <span>{Math.floor(Math.random() * 500) + 100}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Community Activity Feed */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center space-x-2">
          <Globe className="w-5 h-5 text-cyan-400" />
          <span>Community Activity</span>
        </h3>
        <div className="space-y-4">
          {activities.filter(a => a.type === 'community').map((activity, index) => (
            <div key={index} className="flex items-start space-x-3 p-3 bg-gray-700/30 rounded-lg">
              <div className={`w-10 h-10 ${activity.user.avatar} rounded-full flex items-center justify-center text-white font-medium`}>
                {activity.user.initial}
              </div>
              <div className="flex-1">
                <p className="text-gray-300">
                  <span className="font-medium text-white">{activity.user.name}</span> {activity.action} {activity.target}
                </p>
                <p className="text-sm text-gray-400 mt-1">{activity.content}</p>
                <span className="text-xs text-gray-500">{activity.timestamp}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
