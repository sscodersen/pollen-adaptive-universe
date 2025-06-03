
import React, { useState, useEffect } from 'react';
import { Globe, Star, TrendingUp, Users, MessageCircle, Share2, ThumbsUp } from 'lucide-react';
import { ActivityFeed } from './ActivityFeed';
import { pollenAI } from '../services/pollenAI';

interface CommunityHubProps {
  activities: any[];
  isGenerating: boolean;
}

export const CommunityHub = ({ activities, isGenerating }: CommunityHubProps) => {
  const [trending, setTrending] = useState([]);
  const [featured, setFeatured] = useState([]);
  const [communityStats, setCommunityStats] = useState({});

  useEffect(() => {
    generateCommunityContent();
  }, []);

  const generateCommunityContent = async () => {
    try {
      await pollenAI.generate("Generate community trending content and insights", "community");
      
      setTrending([
        {
          id: 1,
          title: "Advanced React Patterns Workshop",
          author: "Tech Community",
          engagement: 234,
          category: "Learning",
          image: "/lovable-uploads/f306d6f2-5915-45e7-b800-05894257a2c7.png"
        },
        {
          id: 2,
          title: "Open Source Design System",
          author: "Design Collective",
          engagement: 189,
          category: "Resources",
          image: null
        },
        {
          id: 3,
          title: "AI Ethics Discussion Panel",
          author: "Future Tech Group",
          engagement: 156,
          category: "Discussion",
          image: null
        }
      ]);

      setFeatured([
        {
          id: 1,
          title: "Building Accessible Components",
          description: "A comprehensive guide to creating inclusive user interfaces",
          author: "Accessibility Team",
          likes: 89,
          comments: 23,
          shares: 12
        },
        {
          id: 2,
          title: "State Management Best Practices",
          description: "Modern approaches to handling application state in React",
          author: "React Experts",
          likes: 156,
          comments: 34,
          shares: 28
        }
      ]);

      setCommunityStats({
        totalMembers: 12547,
        activeToday: 1834,
        postsToday: 89,
        engagement: 94
      });
    } catch (error) {
      console.error('Failed to generate community content:', error);
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Community Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center space-x-2 mb-2">
            <Users className="w-5 h-5 text-blue-400" />
            <span className="text-sm text-gray-400">Members</span>
          </div>
          <div className="text-2xl font-bold text-white">{communityStats.totalMembers?.toLocaleString()}</div>
        </div>
        
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="w-5 h-5 text-green-400" />
            <span className="text-sm text-gray-400">Active Today</span>
          </div>
          <div className="text-2xl font-bold text-white">{communityStats.activeToday?.toLocaleString()}</div>
        </div>
        
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center space-x-2 mb-2">
            <MessageCircle className="w-5 h-5 text-purple-400" />
            <span className="text-sm text-gray-400">Posts Today</span>
          </div>
          <div className="text-2xl font-bold text-white">{communityStats.postsToday}</div>
        </div>
        
        <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
          <div className="flex items-center space-x-2 mb-2">
            <Star className="w-5 h-5 text-yellow-400" />
            <span className="text-sm text-gray-400">Engagement</span>
          </div>
          <div className="text-2xl font-bold text-white">{communityStats.engagement}%</div>
        </div>
      </div>

      {/* Trending and Featured Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Trending */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-2 mb-4">
            <TrendingUp className="w-5 h-5 text-orange-400" />
            <h3 className="font-semibold text-white">Trending Now</h3>
          </div>
          <div className="space-y-4">
            {trending.map((item) => (
              <div key={item.id} className="flex items-start space-x-3 p-3 bg-gray-700/50 rounded-lg hover:bg-gray-700/70 transition-colors cursor-pointer">
                {item.image && (
                  <img src={item.image} alt={item.title} className="w-12 h-12 rounded-lg object-cover" />
                )}
                <div className="flex-1">
                  <h4 className="font-medium text-white text-sm mb-1">{item.title}</h4>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-400">{item.author}</span>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs bg-gray-600 px-2 py-1 rounded text-gray-300">{item.category}</span>
                      <span className="text-xs text-gray-400">{item.engagement} interactions</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Featured Content */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center space-x-2 mb-4">
            <Star className="w-5 h-5 text-yellow-400" />
            <h3 className="font-semibold text-white">Featured Content</h3>
          </div>
          <div className="space-y-4">
            {featured.map((item) => (
              <div key={item.id} className="p-4 bg-gray-700/50 rounded-lg hover:bg-gray-700/70 transition-colors cursor-pointer">
                <h4 className="font-medium text-white mb-2">{item.title}</h4>
                <p className="text-sm text-gray-300 mb-3">{item.description}</p>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">by {item.author}</span>
                  <div className="flex items-center space-x-4 text-xs text-gray-400">
                    <div className="flex items-center space-x-1">
                      <ThumbsUp className="w-3 h-3" />
                      <span>{item.likes}</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <MessageCircle className="w-3 h-3" />
                      <span>{item.comments}</span>
                    </div>
                    <div className="flex items-center space-x-1">
                      <Share2 className="w-3 h-3" />
                      <span>{item.shares}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Community Activity Feed */}
      <div className="bg-gray-800 rounded-xl border border-gray-700">
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center space-x-2">
            <Globe className="w-5 h-5 text-cyan-400" />
            <h3 className="font-semibold text-white">Community Feed</h3>
          </div>
        </div>
        <ActivityFeed 
          activities={activities} 
          isGenerating={isGenerating} 
        />
      </div>
    </div>
  );
};
