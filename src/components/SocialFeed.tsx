
import React, { useState, useEffect, useCallback } from 'react';
import { Heart, MessageCircle, Share2, Bookmark, TrendingUp, Award, Zap, Users, Globe, Sparkles } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

interface SocialFeedProps {
  isGenerating?: boolean;
}

interface Post {
  id: string;
  user: {
    name: string;
    username: string;
    avatar: string;
    verified: boolean;
    badges: string[];
  };
  content: string;
  timestamp: string;
  likes: number;
  comments: number;
  shares: number;
  tags: string[];
  trending: boolean;
  significance: number;
  category: string;
  engagement: number;
  quality: number;
}

export const SocialFeed = ({ isGenerating = false }: SocialFeedProps) => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [sortBy, setSortBy] = useState('trending');

  const realUsers = [
    { name: 'Dr. Sarah Chen', username: 'sarahchen_ai', avatar: 'bg-gradient-to-r from-blue-500 to-purple-500', verified: true, badges: ['AI Expert', 'Researcher'] },
    { name: 'Marcus Rodriguez', username: 'marcus_dev', avatar: 'bg-gradient-to-r from-green-500 to-blue-500', verified: true, badges: ['Developer', 'Open Source'] },
    { name: 'Elena Kowalski', username: 'elena_design', avatar: 'bg-gradient-to-r from-pink-500 to-red-500', verified: false, badges: ['Designer'] },
    { name: 'Dr. James Liu', username: 'james_quantum', avatar: 'bg-gradient-to-r from-purple-500 to-indigo-500', verified: true, badges: ['Physicist', 'Quantum'] },
    { name: 'Aria Patel', username: 'aria_sustainability', avatar: 'bg-gradient-to-r from-emerald-500 to-teal-500', verified: true, badges: ['Climate Tech', 'Activist'] },
    { name: 'Roberto Silva', username: 'roberto_biotech', avatar: 'bg-gradient-to-r from-orange-500 to-red-500', verified: false, badges: ['Biotech'] },
    { name: 'Dr. Maya Zhang', username: 'maya_neuroscience', avatar: 'bg-gradient-to-r from-cyan-500 to-blue-500', verified: true, badges: ['Neuroscientist'] },
    { name: 'Alex Thompson', username: 'alex_crypto', avatar: 'bg-gradient-to-r from-yellow-500 to-orange-500', verified: false, badges: ['Blockchain'] }
  ];

  const topicTemplates = [
    { topic: 'AI Research', content: 'breakthrough in neural architecture search is revolutionizing how we approach machine learning optimization. The new method reduces training time by 60% while improving accuracy across diverse datasets.' },
    { topic: 'Climate Tech', content: 'carbon capture technology just achieved a major milestone - successfully removing 1 million tons of CO2 from the atmosphere while being economically viable at scale.' },
    { topic: 'Quantum Computing', content: 'quantum error correction just reached a critical threshold. We\'re now seeing stable qubits maintaining coherence for over 100 seconds, bringing practical quantum computers much closer to reality.' },
    { topic: 'Biotechnology', content: 'gene therapy breakthrough is showing unprecedented success in treating rare genetic disorders. Clinical trials report 95% improvement rates with minimal side effects.' },
    { topic: 'Space Technology', content: 'reusable rocket technology is making space access 10x more affordable. This opens up incredible opportunities for satellite deployment and space research.' },
    { topic: 'Renewable Energy', content: 'solar panel efficiency just hit 47% in lab conditions using perovskite-silicon tandem cells. This could revolutionize how we approach sustainable energy generation.' },
    { topic: 'Neuroscience', content: 'brain-computer interface technology is enabling paralyzed patients to control robotic arms with thought alone. The precision and speed are approaching natural movement.' },
    { topic: 'Sustainable Tech', content: 'vertical farming breakthrough reduces water usage by 95% while increasing crop yields. This could transform agriculture in water-scarce regions worldwide.' },
    { topic: 'Digital Privacy', content: 'homomorphic encryption advancement allows computation on encrypted data without decryption. This preserves privacy while enabling powerful analytics.' },
    { topic: 'Materials Science', content: 'self-healing materials are now stable enough for real-world applications. Infrastructure that repairs itself could dramatically reduce maintenance costs.' }
  ];

  const generatePost = useCallback(async () => {
    const user = realUsers[Math.floor(Math.random() * realUsers.length)];
    const template = topicTemplates[Math.floor(Math.random() * topicTemplates.length)];
    
    const content = `Just witnessed a major ${template.content}`;
    
    const scored = significanceAlgorithm.scoreContent(content, 'social', 'Community Member');
    
    const post: Post = {
      id: Date.now().toString() + Math.random(),
      user,
      content,
      timestamp: `${Math.floor(Math.random() * 60) + 1}m`,
      likes: Math.floor(Math.random() * 2000) + 50,
      comments: Math.floor(Math.random() * 300) + 5,
      shares: Math.floor(Math.random() * 500) + 10,
      tags: [template.topic, scored.significanceScore > 8 ? 'High Impact' : 'Trending'],
      trending: scored.significanceScore > 7.5,
      significance: scored.significanceScore,
      category: template.topic,
      engagement: Math.floor(Math.random() * 100) + 50,
      quality: Math.floor(scored.significanceScore * 10)
    };

    return post;
  }, []);

  const loadPosts = useCallback(async () => {
    setLoading(true);
    const newPosts = await Promise.all(
      Array.from({ length: 8 }, () => generatePost())
    );
    setPosts(newPosts.sort((a, b) => b.significance - a.significance));
    setLoading(false);
  }, [generatePost]);

  useEffect(() => {
    loadPosts();
    const interval = setInterval(loadPosts, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [loadPosts]);

  const filteredPosts = posts.filter(post => {
    if (filter === 'trending') return post.trending;
    if (filter === 'high-impact') return post.significance > 8;
    return true;
  });

  const sortedPosts = [...filteredPosts].sort((a, b) => {
    if (sortBy === 'trending') return b.significance - a.significance;
    if (sortBy === 'engagement') return b.engagement - a.engagement;
    if (sortBy === 'recent') return parseInt(a.timestamp) - parseInt(b.timestamp);
    return b.significance - a.significance;
  });

  return (
    <div className="flex-1 bg-gray-950">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">Social Feed</h1>
              <p className="text-gray-400">Real-time insights • Community-driven • AI-curated content</p>
            </div>
            <div className="flex items-center space-x-3">
              <div className="px-4 py-2 bg-green-500/10 text-green-400 rounded-full text-sm font-medium border border-green-500/20 flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Live</span>
              </div>
            </div>
          </div>

          {/* Filters */}
          <div className="flex items-center justify-between">
            <div className="flex space-x-2">
              {[
                { id: 'all', name: 'All Posts', icon: Globe },
                { id: 'trending', name: 'Trending', icon: TrendingUp },
                { id: 'high-impact', name: 'High Impact', icon: Award }
              ].map((filterOption) => (
                <button
                  key={filterOption.id}
                  onClick={() => setFilter(filterOption.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
                    filter === filterOption.id
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 border border-gray-700/30'
                  }`}
                >
                  <filterOption.icon className="w-4 h-4" />
                  <span className="text-sm font-medium">{filterOption.name}</span>
                </button>
              ))}
            </div>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-2 text-white text-sm focus:outline-none focus:border-cyan-500/50"
            >
              <option value="trending">Sort by Significance</option>
              <option value="engagement">Sort by Engagement</option>
              <option value="recent">Sort by Recent</option>
            </select>
          </div>
        </div>
      </div>

      {/* Posts */}
      <div className="p-6 space-y-6">
        {loading ? (
          <div className="space-y-6">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="bg-gray-900/50 rounded-xl p-6 border border-gray-800/50 animate-pulse">
                <div className="flex items-center space-x-4 mb-4">
                  <div className="w-12 h-12 bg-gray-700 rounded-full"></div>
                  <div className="flex-1">
                    <div className="w-32 h-4 bg-gray-700 rounded mb-2"></div>
                    <div className="w-24 h-3 bg-gray-700 rounded"></div>
                  </div>
                </div>
                <div className="space-y-2 mb-4">
                  <div className="w-full h-4 bg-gray-700 rounded"></div>
                  <div className="w-3/4 h-4 bg-gray-700 rounded"></div>
                </div>
                <div className="flex space-x-4">
                  <div className="w-16 h-6 bg-gray-700 rounded"></div>
                  <div className="w-16 h-6 bg-gray-700 rounded"></div>
                  <div className="w-16 h-6 bg-gray-700 rounded"></div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          sortedPosts.map((post) => (
            <div key={post.id} className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-colors">
              {/* User Info */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-4">
                  <div className={`w-12 h-12 ${post.user.avatar} rounded-full flex items-center justify-center`}>
                    <span className="text-white font-bold text-lg">
                      {post.user.name.charAt(0)}
                    </span>
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <h3 className="font-semibold text-white">{post.user.name}</h3>
                      {post.user.verified && <Sparkles className="w-4 h-4 text-cyan-400" />}
                    </div>
                    <div className="flex items-center space-x-2">
                      <p className="text-gray-400 text-sm">@{post.user.username}</p>
                      <span className="text-gray-600">•</span>
                      <span className="text-gray-400 text-sm">{post.timestamp}</span>
                    </div>
                  </div>
                </div>
                
                {/* Significance Score */}
                <div className="flex items-center space-x-2">
                  <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                    post.significance > 8 
                      ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                      : post.significance > 7 
                      ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30'
                      : 'bg-gray-500/20 text-gray-400 border border-gray-500/30'
                  }`}>
                    {post.significance.toFixed(1)} Impact
                  </div>
                </div>
              </div>

              {/* User Badges */}
              <div className="flex flex-wrap gap-2 mb-4">
                {post.user.badges.map((badge, index) => (
                  <span key={index} className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-xs border border-purple-500/30">
                    {badge}
                  </span>
                ))}
              </div>

              {/* Content */}
              <div className="mb-4">
                <p className="text-gray-200 leading-relaxed">{post.content}</p>
              </div>

              {/* Tags */}
              <div className="flex flex-wrap gap-2 mb-4">
                {post.tags.map((tag, index) => (
                  <span key={index} className={`px-3 py-1 rounded-full text-xs font-medium ${
                    tag === 'High Impact' 
                      ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                      : tag === 'Trending'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'bg-gray-500/20 text-gray-300 border border-gray-500/30'
                  }`}>
                    #{tag}
                  </span>
                ))}
              </div>

              {/* Engagement */}
              <div className="flex items-center justify-between pt-4 border-t border-gray-800/50">
                <div className="flex items-center space-x-6">
                  <button className="flex items-center space-x-2 text-gray-400 hover:text-red-400 transition-colors">
                    <Heart className="w-5 h-5" />
                    <span className="text-sm">{post.likes.toLocaleString()}</span>
                  </button>
                  <button className="flex items-center space-x-2 text-gray-400 hover:text-blue-400 transition-colors">
                    <MessageCircle className="w-5 h-5" />
                    <span className="text-sm">{post.comments}</span>
                  </button>
                  <button className="flex items-center space-x-2 text-gray-400 hover:text-green-400 transition-colors">
                    <Share2 className="w-5 h-5" />
                    <span className="text-sm">{post.shares}</span>
                  </button>
                </div>
                <button className="text-gray-400 hover:text-yellow-400 transition-colors">
                  <Bookmark className="w-5 h-5" />
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
