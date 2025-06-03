
import React, { useState, useEffect } from 'react';
import { Heart, MessageCircle, Share2, MoreHorizontal, Eye, Sparkles, TrendingUp } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';

interface Post {
  id: string;
  author: string;
  avatar: string;
  content: string;
  images?: string[];
  likes: number;
  comments: number;
  views: number;
  timestamp: string;
  type: 'text' | 'image' | 'video' | 'mixed';
  trending?: boolean;
}

interface SocialFeedProps {
  isGenerating?: boolean;
}

export const SocialFeed = ({ isGenerating = true }: SocialFeedProps) => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [generatingPost, setGeneratingPost] = useState(false);

  useEffect(() => {
    if (!isGenerating) return;

    const generateContent = async () => {
      if (generatingPost) return;
      
      setGeneratingPost(true);
      try {
        const topics = [
          'emerging technology trends',
          'creative breakthrough insights',
          'future of human-AI collaboration',
          'innovation in digital experiences',
          'patterns in collective intelligence'
        ];
        
        const randomTopic = topics[Math.floor(Math.random() * topics.length)];
        
        const response = await pollenAI.generate(
          `Create engaging social content about ${randomTopic}`,
          "social"
        );
        
        const newPost: Post = {
          id: Date.now().toString(),
          author: generateRandomAuthor(),
          avatar: generateAvatar(),
          content: response.content,
          likes: Math.floor(Math.random() * 2000),
          comments: Math.floor(Math.random() * 200),
          views: Math.floor(Math.random() * 10000),
          timestamp: formatTimestamp(new Date()),
          type: 'text',
          trending: Math.random() > 0.7
        };
        
        setPosts(prev => [newPost, ...prev.slice(0, 19)]);
      } catch (error) {
        console.error('Failed to generate content:', error);
      }
      setGeneratingPost(false);
    };

    generateContent();
    const interval = setInterval(generateContent, 25000);
    return () => clearInterval(interval);
  }, [isGenerating, generatingPost]);

  const generateRandomAuthor = () => {
    const names = [
      'Alex Chen', 'Maya Rodriguez', 'Jordan Smith', 'Riley Kim', 
      'Sam Johnson', 'Casey Brown', 'Avery Taylor', 'Quinn Davis'
    ];
    return names[Math.floor(Math.random() * names.length)];
  };

  const generateAvatar = () => {
    const colors = [
      'bg-blue-500', 'bg-purple-500', 'bg-pink-500', 'bg-green-500', 
      'bg-orange-500', 'bg-cyan-500', 'bg-yellow-500', 'bg-red-500'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  const formatTimestamp = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'now';
    if (diffMins < 60) return `${diffMins}m`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h`;
    return `${Math.floor(diffMins / 1440)}d`;
  };

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Social Feed</h1>
            <p className="text-gray-400">AI-generated content continuously evolving</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingPost && (
              <div className="flex items-center space-x-2 text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm">Generating...</span>
              </div>
            )}
            <span className="text-sm text-gray-400">{posts.length} posts</span>
          </div>
        </div>

        {/* Filter Tabs */}
        <div className="flex space-x-6">
          {['Recent', 'Trending', 'Following', 'Discover'].map((tab, index) => (
            <button
              key={tab}
              className={`pb-2 text-sm font-medium transition-colors border-b-2 ${
                index === 0
                  ? 'text-blue-400 border-blue-400'
                  : 'text-gray-400 border-transparent hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Feed */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {posts.map((post) => (
          <div key={post.id} className="bg-gray-800/50 rounded-2xl border border-gray-700/50 p-6 hover:bg-gray-800/70 transition-colors">
            {/* Post Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className={`w-12 h-12 ${post.avatar} rounded-full flex items-center justify-center text-white font-medium text-lg`}>
                  {post.author.split(' ').map(n => n[0]).join('')}
                </div>
                <div>
                  <div className="flex items-center space-x-2">
                    <h3 className="font-semibold text-white">{post.author}</h3>
                    {post.trending && (
                      <div className="flex items-center space-x-1 px-2 py-0.5 bg-orange-500/20 text-orange-400 text-xs rounded-full">
                        <TrendingUp className="w-3 h-3" />
                        <span>Trending</span>
                      </div>
                    )}
                  </div>
                  <p className="text-sm text-gray-400">{post.timestamp}</p>
                </div>
              </div>
              <button className="text-gray-400 hover:text-white transition-colors">
                <MoreHorizontal className="w-5 h-5" />
              </button>
            </div>

            {/* Post Content */}
            <div className="mb-6">
              <p className="text-gray-100 leading-relaxed whitespace-pre-line">{post.content}</p>
            </div>

            {/* Post Actions */}
            <div className="flex items-center justify-between pt-4 border-t border-gray-700/50">
              <div className="flex items-center space-x-6">
                <button className="flex items-center space-x-2 text-gray-400 hover:text-red-400 transition-colors group">
                  <Heart className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  <span className="text-sm font-medium">{post.likes.toLocaleString()}</span>
                </button>
                <button className="flex items-center space-x-2 text-gray-400 hover:text-blue-400 transition-colors group">
                  <MessageCircle className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  <span className="text-sm font-medium">{post.comments}</span>
                </button>
                <button className="flex items-center space-x-2 text-gray-400 hover:text-green-400 transition-colors group">
                  <Share2 className="w-5 h-5 group-hover:scale-110 transition-transform" />
                  <span className="text-sm font-medium">Share</span>
                </button>
              </div>
              <div className="flex items-center space-x-2 text-gray-400">
                <Eye className="w-4 h-4" />
                <span className="text-sm">{post.views.toLocaleString()}</span>
              </div>
            </div>
          </div>
        ))}

        {posts.length === 0 && (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-gradient-to-r from-blue-400 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-6">
              <Sparkles className="w-10 h-10 text-white animate-pulse" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Pollen is Thinking...</h3>
            <p className="text-gray-400 max-w-md mx-auto">
              Our AI is analyzing patterns and generating engaging social content tailored to emerging trends and community interests.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
