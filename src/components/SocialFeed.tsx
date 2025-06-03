
import React, { useState, useEffect } from 'react';
import { Heart, MessageCircle, Share2, MoreHorizontal, Eye } from 'lucide-react';
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
}

export const SocialFeed = () => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  // Continuous content generation
  useEffect(() => {
    const generateContent = async () => {
      if (isGenerating) return;
      
      setIsGenerating(true);
      try {
        const response = await pollenAI.generate(
          "Generate a creative social media post with engaging content",
          "creative"
        );
        
        const newPost: Post = {
          id: Date.now().toString(),
          author: generateRandomAuthor(),
          avatar: generateAvatar(),
          content: response.content,
          likes: Math.floor(Math.random() * 1000),
          comments: Math.floor(Math.random() * 100),
          views: Math.floor(Math.random() * 5000),
          timestamp: new Date().toLocaleTimeString(),
          type: 'text'
        };
        
        setPosts(prev => [newPost, ...prev.slice(0, 19)]); // Keep last 20 posts
      } catch (error) {
        console.error('Failed to generate content:', error);
      }
      setIsGenerating(false);
    };

    // Generate initial content
    generateContent();
    
    // Generate new content every 30 seconds
    const interval = setInterval(generateContent, 30000);
    return () => clearInterval(interval);
  }, []);

  const generateRandomAuthor = () => {
    const names = ['Alex Chen', 'Maya Rodriguez', 'Jordan Smith', 'Riley Kim', 'Sam Johnson', 'Casey Brown'];
    return names[Math.floor(Math.random() * names.length)];
  };

  const generateAvatar = () => {
    const colors = ['bg-cyan-500', 'bg-purple-500', 'bg-pink-500', 'bg-green-500', 'bg-orange-500', 'bg-blue-500'];
    return colors[Math.floor(Math.random() * colors.length)];
  };

  return (
    <div className="flex-1 overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b border-white/10 bg-black/20 backdrop-blur-xl">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">Social Feed</h1>
            <p className="text-white/60">AI-generated content continuously evolving</p>
          </div>
          <div className="flex items-center space-x-2">
            {isGenerating && (
              <div className="flex items-center space-x-2 text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm">Generating...</span>
              </div>
            )}
          </div>
        </div>
        
        {/* Tabs */}
        <div className="flex space-x-6 mt-4">
          {['Recents', 'Friends', 'Popular', 'Stories'].map((tab) => (
            <button
              key={tab}
              className={`pb-2 text-sm font-medium transition-colors ${
                tab === 'Recents' 
                  ? 'text-white border-b-2 border-cyan-400' 
                  : 'text-white/60 hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Feed */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {posts.map((post) => (
          <div key={post.id} className="bg-white/10 backdrop-blur-sm rounded-2xl border border-white/10 p-6">
            {/* Post Header */}
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className={`w-10 h-10 ${post.avatar} rounded-full flex items-center justify-center text-white font-medium`}>
                  {post.author.split(' ').map(n => n[0]).join('')}
                </div>
                <div>
                  <h3 className="font-semibold text-white">{post.author}</h3>
                  <p className="text-xs text-white/60">{post.timestamp}</p>
                </div>
              </div>
              <button className="text-white/60 hover:text-white">
                <MoreHorizontal className="w-5 h-5" />
              </button>
            </div>

            {/* Post Content */}
            <div className="mb-4">
              <p className="text-white/90 leading-relaxed">{post.content}</p>
            </div>

            {/* Post Actions */}
            <div className="flex items-center justify-between pt-4 border-t border-white/10">
              <div className="flex items-center space-x-6">
                <button className="flex items-center space-x-2 text-white/60 hover:text-red-400 transition-colors">
                  <Heart className="w-5 h-5" />
                  <span className="text-sm">{post.likes}</span>
                </button>
                <button className="flex items-center space-x-2 text-white/60 hover:text-blue-400 transition-colors">
                  <MessageCircle className="w-5 h-5" />
                  <span className="text-sm">{post.comments}</span>
                </button>
                <button className="flex items-center space-x-2 text-white/60 hover:text-green-400 transition-colors">
                  <Share2 className="w-5 h-5" />
                  <span className="text-sm">Share</span>
                </button>
              </div>
              <div className="flex items-center space-x-2 text-white/60">
                <Eye className="w-4 h-4" />
                <span className="text-sm">{post.views}</span>
              </div>
            </div>
          </div>
        ))}

        {posts.length === 0 && (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full flex items-center justify-center mx-auto mb-4">
              <Sparkles className="w-8 h-8 text-white animate-pulse" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Pollen is Thinking...</h3>
            <p className="text-white/60">Generating your first social content</p>
          </div>
        )}
      </div>
    </div>
  );
};
