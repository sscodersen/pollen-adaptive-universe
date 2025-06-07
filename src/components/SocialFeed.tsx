
import React, { useState, useEffect } from 'react';
import { Heart, MessageCircle, Share, Bookmark, MoreHorizontal, Verified, TrendingUp, Brain, Code, Lightbulb, Coffee, Music, Camera, Globe, Rocket } from 'lucide-react';

interface SocialPost {
  id: string;
  author: string;
  username: string;
  avatar: string;
  verified: boolean;
  timestamp: string;
  content: string;
  type: 'text' | 'image' | 'link' | 'poll' | 'thread' | 'quote' | 'achievement';
  likes: number;
  comments: number;
  shares: number;
  trending: boolean;
  category: string;
  media?: {
    type: 'image' | 'video' | 'link';
    url: string;
    thumbnail?: string;
    title?: string;
    description?: string;
  };
  poll?: {
    question: string;
    options: { text: string; votes: number }[];
    totalVotes: number;
  };
  achievement?: {
    title: string;
    description: string;
    icon: string;
    rarity: 'common' | 'rare' | 'epic' | 'legendary';
  };
}

export const SocialFeed = () => {
  const [posts, setPosts] = useState<SocialPost[]>([]);
  const [filter, setFilter] = useState('all');

  const postTemplates: Omit<SocialPost, 'id' | 'timestamp'>[] = [
    // AI & Tech Posts
    {
      author: "Alex Chen",
      username: "@alexchen_ai",
      avatar: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face",
      verified: true,
      content: "Just discovered a breakthrough in neural architecture search. The new transformer variant achieves 40% better efficiency while maintaining accuracy. The future of AI is getting more exciting every day! 🧠✨",
      type: "text",
      likes: 342,
      comments: 89,
      shares: 156,
      trending: true,
      category: "AI"
    },
    {
      author: "Sarah Kim",
      username: "@sarahbuilds",
      avatar: "https://images.unsplash.com/photo-1494790108755-2616b332b932?w=150&h=150&fit=crop&crop=face",
      verified: false,
      content: "Building my first neural network from scratch today. The math is beautiful but debugging is... an adventure 😅 Any tips for someone starting their ML journey?",
      type: "text",
      likes: 127,
      comments: 45,
      shares: 23,
      trending: false,
      category: "Learning"
    },
    // Creative & Design
    {
      author: "Maya Rodriguez",
      username: "@maya_creates",
      avatar: "https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=150&h=150&fit=crop&crop=face",
      verified: true,
      content: "AI-generated art collaboration: I prompted, refined, and curated this piece over 200 iterations. The relationship between human creativity and AI assistance is becoming more nuanced.",
      type: "image",
      likes: 892,
      comments: 234,
      shares: 445,
      trending: true,
      category: "Creative",
      media: {
        type: "image",
        url: "https://images.unsplash.com/photo-1547036967-23d11aacaee0?w=600&h=400&fit=crop",
        title: "AI-Human Collaborative Art"
      }
    },
    // Professional Development
    {
      author: "David Park",
      username: "@devdavid",
      avatar: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=150&h=150&fit=crop&crop=face",
      verified: false,
      content: "6 months into my career transition from finance to AI engineering. Key lessons: 1) Math fundamentals matter 2) Build projects, not just tutorials 3) The community is incredibly supportive",
      type: "thread",
      likes: 567,
      comments: 123,
      shares: 289,
      trending: false,
      category: "Career"
    },
    // Research & Science
    {
      author: "Dr. Lisa Chen",
      username: "@drlistech",
      avatar: "https://images.unsplash.com/photo-1560250097-0b93528c311a?w=150&h=150&fit=crop&crop=face",
      verified: true,
      content: "New paper published: 'Quantum-Inspired Optimization for Large Language Models' - We achieved 30% reduction in training time while improving performance. Open source implementation coming soon!",
      type: "link",
      likes: 1234,
      comments: 345,
      shares: 678,
      trending: true,
      category: "Research",
      media: {
        type: "link",
        url: "https://arxiv.org/abs/2023.12345",
        title: "Quantum-Inspired Optimization for LLMs",
        description: "A novel approach to accelerating language model training using quantum-inspired algorithms."
      }
    },
    // Casual/Personal
    {
      author: "Jordan Lee",
      username: "@jordancodes",
      avatar: "https://images.unsplash.com/photo-1519345182560-3f2917c472ef?w=150&h=150&fit=crop&crop=face",
      verified: false,
      content: "Coffee shop coding session complete ☕ Sometimes the best debugging happens away from your usual setup. What's your favorite place to code?",
      type: "text",
      likes: 234,
      comments: 67,
      shares: 45,
      trending: false,
      category: "Lifestyle"
    },
    // Poll Post
    {
      author: "Tech Community",
      username: "@techpoll",
      avatar: "https://images.unsplash.com/photo-1551434678-e076c223a692?w=150&h=150&fit=crop&crop=face",
      verified: true,
      content: "Which AI trend will have the biggest impact in 2024?",
      type: "poll",
      likes: 445,
      comments: 156,
      shares: 234,
      trending: true,
      category: "Discussion",
      poll: {
        question: "Which AI trend will have the biggest impact in 2024?",
        options: [
          { text: "Multimodal AI", votes: 234 },
          { text: "AI Agents", votes: 345 },
          { text: "Edge AI", votes: 123 },
          { text: "Quantum ML", votes: 89 }
        ],
        totalVotes: 791
      }
    },
    // Achievement Post
    {
      author: "Rachel Wang",
      username: "@rachelml",
      avatar: "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=150&h=150&fit=crop&crop=face",
      verified: false,
      content: "Just completed my first end-to-end ML pipeline deployment! From data collection to production monitoring. Feeling accomplished! 🎉",
      type: "achievement",
      likes: 678,
      comments: 234,
      shares: 123,
      trending: false,
      category: "Achievement",
      achievement: {
        title: "ML Pipeline Master",
        description: "Successfully deployed first production ML pipeline",
        icon: "🚀",
        rarity: "epic"
      }
    },
    // Music/Creative Tech
    {
      author: "Alex Sound",
      username: "@alexsound",
      avatar: "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150&h=150&fit=crop&crop=face",
      verified: false,
      content: "AI-generated beats are getting insane. Just fed my morning routine data into a neural network and it composed a perfect 'productivity flow' soundtrack. The future of personalized music is here! 🎵",
      type: "text",
      likes: 456,
      comments: 89,
      shares: 167,
      trending: false,
      category: "Music"
    }
  ];

  useEffect(() => {
    const generatePosts = () => {
      const shuffled = [...postTemplates]
        .sort(() => Math.random() - 0.5)
        .slice(0, 8)
        .map((post, index) => ({
          ...post,
          id: `post-${Date.now()}-${index}`,
          timestamp: new Date(Date.now() - Math.random() * 86400000 * 7).toISOString()
        }));
      setPosts(shuffled);
    };

    generatePosts();
  }, []);

  const categories = ['all', 'AI', 'Creative', 'Research', 'Career', 'Lifestyle', 'Discussion', 'Music'];

  const filteredPosts = filter === 'all' 
    ? posts 
    : posts.filter(post => post.category.toLowerCase() === filter.toLowerCase());

  const getCategoryIcon = (category: string) => {
    const icons = {
      'AI': Brain,
      'Creative': Camera,
      'Research': Globe,
      'Career': Rocket,
      'Lifestyle': Coffee,
      'Discussion': MessageCircle,
      'Music': Music,
      'Learning': Lightbulb
    };
    return icons[category as keyof typeof icons] || Brain;
  };

  const getRarityColor = (rarity: string) => {
    const colors = {
      'common': 'border-gray-500',
      'rare': 'border-blue-500',
      'epic': 'border-purple-500',
      'legendary': 'border-yellow-500'
    };
    return colors[rarity as keyof typeof colors] || 'border-gray-500';
  };

  const formatTimestamp = (timestamp: string) => {
    const now = new Date();
    const postTime = new Date(timestamp);
    const diff = now.getTime() - postTime.getTime();
    const hours = Math.floor(diff / (1000 * 60 * 60));
    
    if (hours < 1) return 'now';
    if (hours < 24) return `${hours}h`;
    return `${Math.floor(hours / 24)}d`;
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Social Feed</h2>
        <div className="flex items-center space-x-2">
          <TrendingUp className="w-5 h-5 text-cyan-400" />
          <span className="text-sm text-slate-400">Live Updates</span>
        </div>
      </div>

      {/* Category Filters */}
      <div className="flex space-x-2 overflow-x-auto pb-2">
        {categories.map((category) => {
          const IconComponent = getCategoryIcon(category);
          return (
            <button
              key={category}
              onClick={() => setFilter(category)}
              className={`flex items-center space-x-2 px-3 py-2 rounded-lg whitespace-nowrap transition-all ${
                filter === category
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700/50'
              }`}
            >
              <IconComponent className="w-4 h-4" />
              <span className="text-sm capitalize">{category}</span>
            </button>
          );
        })}
      </div>

      {/* Posts */}
      <div className="space-y-4">
        {filteredPosts.map((post) => (
          <div key={post.id} className="bg-slate-800/50 rounded-xl border border-slate-700/50 p-6 hover:bg-slate-800/70 transition-all">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                <img
                  src={post.avatar}
                  alt={post.author}
                  className="w-12 h-12 rounded-full object-cover"
                />
                <div>
                  <div className="flex items-center space-x-2">
                    <h3 className="font-semibold text-white">{post.author}</h3>
                    {post.verified && <Verified className="w-4 h-4 text-blue-400 fill-current" />}
                    {post.trending && <TrendingUp className="w-4 h-4 text-red-400" />}
                  </div>
                  <div className="flex items-center space-x-2 text-sm text-slate-400">
                    <span>{post.username}</span>
                    <span>•</span>
                    <span>{formatTimestamp(post.timestamp)}</span>
                    <span>•</span>
                    <span className="text-cyan-400">{post.category}</span>
                  </div>
                </div>
              </div>
              <button className="text-slate-400 hover:text-white">
                <MoreHorizontal className="w-5 h-5" />
              </button>
            </div>

            {/* Content */}
            <div className="mb-4">
              <p className="text-slate-200 leading-relaxed">{post.content}</p>

              {/* Media */}
              {post.media && (
                <div className="mt-4">
                  {post.media.type === 'image' && (
                    <img
                      src={post.media.url}
                      alt={post.media.title || 'Post image'}
                      className="w-full max-h-96 object-cover rounded-lg"
                    />
                  )}
                  {post.media.type === 'link' && (
                    <div className="border border-slate-600/50 rounded-lg p-4 bg-slate-700/30">
                      <h4 className="font-medium text-white mb-1">{post.media.title}</h4>
                      <p className="text-sm text-slate-400 mb-2">{post.media.description}</p>
                      <a 
                        href={post.media.url} 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-sm text-cyan-400 hover:text-cyan-300"
                      >
                        {post.media.url}
                      </a>
                    </div>
                  )}
                </div>
              )}

              {/* Poll */}
              {post.poll && (
                <div className="mt-4 space-y-3">
                  <h4 className="font-medium text-white">{post.poll.question}</h4>
                  {post.poll.options.map((option, index) => {
                    const percentage = Math.round((option.votes / post.poll!.totalVotes) * 100);
                    return (
                      <div key={index} className="relative">
                        <div className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg border border-slate-600/50 hover:bg-slate-600/30 cursor-pointer">
                          <span className="text-slate-200">{option.text}</span>
                          <span className="text-sm text-slate-400">{percentage}%</span>
                        </div>
                        <div 
                          className="absolute top-0 left-0 h-full bg-cyan-500/20 rounded-lg transition-all"
                          style={{ width: `${percentage}%` }}
                        />
                      </div>
                    );
                  })}
                  <p className="text-sm text-slate-400">{post.poll.totalVotes.toLocaleString()} votes</p>
                </div>
              )}

              {/* Achievement */}
              {post.achievement && (
                <div className={`mt-4 p-4 rounded-lg border-2 ${getRarityColor(post.achievement.rarity)} bg-gradient-to-r from-slate-800/50 to-slate-700/50`}>
                  <div className="flex items-center space-x-3">
                    <span className="text-2xl">{post.achievement.icon}</span>
                    <div>
                      <h4 className="font-medium text-white">{post.achievement.title}</h4>
                      <p className="text-sm text-slate-400">{post.achievement.description}</p>
                      <span className={`inline-block mt-1 px-2 py-1 rounded text-xs font-medium ${
                        post.achievement.rarity === 'legendary' ? 'bg-yellow-500/20 text-yellow-300' :
                        post.achievement.rarity === 'epic' ? 'bg-purple-500/20 text-purple-300' :
                        post.achievement.rarity === 'rare' ? 'bg-blue-500/20 text-blue-300' :
                        'bg-gray-500/20 text-gray-300'
                      }`}>
                        {post.achievement.rarity}
                      </span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Actions */}
            <div className="flex items-center justify-between pt-4 border-t border-slate-700/50">
              <div className="flex items-center space-x-6">
                <button className="flex items-center space-x-2 text-slate-400 hover:text-red-400 transition-colors">
                  <Heart className="w-5 h-5" />
                  <span className="text-sm">{post.likes}</span>
                </button>
                <button className="flex items-center space-x-2 text-slate-400 hover:text-cyan-400 transition-colors">
                  <MessageCircle className="w-5 h-5" />
                  <span className="text-sm">{post.comments}</span>
                </button>
                <button className="flex items-center space-x-2 text-slate-400 hover:text-green-400 transition-colors">
                  <Share className="w-5 h-5" />
                  <span className="text-sm">{post.shares}</span>
                </button>
              </div>
              <button className="text-slate-400 hover:text-yellow-400 transition-colors">
                <Bookmark className="w-5 h-5" />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
