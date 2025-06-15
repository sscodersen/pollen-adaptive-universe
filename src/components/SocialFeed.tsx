import React, { useState, useEffect, useCallback } from 'react';
import { Heart, MessageCircle, Share2, Bookmark, TrendingUp, Award, Zap, Users, Globe, Sparkles } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

interface SocialFeedProps {
  isGenerating?: boolean;
  filter?: string; // Pass filter from MainTabs, not local state
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

export const SocialFeed = ({ isGenerating = false, filter = "all" }: SocialFeedProps) => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [sortBy, setSortBy] = useState('trending');

  const realUsers = [
    { name: 'Dr. Sarah Chen', username: 'sarahchen_ai', avatar: 'bg-gradient-to-r from-blue-500 to-purple-500', verified: true, badges: ['AI Expert', 'Researcher'] },
    { name: 'Marcus Rodriguez', username: 'marcus_dev', avatar: 'bg-gradient-to-r from-green-500 to-blue-500', verified: true, badges: ['Developer', 'Open Source'] },
    { name: 'Elena Kowalski', username: 'elena_design', avatar: 'bg-gradient-to-r from-pink-500 to-red-500', verified: false, badges: ['Designer'] },
    { name: 'Alex Chen', username: 'alex_crypto', avatar: 'bg-gradient-to-r from-yellow-500 to-orange-500', verified: false, badges: ['Blockchain'] },
    { name: 'Maya Thompson', username: 'maya_startup', avatar: 'bg-gradient-to-r from-purple-500 to-pink-500', verified: true, badges: ['Entrepreneur'] },
    { name: 'James Wilson', username: 'james_fitness', avatar: 'bg-gradient-to-r from-red-500 to-orange-500', verified: false, badges: ['Fitness Coach'] },
    { name: 'Lisa Zhang', username: 'lisa_finance', avatar: 'bg-gradient-to-r from-emerald-500 to-cyan-500', verified: true, badges: ['Finance Expert'] },
    { name: 'David Park', username: 'david_travel', avatar: 'bg-gradient-to-r from-indigo-500 to-purple-500', verified: false, badges: ['Travel Blogger'] }
  ];

  const postTemplates = [
    // Tech & Innovation Posts
    { topic: 'AI Research', content: 'Just witnessed a breakthrough in neural architecture search that reduces training time by 60% while improving accuracy. The implications for democratizing AI development are huge! 🚀', category: 'Technology' },
    { topic: 'Climate Tech', content: 'Amazing news! New carbon capture technology just hit a major milestone - 1 million tons of CO2 removed from atmosphere while being economically viable. This could be a game changer! 🌱', category: 'Climate' },
    { topic: 'Quantum Computing', content: 'Quantum error correction breakthrough! Stable qubits maintaining coherence for over 100 seconds. We\'re getting closer to practical quantum computers every day! ⚛️', category: 'Science' },
    
    // Lifestyle & Personal Posts
    { topic: 'Productivity', content: 'Finally found the perfect morning routine! 6am workout, 30min meditation, then deep work blocks. Productivity up 40% this month. What works for you? ☀️', category: 'Lifestyle' },
    { topic: 'Learning', content: 'Been learning Python for 3 months now. Just built my first machine learning model! The feeling when your code actually works is unmatched 😄', category: 'Education' },
    { topic: 'Travel', content: 'Just got back from Iceland! The Northern Lights were incredible. Sometimes you need to disconnect from tech and reconnect with nature 🌌', category: 'Travel' },
    
    // Industry & Business Posts
    { topic: 'Startup Life', content: 'Month 6 of building our startup. Learned more about customer discovery in the past week than in my entire MBA. Talk to your users, early and often! 💡', category: 'Business' },
    { topic: 'Remote Work', content: 'Unpopular opinion: The best remote work setup isn\'t about the fanciest gear, it\'s about clear boundaries and communication. Less Slack, more focus time! 🏠', category: 'Work' },
    { topic: 'Investment', content: 'Interesting trend: 73% of Gen Z is investing in sustainable ETFs. The next generation is literally putting their money where their values are 📈', category: 'Finance' },
    
    // Creative & Arts Posts
    { topic: 'Design', content: 'Working on UI for accessibility has completely changed how I think about design. Good design isn\'t just beautiful, it\'s inclusive. Every pixel should have purpose ✨', category: 'Design' },
    { topic: 'Music', content: 'Discovered this amazing artist who creates music entirely with AI tools, then performs it live with traditional instruments. The future of creativity is collaborative! 🎵', category: 'Arts' },
    { topic: 'Photography', content: 'Spent the weekend doing street photography with just my phone. Sometimes constraints breed the most creativity. The best camera is the one you have with you 📸', category: 'Arts' },
    
    // Social & Community Posts
    { topic: 'Community', content: 'Organized my first local tech meetup! 50 people showed up to talk about AI ethics. There\'s so much hunger for meaningful conversations about technology\'s impact 🤝', category: 'Community' },
    { topic: 'Volunteering', content: 'Teaching coding to kids at the local library. Their questions are so much better than any interview I\'ve ever had. \'Why do computers only understand 1s and 0s?\' 👨‍🏫', category: 'Education' },
    { topic: 'Wellness', content: 'Week 2 of digital detox evenings. Reading actual books, having real conversations. My brain feels less scattered. Maybe we don\'t need to be always-on? 🧘‍♀️', category: 'Wellness' }
  ];

  const generatePost = useCallback(async () => {
    const user = realUsers[Math.floor(Math.random() * realUsers.length)];
    const template = postTemplates[Math.floor(Math.random() * postTemplates.length)];
    
    const scored = significanceAlgorithm.scoreContent(template.content, 'social', 'Community Member');
    
    const post: Post = {
      id: Date.now().toString() + Math.random(),
      user,
      content: template.content,
      timestamp: `${Math.floor(Math.random() * 60) + 1}m`,
      likes: Math.floor(Math.random() * 2000) + 50,
      comments: Math.floor(Math.random() * 300) + 5,
      shares: Math.floor(Math.random() * 500) + 10,
      tags: [template.topic, template.category, scored.significanceScore > 8 ? 'High Impact' : 'Trending'],
      trending: scored.significanceScore > 7.5,
      significance: scored.significanceScore,
      category: template.category,
      engagement: Math.floor(Math.random() * 100) + 50,
      quality: Math.floor(scored.significanceScore * 10)
    };

    return post;
  }, []);

  const loadPosts = useCallback(async () => {
    setLoading(true);
    const newPosts = await Promise.all(
      Array.from({ length: 12 }, () => generatePost())
    );
    setPosts(newPosts.sort((a, b) => b.significance - a.significance));
    setLoading(false);
  }, [generatePost]);

  useEffect(() => {
    loadPosts();
    const interval = setInterval(loadPosts, 30000);
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
        </div>
      </div>

      {/* Posts */}
      <div className="p-6 space-y-6">
        {loading ? (
          <div className="space-y-6">
            {[...Array(8)].map((_, i) => (
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
