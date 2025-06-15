import React, { useState, useEffect, useCallback } from 'react';
import { Eye, TrendingUp, Award, Zap, Users, Globe, Sparkles, Search, Star, Clock, Target } from 'lucide-react';
import { significanceAlgorithm } from '../services/significanceAlgorithm';
import { Input } from "@/components/ui/input";

interface SocialFeedProps {
  isGenerating?: boolean;
  filter?: string;
}

interface Post {
  id: string;
  user: {
    name: string;
    username: string;
    avatar: string;
    verified: boolean;
    badges: string[];
    rank: number;
  };
  content: string;
  timestamp: string;
  views: number;
  tags: string[];
  trending: boolean;
  significance: number;
  category: string;
  engagement: number;
  quality: number;
  type: 'social' | 'news' | 'discussion';
  impact: 'low' | 'medium' | 'high' | 'critical';
  readTime: string;
}

export const SocialFeed = ({ isGenerating = false, filter = "all" }: SocialFeedProps) => {
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  const realUsers = [
    { name: 'Dr. Sarah Chen', username: 'sarahchen_ai', avatar: 'bg-gradient-to-r from-blue-500 to-purple-500', verified: true, badges: ['AI Expert', 'Researcher'], rank: 98 },
    { name: 'Marcus Rodriguez', username: 'marcus_dev', avatar: 'bg-gradient-to-r from-green-500 to-blue-500', verified: true, badges: ['Developer', 'Open Source'], rank: 95 },
    { name: 'Elena Kowalski', username: 'elena_design', avatar: 'bg-gradient-to-r from-pink-500 to-red-500', verified: false, badges: ['Designer', 'UX'], rank: 87 },
    { name: 'Alex Chen', username: 'alex_crypto', avatar: 'bg-gradient-to-r from-yellow-500 to-orange-500', verified: false, badges: ['Blockchain', 'DeFi'], rank: 92 },
    { name: 'Maya Thompson', username: 'maya_startup', avatar: 'bg-gradient-to-r from-purple-500 to-pink-500', verified: true, badges: ['Entrepreneur', 'VC'], rank: 94 },
    { name: 'Global News Network', username: 'gnn_official', avatar: 'bg-gradient-to-r from-red-600 to-red-400', verified: true, badges: ['News', 'Breaking'], rank: 96 },
    { name: 'TechCrunch', username: 'techcrunch', avatar: 'bg-gradient-to-r from-emerald-500 to-cyan-500', verified: true, badges: ['Tech News', 'Verified'], rank: 97 },
    { name: 'The Economist', username: 'theeconomist', avatar: 'bg-gradient-to-r from-indigo-500 to-purple-500', verified: true, badges: ['Economics', 'Analysis'], rank: 99 },
    { name: 'Nature Journal', username: 'nature_journal', avatar: 'bg-gradient-to-r from-green-600 to-emerald-500', verified: true, badges: ['Science', 'Research'], rank: 99 },
    { name: 'World Health Org', username: 'who_official', avatar: 'bg-gradient-to-r from-blue-600 to-cyan-500', verified: true, badges: ['Health', 'Global'], rank: 98 }
  ];

  const broadPostTemplates = [
    // Global News & Current Events
    { topic: 'Climate Change', content: 'BREAKING: Antarctic ice shelf collapses faster than predicted. Scientists warn this could accelerate sea level rise by decades. The implications for coastal cities worldwide are staggering. 🌍❄️', category: 'News', type: 'news' as const },
    { topic: 'Space Exploration', content: 'NASA\'s James Webb telescope discovers potentially habitable exoplanet just 22 light-years away. The planet shows signs of water vapor and oxygen in its atmosphere. This could be humanity\'s first real chance at finding extraterrestrial life! 🚀🌌', category: 'Science', type: 'news' as const },
    { topic: 'Global Economy', content: 'Major economic shift: 15 countries announce plans to create new digital currency union, potentially challenging the dollar\'s dominance. Markets are volatile but many see this as inevitable evolution. 💰📊', category: 'Economics', type: 'news' as const },
    
    // Technology & Innovation
    { topic: 'AI Breakthrough', content: 'Incredible breakthrough in AI consciousness research! New neural architecture shows signs of genuine self-awareness and emotional responses. The ethical implications are mind-blowing. Are we on the verge of creating truly sentient beings? 🤖🧠', category: 'Technology', type: 'discussion' as const },
    { topic: 'Quantum Computing', content: 'Quantum computer successfully simulates complex molecular interactions, potentially revolutionizing drug discovery. What used to take years of research could now be done in hours. The future of medicine is here! ⚛️💊', category: 'Science', type: 'social' as const },
    { topic: 'Renewable Energy', content: 'Solar efficiency breakthrough: New perovskite cells achieve 47% efficiency in lab tests. If this scales, we could see solar power become cheaper than fossil fuels everywhere within 5 years. 🌞⚡', category: 'Environment', type: 'news' as const },
    
    // Social & Cultural
    { topic: 'Digital Rights', content: 'The debate over digital privacy has reached a tipping point. With AI systems knowing more about us than we know about ourselves, how do we maintain human autonomy? This isn\'t just about data - it\'s about the future of free will. 🔐👁️', category: 'Society', type: 'discussion' as const },
    { topic: 'Future of Work', content: 'Remote work isn\'t just a trend anymore - it\'s reshaping entire economies. Cities are depopulating while rural areas with good internet are booming. We\'re witnessing the largest migration pattern in human history, and it\'s digital. 🏠💻', category: 'Work', type: 'social' as const },
    { topic: 'Education Revolution', content: 'Traditional universities are struggling while online learning platforms explode. Students are getting world-class education for 1/10th the cost. Are we witnessing the end of the college system as we know it? 🎓📚', category: 'Education', type: 'discussion' as const },
    
    // Health & Science
    { topic: 'Longevity Research', content: 'Scientists successfully reverse aging in human cells using a combination of gene therapy and AI-designed compounds. Early trials show 20+ year cellular age reversal. We might be the first generation that doesn\'t have to age. 🧬⏰', category: 'Health', type: 'news' as const },
    { topic: 'Mental Health', content: 'Depression rates among young adults have tripled since 2019. But here\'s what\'s interesting: communities with strong digital wellness programs show 60% better outcomes. Maybe the solution to tech-induced problems is better tech. 🧠💚', category: 'Health', type: 'social' as const },
    { topic: 'Food Security', content: 'Lab-grown meat just achieved price parity with traditional meat in major markets. This isn\'t just about animal welfare anymore - it\'s about feeding 10 billion people without destroying the planet. 🥩🌱', category: 'Food Tech', type: 'news' as const },
    
    // Arts & Culture
    { topic: 'Digital Art', content: 'AI-generated art just sold for $69 million at Christie\'s. But here\'s the twist - the AI was trained exclusively on works by living artists who get royalties from every sale. Finally, technology that empowers creators instead of replacing them! 🎨🤖', category: 'Arts', type: 'social' as const },
    { topic: 'Virtual Reality', content: 'Spent 8 hours in a VR world today and honestly forgot it wasn\'t real. The haptic feedback, the visual fidelity, the social presence - we\'re not just building games anymore, we\'re building alternate realities. What does this mean for human experience? 🥽✨', category: 'VR/AR', type: 'social' as const },
    
    // Philosophy & Ethics
    { topic: 'AI Ethics', content: 'If an AI saves a human life by making a decision no human programmed it to make, who gets the credit? And if it makes a mistake, who\'s responsible? We\'re entering an era where our creations have agency, and our legal systems aren\'t ready. ⚖️🤖', category: 'Philosophy', type: 'discussion' as const },
    { topic: 'Human Enhancement', content: 'Brain-computer interfaces now allow paralyzed patients to control computers with thought alone. But the same technology could enhance healthy brains. At what point do we stop being human and start being something else? 🧠⚡', category: 'Biotech', type: 'discussion' as const },
    
    // Global Issues
    { topic: 'Water Crisis', content: 'Atmospheric water generators now cost less than drilling wells in many regions. Desert communities are becoming self-sufficient in water for the first time in history. Technology solving what politics couldn\'t. 💧🏜️', category: 'Environment', type: 'news' as const },
    { topic: 'Immigration', content: 'Digital nomad visas are reshaping global migration. Countries are competing for remote workers like they used to compete for factories. Talent is becoming truly global, and borders are becoming more fluid. 🌍✈️', category: 'Society', type: 'social' as const }
  ];

  const generatePost = useCallback(async () => {
    const user = realUsers[Math.floor(Math.random() * realUsers.length)];
    const template = broadPostTemplates[Math.floor(Math.random() * broadPostTemplates.length)];
    
    const scored = significanceAlgorithm.scoreContent(template.content, 'social', user.name);
    
    const post: Post = {
      id: Date.now().toString() + Math.random(),
      user,
      content: template.content,
      timestamp: `${Math.floor(Math.random() * 240) + 1}m`,
      views: Math.floor(Math.random() * 50000) + 1000,
      tags: [template.topic, template.category, scored.significanceScore > 8 ? 'High Impact' : 'Trending'],
      trending: scored.significanceScore > 7.5,
      significance: scored.significanceScore,
      category: template.category,
      engagement: Math.floor(Math.random() * 100) + 50,
      quality: Math.floor(scored.significanceScore * 10),
      type: template.type,
      impact: scored.significanceScore > 9 ? 'critical' : scored.significanceScore > 8 ? 'high' : scored.significanceScore > 6.5 ? 'medium' : 'low',
      readTime: `${Math.floor(Math.random() * 5) + 1} min read`
    };

    return post;
  }, []);

  const loadPosts = useCallback(async () => {
    setLoading(true);
    const newPosts = await Promise.all(
      Array.from({ length: 20 }, () => generatePost())
    );
    setPosts(newPosts.sort((a, b) => b.significance - a.significance));
    setLoading(false);
  }, [generatePost]);

  useEffect(() => {
    loadPosts();
    const interval = setInterval(loadPosts, 45000);
    return () => clearInterval(interval);
  }, [loadPosts]);

  const filteredPosts = posts.filter(post => {
    const matchesFilter = filter === 'all' || 
      (filter === 'trending' && post.trending) || 
      (filter === 'high-impact' && post.significance > 8);
    
    const matchesSearch = !searchQuery || 
      post.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
      post.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase())) ||
      post.category.toLowerCase().includes(searchQuery.toLowerCase());
    
    return matchesFilter && matchesSearch;
  });

  const getImpactColor = (impact: string) => {
    switch (impact) {
      case 'critical': return 'bg-red-500/20 text-red-300 border-red-500/30';
      case 'high': return 'bg-orange-500/20 text-orange-300 border-orange-500/30';
      case 'medium': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30';
    }
  };

  const getRankBadge = (rank: number) => {
    if (rank >= 95) return 'bg-gradient-to-r from-yellow-500 to-orange-500 text-white';
    if (rank >= 90) return 'bg-gradient-to-r from-purple-500 to-pink-500 text-white';
    if (rank >= 80) return 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white';
    return 'bg-gradient-to-r from-gray-600 to-gray-500 text-white';
  };

  return (
    <div className="flex-1 bg-gray-950">
      {/* Header with Search */}
      <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                <Globe className="w-8 h-8 text-cyan-400" />
                Global Feed
              </h1>
              <p className="text-gray-400">Real-time insights • Ranked content • AI-curated quality</p>
            </div>
            <div className="flex items-center space-x-3">
              <div className="px-4 py-2 bg-green-500/10 text-green-400 rounded-full text-sm font-medium border border-green-500/20 flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Live Ranking</span>
              </div>
            </div>
          </div>
          
          {/* Search Bar */}
          <div className="relative max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              type="text"
              placeholder="Search by topic, category, or impact level..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 bg-gray-800/50 border-gray-700 text-white placeholder-gray-400 focus:border-cyan-500"
            />
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
              </div>
            ))}
          </div>
        ) : (
          filteredPosts.map((post, index) => (
            <div key={post.id} className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-all group relative overflow-hidden">
              {/* Ranking Badge */}
              <div className="absolute top-4 right-4 flex items-center space-x-2">
                <div className="px-3 py-1 bg-gray-800/80 text-gray-300 rounded-full text-xs font-bold border border-gray-700">
                  #{index + 1}
                </div>
              </div>

              {/* User Info with Enhanced Ranking */}
              <div className="flex items-center justify-between mb-4 mr-16">
                <div className="flex items-center space-x-4">
                  <div className={`w-12 h-12 ${post.user.avatar} rounded-full flex items-center justify-center relative`}>
                    <span className="text-white font-bold text-lg">
                      {post.user.name.charAt(0)}
                    </span>
                    {/* User Rank Indicator */}
                    <div className={`absolute -bottom-1 -right-1 w-6 h-6 ${getRankBadge(post.user.rank)} rounded-full flex items-center justify-center text-xs font-bold`}>
                      {post.user.rank}
                    </div>
                  </div>
                  <div>
                    <div className="flex items-center space-x-2">
                      <h3 className="font-semibold text-white">{post.user.name}</h3>
                      {post.user.verified && <Sparkles className="w-4 h-4 text-cyan-400" />}
                      <span className={`px-2 py-0.5 text-xs rounded font-medium ${
                        post.type === 'news' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                        post.type === 'discussion' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30' :
                        'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                      }`}>
                        {post.type.toUpperCase()}
                      </span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm">
                      <p className="text-gray-400">@{post.user.username}</p>
                      <span className="text-gray-600">•</span>
                      <span className="text-gray-400">{post.timestamp}</span>
                      <span className="text-gray-600">•</span>
                      <div className="flex items-center space-x-1 text-gray-400">
                        <Clock className="w-3 h-3" />
                        <span>{post.readTime}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Enhanced Badges */}
              <div className="flex flex-wrap gap-2 mb-4">
                {post.user.badges.map((badge, index) => (
                  <span key={index} className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-xs border border-purple-500/30 font-medium">
                    {badge}
                  </span>
                ))}
                <div className={`px-2 py-1 rounded text-xs font-bold border ${getImpactColor(post.impact)}`}>
                  <Target className="w-3 h-3 inline mr-1" />
                  {post.impact.toUpperCase()} IMPACT
                </div>
              </div>

              {/* Content */}
              <div className="mb-4">
                <p className="text-gray-200 leading-relaxed">{post.content}</p>
              </div>

              {/* Enhanced Tags */}
              <div className="flex flex-wrap gap-2 mb-4">
                {post.tags.map((tag, index) => (
                  <span key={index} className={`px-3 py-1 rounded-full text-xs font-medium border ${
                    tag === 'High Impact' 
                      ? 'bg-red-500/20 text-red-300 border-red-500/30'
                      : tag === 'Trending'
                      ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'
                      : 'bg-gray-500/20 text-gray-300 border-gray-500/30'
                  }`}>
                    #{tag}
                  </span>
                ))}
              </div>

              {/* Stats and Significance */}
              <div className="flex items-center justify-between pt-4 border-t border-gray-800/50">
                <div className="flex items-center space-x-6">
                  <div className="flex items-center space-x-2 text-gray-400 hover:text-cyan-400 transition-colors">
                    <Eye className="w-5 h-5" />
                    <span className="text-sm font-medium">{post.views.toLocaleString()}</span>
                    <span className="text-xs text-gray-500">views</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className={`px-3 py-1 rounded-full text-xs font-bold flex items-center space-x-1 ${
                      post.significance > 9 
                        ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                        : post.significance > 8 
                        ? 'bg-orange-500/20 text-orange-300 border border-orange-500/30'
                        : post.significance > 7 
                        ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30'
                        : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                    }`}>
                      <Award className="w-3 h-3" />
                      <span>{post.significance.toFixed(1)} Significance</span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="px-2 py-1 bg-gray-700/50 text-gray-300 rounded text-xs">
                    Quality: {post.quality}/100
                  </div>
                  {post.trending && (
                    <div className="flex items-center space-x-1 px-2 py-1 bg-red-500/20 text-red-300 rounded text-xs border border-red-500/30 animate-pulse">
                      <TrendingUp className="w-3 h-3" />
                      <span>Trending</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
