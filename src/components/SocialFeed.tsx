import React, { useState, useEffect, useCallback } from 'react';
import { Eye, TrendingUp, Award, Zap, Users, Globe, Sparkles, Search, Star, Clock, Target } from 'lucide-react';
import { contentOrchestrator } from '../services/contentOrchestrator';
import { SocialContent } from '../services/unifiedContentEngine';
import { enhancedTrendEngine, GeneratedPost } from '../services/enhancedTrendEngine';
import { Input } from "@/components/ui/input";

import { cleanText, truncateText, normalizeTags } from '@/lib/textUtils';
import { isBlacklistedText } from '@/lib/blacklist';

// IMPORTANT: src/components/SocialFeed.tsx is 263 lines long and should be refactored into smaller components

interface SocialFeedProps {
  activities?: SocialContent[];
  isGenerating?: boolean;
  filter?: string;
}

// Using SocialContent from unified content engine

export const SocialFeed = ({ activities, isGenerating = false, filter = "all" }: SocialFeedProps) => {
  const [posts, setPosts] = useState<SocialContent[]>(activities || []);
  const [trendPosts, setTrendPosts] = useState<GeneratedPost[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  // Removed hardcoded data - now using unified content engine

  // Removed hardcoded templates - now using AI-generated content

  // Using unified content engine now

  const loadPosts = useCallback(async () => {
    if (activities && activities.length > 0) {
      setPosts(activities);
      setLoading(false);
      return;
    }
    
    setLoading(true);
    try {
      const strategy = {
        diversity: 0.9,
        freshness: 0.8,
        personalization: 0.6,
        qualityThreshold: 6.0,
        trendingBoost: 1.5
      };
      
      const { content: newPosts } = await contentOrchestrator.generateContent({
        type: 'social',
        count: 15,
        strategy
      });
      const trendGeneratedPosts = enhancedTrendEngine.getGeneratedPosts().slice(0, 10);
      setPosts(newPosts as SocialContent[]);
      setTrendPosts(trendGeneratedPosts);
    } catch (error) {
      console.error('Failed to load posts:', error);
    }
    setLoading(false);
  }, [activities]);

  useEffect(() => {
    loadPosts();

    // Prime trend posts immediately
    setTrendPosts(enhancedTrendEngine.getGeneratedPosts().slice(0, 10));

    // Subscribe to trend engine updates
    const unsubscribe = enhancedTrendEngine.subscribe((data) => {
      if (data.type === 'posts_generated' || data.type === 'manual_content_generated') {
        setTrendPosts(enhancedTrendEngine.getGeneratedPosts().slice(0, 10));
      }
    });
    
    const interval = setInterval(loadPosts, 45000);
    
    return () => {
      clearInterval(interval);
      unsubscribe();
    };
  }, [loadPosts]);

  // Generate more original, less templated copy for trend posts
  const hashString = (s: string) => {
    let h = 0; for (let i = 0; i < s.length; i++) { h = ((h << 5) - h) + s.charCodeAt(i); h |= 0; }
    return Math.abs(h);
  };

  const makeOriginalSummary = (p: GeneratedPost): string => {
    const impact = p.engagement_score > 80 ? 'high' : p.engagement_score > 60 ? 'medium' : 'rising';
    // Remove leading sources in parentheses
    const stripped = (p.content || '').replace(/^\([^)]*\)\s*/,'');
    const cleaned = cleanText(stripped).replace(/\s*(is\s+trending|is\s+surging)[.!…]*$/i, '');
    const clip = truncateText(cleaned, 140);
    const tags = p.hashtags.slice(0, 2).join(' ');
    const options = [
      `${p.topic}: ${clip}`.trim(),
      `Signal check • ${p.topic}. ${tags}`.trim(),
      `${p.topic} — ${clip}`.trim(),
    ];
    return options[hashString(p.id) % options.length];
  };

  // Convert trend posts to social content format for display
  const convertTrendPostsToSocialContent = (trendPosts: GeneratedPost[]): SocialContent[] => {
    return trendPosts.map(post => ({
      id: post.id,
      type: 'social' as const,
      title: post.topic,
      description: makeOriginalSummary(post),
      content: makeOriginalSummary(post),
      category: 'trending',
      user: {
        name: 'TrendBot',
        username: 'trendbot',
        avatar: 'bg-gradient-to-r from-purple-500 to-pink-500',
        verified: true,
        rank: 98,
        badges: ['AI Generated', 'Trending']
      },
      timestamp: new Date(post.timestamp).toLocaleString(),
      views: Math.floor(Math.random() * 5000) + 1000,
      engagement: Math.floor(Math.random() * 1000) + 500,
      significance: post.engagement_score / 10,
      quality: Math.floor(post.engagement_score),
      trending: true,
      impact: post.engagement_score > 80 ? 'high' : post.engagement_score > 60 ? 'medium' : 'low',
      contentType: post.type === 'social' ? 'social' : post.type === 'news' ? 'news' : 'discussion',
      tags: post.hashtags.map(h => h.replace('#', '')),
      readTime: '2 min'
    }));
  };

  // Combine regular posts with trend-generated posts
  const allPosts = [
    ...convertTrendPostsToSocialContent(trendPosts),
    ...posts
  ];

  const nonBlacklistedPosts = allPosts.filter(p => !isBlacklistedText(p.title) && !isBlacklistedText(p.description) && !p.tags.some(t => isBlacklistedText(t)));


  const filteredPosts = nonBlacklistedPosts.filter(post => {
    const matchesFilter = filter === 'all' || 
      (filter === 'trending' && post.trending) || 
      (filter === 'high-impact' && post.significance > 8);
    
    const matchesSearch = !searchQuery || 
      post.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      post.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase())) ||
      post.category.toLowerCase().includes(searchQuery.toLowerCase()) ||
      post.user.name.toLowerCase().includes(searchQuery.toLowerCase());
    
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
    <div className="flex-1 bg-background min-h-0 flex flex-col">
      {/* Header with Search */}
      <div className="sticky top-0 z-10 bg-card backdrop-blur-sm border-b border-border">
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
      <div className="flex-1 overflow-auto p-6 space-y-6">
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
                    post.contentType === 'news' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                    post.contentType === 'discussion' ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30' :
                    'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                  }`}>
                    {post.contentType.toUpperCase()}
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
                <p className="text-gray-200 leading-relaxed line-clamp-3">{truncateText(cleanText(post.description), 280)}</p>
              </div>

              {/* Enhanced Tags */}
              <div className="flex flex-wrap gap-2 mb-4">
                {normalizeTags(post.tags).map((tag, index) => (
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
