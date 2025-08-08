
import React, { useState, useEffect, useCallback } from 'react';
import { Search, TrendingUp, Compass, Filter, Globe, Zap, Users, Calendar, RefreshCw } from 'lucide-react';
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { enhancedTrendEngine, TrendData } from '../services/enhancedTrendEngine';
import { trendAggregator } from '../services/trendAggregator';
import { cleanText, truncateText } from '@/lib/textUtils';


const discoveryCategories = [
  { name: 'Science & Research', icon: Zap, count: 2340, color: 'bg-blue-500/20 text-blue-300' },
  { name: 'Technology', icon: Globe, count: 4567, color: 'bg-cyan-500/20 text-cyan-300' },
  { name: 'Society & Culture', icon: Users, count: 3421, color: 'bg-purple-500/20 text-purple-300' },
  { name: 'Business & Economy', icon: TrendingUp, count: 2890, color: 'bg-green-500/20 text-green-300' },
  { name: 'Environment', icon: Globe, count: 1987, color: 'bg-emerald-500/20 text-emerald-300' },
  { name: 'Health & Medicine', icon: Zap, count: 2156, color: 'bg-red-500/20 text-red-300' }
];

export function ExplorePage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('all');
  const [trends, setTrends] = useState<TrendData[]>([]);
  const [newsResults, setNewsResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const loadTrendingContent = useCallback(async () => {
    setLoading(true);
    try {
      // Get trending data from enhanced trend engine
      const trendingData = enhancedTrendEngine.getTrends().slice(0, 8);
      setTrends(trendingData);

      // Aggregated headlines from multi-source trend aggregator
      let headlines: any[] = [];
      try {
        headlines = await trendAggregator.fetchHeadlines();
      } catch (e) {
        headlines = [];
      }

      const items = (headlines || []).slice(0, 6).map(h => ({
        title: truncateText(cleanText(h.title), 100),
        source: h.source,
        time: 'Now',
        category: h.category,
        snippet: truncateText(cleanText(h.snippet || ''), 160)
      }));

      if (items.length === 0) {
        const fallback = trendingData.slice(0, 6).map(t => ({
          title: truncateText(cleanText(t.topic), 100),
          source: 'Trending Now',
          time: 'Now',
          category: t.category,
          snippet: truncateText(cleanText(`Developing: ${t.topic}`), 160)
        }));
        setNewsResults(fallback);
      } else {
        setNewsResults(items);
      }
    } catch (error) {
      console.error('Failed to load trending content:', error);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadTrendingContent();
    
    // Subscribe to trend engine updates
    const unsubscribe = enhancedTrendEngine.subscribe((data) => {
      if (data.type === 'trends_updated') {
        setTrends(enhancedTrendEngine.getTrends().slice(0, 8));
      }
    });

    const interval = setInterval(loadTrendingContent, 60000);
    
    return () => {
      clearInterval(interval);
      unsubscribe();
    };
  }, [loadTrendingContent]);

  return (
    <div className="flex-1 bg-background min-h-0 flex flex-col">
      {/* Header */}
      <div className="bg-card backdrop-blur-sm border-b border-border p-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                <Compass className="w-8 h-8 text-cyan-400" />
                Explore
              </h1>
              <p className="text-gray-400">Discover trending topics, breaking news, and global conversations</p>
            </div>
            <div className="flex items-center space-x-3">
              <Button 
                onClick={loadTrendingContent} 
                size="sm" 
                className="bg-cyan-500 hover:bg-cyan-600"
                disabled={loading}
              >
                <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              <div className="px-3 py-1 bg-green-500/10 text-green-400 rounded-full text-sm font-medium border border-green-500/20 flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Live Updates</span>
              </div>
            </div>
          </div>
          
          {/* Search and Filters */}
          <div className="flex flex-col md:flex-row gap-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <Input
                type="text"
                placeholder="Search topics, news, people, trends..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-12 h-12 bg-gray-800/50 border-gray-700 text-white placeholder-gray-400 focus:border-cyan-500 text-lg"
              />
            </div>
            <div className="flex gap-2">
              {['all', 'trending', 'news', 'people'].map((filter) => (
                <Button
                  key={filter}
                  variant={activeFilter === filter ? "default" : "outline"}
                  onClick={() => setActiveFilter(filter)}
                  className={`capitalize ${
                    activeFilter === filter 
                      ? 'bg-cyan-500 hover:bg-cyan-600' 
                      : 'border-gray-700 text-gray-300 hover:bg-gray-800'
                  }`}
                >
                  {filter}
                </Button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-auto">
        <div className="max-w-6xl mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-8">
          {/* Breaking News */}
          <div>
            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
              <Calendar className="w-6 h-6 text-red-400" />
              Breaking News
            </h2>
            <div className="space-y-4">
              {loading ? (
                [...Array(4)].map((_, i) => (
                  <div key={i} className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-6 animate-pulse">
                    <div className="w-20 h-4 bg-gray-700 rounded mb-3"></div>
                    <div className="w-3/4 h-6 bg-gray-700 rounded mb-2"></div>
                    <div className="w-full h-4 bg-gray-700 rounded mb-3"></div>
                    <div className="w-1/3 h-4 bg-gray-700 rounded"></div>
                  </div>
                ))
              ) : (
                newsResults.map((news, index) => (
                <div key={index} className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-colors">
                  <div className="flex items-start justify-between mb-3">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      news.category === 'Science' ? 'bg-blue-500/20 text-blue-300' :
                      news.category === 'Technology' ? 'bg-cyan-500/20 text-cyan-300' :
                      news.category === 'Environment' ? 'bg-green-500/20 text-green-300' :
                      'bg-purple-500/20 text-purple-300'
                    }`}>
                      {news.category}
                    </span>
                    <span className="text-gray-400 text-sm">{news.time}</span>
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2 line-clamp-2">{news.title}</h3>
                  <p className="text-gray-300 mb-3 line-clamp-3">{news.snippet}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-400">{news.source}</span>
                    <Button size="sm" variant="outline" className="border-gray-700 text-gray-300 hover:bg-gray-800">
                      Read Source
                    </Button>
                  </div>
                </div>
                ))
              )}
            </div>
          </div>

          {/* Discovery Categories */}
          <div>
            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
              <Filter className="w-6 h-6 text-cyan-400" />
              Explore Categories
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {discoveryCategories.map((category, index) => (
                <div key={index} className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-colors cursor-pointer">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                      <div className={`p-3 rounded-lg ${category.color}`}>
                        <category.icon className="w-6 h-6" />
                      </div>
                      <div>
                        <h3 className="font-semibold text-white">{category.name}</h3>
                        <p className="text-sm text-gray-400">{category.count.toLocaleString()} posts</p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-8">
          {/* Trending Topics */}
          <div>
            <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-orange-400" />
              Trending Now
            </h2>
            <div className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-4">
              <div className="space-y-4">
                {loading ? (
                  [...Array(6)].map((_, i) => (
                    <div key={i} className="flex items-center justify-between p-3 animate-pulse">
                      <div className="flex-1">
                        <div className="w-32 h-4 bg-gray-700 rounded mb-2"></div>
                        <div className="w-20 h-3 bg-gray-700 rounded"></div>
                      </div>
                      <div className="w-12 h-4 bg-gray-700 rounded"></div>
                    </div>
                  ))
                ) : (
                  trends.map((trend, index) => (
                    <div key={index} className="flex items-center justify-between hover:bg-gray-800/50 rounded-lg p-3 transition-colors cursor-pointer">
                      <div>
                        <h4 className="font-medium text-white">{trend.topic.length > 30 ? trend.topic.substring(0, 30) + '...' : trend.topic}</h4>
                        <p className="text-sm text-gray-400">{trend.reach.toLocaleString()} reach • {trend.category}</p>
                      </div>
                      <span className="text-green-400 text-sm font-medium">+{trend.momentum.toFixed(0)}%</span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Quick Stats */}
          <div>
            <h2 className="text-xl font-bold text-white mb-4">Platform Stats</h2>
            <div className="bg-gray-900/50 rounded-lg border border-gray-800/50 p-4 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Active Users</span>
                <span className="text-white font-semibold">2.4M</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Posts Today</span>
                <span className="text-white font-semibold">156K</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Trending Topics</span>
                <span className="text-white font-semibold">847</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-300">Global Reach</span>
                <span className="text-white font-semibold">195 Countries</span>
              </div>
            </div>
          </div>
        </div>
        </div>
      </div>
    </div>
  );
}
