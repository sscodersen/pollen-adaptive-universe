import React, { useState, useEffect } from 'react';
import { TrendingUp, Flame, Zap, Globe, Hash, ArrowUp, Eye, MessageCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { enhancedTrendEngine, GeneratedPost } from '@/services/enhancedTrendEngine';
import { personalizationEngine } from '@/services/personalizationEngine';

const TREND_CATEGORIES = [
  { id: 'all', label: 'All Trends', icon: TrendingUp },
  { id: 'technology', label: 'Technology', icon: Zap },
  { id: 'social', label: 'Social', icon: MessageCircle },
  { id: 'business', label: 'Business', icon: Globe }
];

const Trends = () => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [trends, setTrends] = useState<GeneratedPost[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadTrends();
  }, [selectedCategory]);

  const loadTrends = async () => {
    setLoading(true);
    try {
      const allPosts = await enhancedTrendEngine.getGeneratedPosts();
      const filteredPosts = selectedCategory === 'all' 
        ? allPosts 
        : allPosts.filter(post => post.type === selectedCategory);
      
      setTrends(filteredPosts.slice(0, selectedCategory === 'all' ? 30 : 20));
    } catch (error) {
      console.error('Failed to load trends:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleTrendClick = (trend: GeneratedPost) => {
    personalizationEngine.trackBehavior({
      action: 'click',
      contentId: trend.id,
      contentType: 'educational',
      metadata: {
        category: trend.type,
        topic: trend.topic,
        hashtags: trend.hashtags
      }
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-red-50 to-pink-50 dark:from-gray-900 dark:via-red-900/20 dark:to-gray-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-orange-500 to-red-500 rounded-2xl">
              <Flame className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-orange-600 to-red-600 bg-clip-text text-transparent">
                Trending Now
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Real-time trends powered by Pollen AI
              </p>
            </div>
          </div>
        </div>

        {/* Category Tabs */}
        <div className="flex gap-2 mb-8 overflow-x-auto scrollbar-thin pb-2">
          {TREND_CATEGORIES.map(category => (
            <Button
              key={category.id}
              variant={selectedCategory === category.id ? 'default' : 'outline'}
              onClick={() => setSelectedCategory(category.id)}
              className="whitespace-nowrap"
            >
              <category.icon className="w-4 h-4 mr-2" />
              {category.label}
            </Button>
          ))}
        </div>

        {/* Trends Grid */}
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[...Array(9)].map((_, i) => (
              <div key={i} className="animate-pulse bg-white/60 dark:bg-gray-800/60 rounded-2xl p-6 h-48" />
            ))}
          </div>
        ) : (
          <div className="space-y-4">
            {/* Top Trending */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {trends.slice(0, 3).map((trend, index) => (
                <div
                  key={trend.id}
                  onClick={() => handleTrendClick(trend)}
                  className="relative bg-gradient-to-br from-orange-500/20 to-red-500/20 backdrop-blur-sm rounded-2xl p-6 border-2 border-orange-500/30 hover:border-orange-500/50 hover:shadow-2xl transition-all duration-300 cursor-pointer group"
                >
                  {/* Rank Badge */}
                  <div className="absolute top-4 right-4 w-12 h-12 bg-gradient-to-br from-orange-500 to-red-500 rounded-full flex items-center justify-center text-white font-bold text-lg shadow-lg">
                    #{index + 1}
                  </div>

                  {/* Content */}
                  <div className="mb-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Flame className="w-5 h-5 text-orange-500" />
                      <h3 className="font-bold text-gray-900 dark:text-white text-lg line-clamp-1">
                        {trend.topic}
                      </h3>
                    </div>
                    <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-2 mb-3">
                      {trend.content}
                    </p>
                  </div>

                  {/* Stats */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                      <div className="flex items-center gap-1">
                        <Eye className="w-4 h-4" />
                        <span>{(trend.engagement_score * 100).toFixed(0)}k</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <ArrowUp className="w-4 h-4 text-green-500" />
                        <span className="text-green-600 font-medium">
                          {trend.engagement_score.toFixed(1)}
                        </span>
                      </div>
                    </div>
                    <span className="px-3 py-1 bg-orange-500/20 text-orange-600 rounded-full text-xs font-medium capitalize">
                      {trend.type}
                    </span>
                  </div>

                  {/* Hashtags */}
                  <div className="mt-3 flex gap-2 flex-wrap">
                    {trend.hashtags.slice(0, 3).map((tag, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs text-gray-600 dark:text-gray-300"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* More Trends */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {trends.slice(3).map((trend, index) => (
                <div
                  key={trend.id}
                  onClick={() => handleTrendClick(trend)}
                  className="bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-xl p-4 border border-gray-200 dark:border-gray-700 hover:shadow-lg hover:scale-[1.02] transition-all duration-300 cursor-pointer"
                >
                  {/* Header */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Hash className="w-4 h-4 text-orange-500" />
                      <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
                        #{index + 4}
                      </span>
                    </div>
                    <div className="flex items-center gap-1 text-green-600 text-xs font-medium">
                      <TrendingUp className="w-3 h-3" />
                      {trend.engagement_score.toFixed(1)}
                    </div>
                  </div>

                  {/* Content */}
                  <h4 className="font-semibold text-gray-900 dark:text-white mb-2 line-clamp-2">
                    {trend.topic}
                  </h4>
                  <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2 mb-3">
                    {trend.content}
                  </p>

                  {/* Footer */}
                  <div className="flex items-center justify-between">
                    <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs text-gray-600 dark:text-gray-300 capitalize">
                      {trend.type}
                    </span>
                    <div className="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400">
                      <Eye className="w-3 h-3" />
                      {(trend.engagement_score * 50).toFixed(0)}k
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Trends;
