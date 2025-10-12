import React, { useState, useEffect, useCallback } from 'react';
import { Newspaper, TrendingUp, Clock, Eye, Bookmark, Share2, Filter, Search, Sparkles, Globe, ChevronRight } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { contentOrchestrator } from '@/services/contentOrchestrator';
import { NewsContent } from '@/services/unifiedContentEngine';
import { personalizationEngine } from '@/services/personalizationEngine';
import { enhancedTrendEngine } from '@/services/enhancedTrendEngine';

const NEWS_CATEGORIES = [
  { id: 'all', label: 'All News', icon: Globe },
  { id: 'technology', label: 'Technology', icon: Sparkles },
  { id: 'business', label: 'Business', icon: TrendingUp },
  { id: 'science', label: 'Science', icon: Sparkles },
  { id: 'health', label: 'Health', icon: Sparkles },
  { id: 'environment', label: 'Environment', icon: Globe }
];

const News = () => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [newsArticles, setNewsArticles] = useState<NewsContent[]>([]);
  const [trendingTopics, setTrendingTopics] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [savedArticles, setSavedArticles] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadNews();
    loadTrendingTopics();
  }, [selectedCategory]);

  const loadNews = async () => {
    setLoading(true);
    try {
      const query = selectedCategory === 'all' 
        ? 'latest breaking news and important developments'
        : `latest ${selectedCategory} news and updates`;

      const response = await contentOrchestrator.generateContent<NewsContent>({
        type: 'news',
        query,
        count: 20,
        strategy: {
          diversity: 0.9,
          freshness: 1.0,
          personalization: 0.8,
          qualityThreshold: 8.0,
          trendingBoost: 0.7
        }
      });

      const personalizedNews = await personalizationEngine.generateRecommendations(
        response.content,
        20
      );

      setNewsArticles(personalizedNews);
    } catch (error) {
      console.error('Failed to load news:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadTrendingTopics = async () => {
    try {
      const trends = await enhancedTrendEngine.getTrends();
      const topics = trends.slice(0, 8).map(trend => trend.topic);
      setTrendingTopics(topics);
    } catch (error) {
      console.error('Failed to load trending topics:', error);
    }
  };

  const handleSaveArticle = useCallback((articleId: string) => {
    setSavedArticles(prev => {
      const newSet = new Set(prev);
      if (newSet.has(articleId)) {
        newSet.delete(articleId);
      } else {
        newSet.add(articleId);
      }
      return newSet;
    });

    personalizationEngine.trackBehavior({
      action: 'save',
      contentId: articleId,
      contentType: 'educational',
      metadata: { category: selectedCategory, type: 'news' }
    });
  }, [selectedCategory]);

  const handleArticleClick = useCallback((article: NewsContent) => {
    personalizationEngine.trackBehavior({
      action: 'click',
      contentId: article.id,
      contentType: 'educational',
      metadata: { 
        category: article.category,
        source: article.source,
        tags: article.tags
      }
    });
  }, []);

  const filteredArticles = searchQuery
    ? newsArticles.filter(article =>
        article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        article.snippet.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : newsArticles;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl">
              <Newspaper className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                AI News Hub
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Curated news powered by Pollen AI
              </p>
            </div>
          </div>

          {/* Search Bar */}
          <div className="relative max-w-2xl">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <Input
              type="text"
              placeholder="Search news articles..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-12 h-12 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-gray-200 dark:border-gray-700"
            />
          </div>
        </div>

        {/* Category Tabs */}
        <div className="flex gap-2 mb-6 overflow-x-auto scrollbar-thin pb-2">
          {NEWS_CATEGORIES.map(category => (
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

        {/* Trending Topics */}
        {trendingTopics.length > 0 && (
          <div className="mb-8 p-6 bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-2xl border border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-orange-500" />
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                Trending Topics
              </h2>
            </div>
            <div className="flex gap-2 flex-wrap">
              {trendingTopics.map((topic, index) => (
                <button
                  key={index}
                  onClick={() => setSearchQuery(topic)}
                  className="px-4 py-2 bg-gradient-to-r from-orange-500/20 to-red-500/20 hover:from-orange-500/30 hover:to-red-500/30 rounded-full text-sm font-medium text-gray-700 dark:text-gray-300 transition-all"
                >
                  {topic}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* News Articles Grid */}
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="bg-white/60 dark:bg-gray-800/60 rounded-2xl p-6 h-64">
                  <div className="h-4 bg-gray-300 dark:bg-gray-600 rounded w-3/4 mb-4" />
                  <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-full mb-2" />
                  <div className="h-3 bg-gray-300 dark:bg-gray-600 rounded w-5/6" />
                </div>
              </div>
            ))}
          </div>
        ) : filteredArticles.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredArticles.map(article => (
              <article
                key={article.id}
                onClick={() => handleArticleClick(article)}
                className="group bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-2xl p-6 border border-gray-200 dark:border-gray-700 hover:shadow-xl hover:scale-[1.02] transition-all duration-300 cursor-pointer"
              >
                {/* Article Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      article.impact === 'critical' ? 'bg-red-500/20 text-red-600' :
                      article.impact === 'high' ? 'bg-orange-500/20 text-orange-600' :
                      'bg-blue-500/20 text-blue-600'
                    }`}>
                      {article.category}
                    </span>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleSaveArticle(article.id);
                    }}
                    className="p-2 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors"
                  >
                    <Bookmark 
                      className={`w-4 h-4 ${savedArticles.has(article.id) ? 'fill-blue-500 text-blue-500' : 'text-gray-400'}`}
                    />
                  </button>
                </div>

                {/* Article Title */}
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3 line-clamp-2 group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors">
                  {article.title}
                </h3>

                {/* Article Snippet */}
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4 line-clamp-3">
                  {article.snippet}
                </p>

                {/* Article Footer */}
                <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
                  <div className="flex items-center gap-4 text-xs text-gray-500 dark:text-gray-400">
                    <div className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {article.readTime}
                    </div>
                    <div className="flex items-center gap-1">
                      <Eye className="w-3 h-3" />
                      {article.views}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
                      {article.source}
                    </span>
                    <ChevronRight className="w-4 h-4 text-gray-400 group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>

                {/* Tags */}
                {article.tags && article.tags.length > 0 && (
                  <div className="flex gap-1 mt-3 flex-wrap">
                    {article.tags.slice(0, 3).map((tag, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded text-xs text-gray-600 dark:text-gray-300"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}
              </article>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <Newspaper className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-600 dark:text-gray-400">No news articles found</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default News;
