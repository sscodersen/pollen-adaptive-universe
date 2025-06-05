
import React, { useState, useEffect } from 'react';
import { Clock, TrendingUp, Globe, Zap, ExternalLink, Filter, BarChart3, Star } from 'lucide-react';
import { contentCurator, type WebContent } from '../services/contentCurator';

interface NewsEngineProps {
  isGenerating?: boolean;
}

export const NewsEngine = ({ isGenerating = true }: NewsEngineProps) => {
  const [newsItems, setNewsItems] = useState<WebContent[]>([]);
  const [generatingNews, setGeneratingNews] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [sortBy, setSortBy] = useState('significance');

  const categories = ['All', 'Technology', 'Science', 'Politics', 'Business', 'Health', 'Environment'];

  useEffect(() => {
    if (!isGenerating) return;

    const generateNews = async () => {
      if (generatingNews) return;
      
      setGeneratingNews(true);
      try {
        const curated = await contentCurator.scrapeAndCurateContent('news', 25);
        setNewsItems(prev => {
          const newItems = curated.filter(item => 
            !prev.some(existing => existing.title === item.title)
          );
          return [...newItems, ...prev].slice(0, 50);
        });
      } catch (error) {
        console.error('Failed to curate news:', error);
      }
      setGeneratingNews(false);
    };

    generateNews();
    const interval = setInterval(generateNews, Math.random() * 30000 + 45000);
    return () => clearInterval(interval);
  }, [isGenerating, generatingNews]);

  const formatNewsTimestamp = (timestamp: number) => {
    const now = Date.now();
    const diffMs = now - timestamp;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minutes ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)} hours ago`;
    return `${Math.floor(diffMins / 1440)} days ago`;
  };

  const openLink = (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  const filteredNews = newsItems;
  const sortedNews = [...filteredNews].sort((a, b) => {
    if (sortBy === 'significance') return b.significance - a.significance;
    if (sortBy === 'time') return b.timestamp - a.timestamp;
    return 0;
  });

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Global News Intelligence</h1>
            <p className="text-gray-400">AI-curated • 7+ significance rated • Live web analysis</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingNews && (
              <div className="flex items-center space-x-2 text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm">Analyzing web sources...</span>
              </div>
            )}
            <div className="flex items-center space-x-2">
              <Globe className="w-4 h-4 text-blue-400" />
              <span className="text-sm text-gray-400">{sortedNews.length} verified articles</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-gray-700/50 border border-gray-600/50 rounded-lg px-3 py-1 text-sm text-white"
            >
              <option value="significance">Significance Score</option>
              <option value="time">Latest First</option>
            </select>
          </div>
        </div>
      </div>

      {/* News Feed */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {sortedNews.map((item) => (
          <article key={item.id} className="bg-gray-800/50 rounded-xl border border-gray-700/50 p-6 hover:bg-gray-800/70 transition-colors group cursor-pointer"
                   onClick={() => openLink(item.url)}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-3">
                  <span className="px-3 py-1 bg-blue-500/20 text-blue-300 text-xs font-medium rounded-full">
                    {item.source}
                  </span>
                  <span className="flex items-center space-x-1 px-3 py-1 bg-yellow-500/20 text-yellow-300 text-xs font-medium rounded-full">
                    <Star className="w-3 h-3" />
                    <span>{item.significance.toFixed(1)}/10</span>
                  </span>
                  <span className="text-xs text-gray-400">{formatNewsTimestamp(item.timestamp)}</span>
                </div>
                
                <h2 className="text-xl font-bold text-white mb-3 leading-tight group-hover:text-blue-300 transition-colors">
                  {item.title}
                </h2>
                
                <p className="text-gray-300 text-sm leading-relaxed mb-4 line-clamp-3">
                  {item.description}
                </p>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-6 text-xs text-gray-400">
                <div className="flex items-center space-x-2">
                  <Globe className="w-3 h-3" />
                  <span>Verified Source</span>
                </div>
              </div>
              
              <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-gray-400">Impact:</span>
                  <div className="w-16 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-yellow-400 rounded-full transition-all"
                      style={{ width: `${(item.significance / 10) * 100}%` }}
                    />
                  </div>
                </div>
                
                <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors" />
              </div>
            </div>
          </article>
        ))}

        {sortedNews.length === 0 && (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full flex items-center justify-center mx-auto mb-6">
              <Globe className="w-10 h-10 text-white animate-pulse" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Scanning Global Sources...</h3>
            <p className="text-gray-400 max-w-md mx-auto">
              Pollen is analyzing thousands of news sources worldwide using our 7-factor significance algorithm. Only content scoring 7+ will be curated.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
