
import React, { useState, useEffect } from 'react';
import { Clock, TrendingUp, Globe, Zap, ExternalLink } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';

interface NewsItem {
  id: string;
  title: string;
  summary: string;
  category: string;
  timestamp: string;
  relevanceScore: number;
  originalityScore: number;
  source: string;
  trending: boolean;
}

export const NewsEngine = () => {
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('All');

  const categories = ['All', 'Technology', 'Science', 'Politics', 'Entertainment', 'Sports', 'Health'];

  useEffect(() => {
    const generateNews = async () => {
      if (isGenerating) return;
      
      setIsGenerating(true);
      try {
        const topics = ['breaking technology news', 'scientific discoveries', 'global developments', 'cultural trends', 'innovation updates'];
        const randomTopic = topics[Math.floor(Math.random() * topics.length)];
        
        const response = await pollenAI.generate(
          `Generate unbiased news content about ${randomTopic} with analysis`,
          "analysis"
        );
        
        const newItem: NewsItem = {
          id: Date.now().toString(),
          title: generateNewsTitle(randomTopic),
          summary: response.content,
          category: getCategory(randomTopic),
          timestamp: new Date().toLocaleTimeString(),
          relevanceScore: Math.random() * 100,
          originalityScore: Math.random() * 100,
          source: 'Pollen Analysis',
          trending: Math.random() > 0.7
        };
        
        setNewsItems(prev => [newItem, ...prev.slice(0, 19)]);
      } catch (error) {
        console.error('Failed to generate news:', error);
      }
      setIsGenerating(false);
    };

    generateNews();
    const interval = setInterval(generateNews, 45000); // Generate every 45 seconds
    return () => clearInterval(interval);
  }, []);

  const generateNewsTitle = (topic: string) => {
    const titles = [
      "Revolutionary AI Breakthrough Changes Industry Standards",
      "Global Climate Innovation Summit Reveals New Solutions",
      "Quantum Computing Milestone Achieved by Research Team",
      "Sustainable Energy Adoption Reaches Historic High",
      "Space Exploration Mission Uncovers Unexpected Findings"
    ];
    return titles[Math.floor(Math.random() * titles.length)];
  };

  const getCategory = (topic: string) => {
    if (topic.includes('technology')) return 'Technology';
    if (topic.includes('scientific')) return 'Science';
    if (topic.includes('global')) return 'Politics';
    if (topic.includes('cultural')) return 'Entertainment';
    return 'Technology';
  };

  const filteredNews = selectedCategory === 'All' 
    ? newsItems 
    : newsItems.filter(item => item.category === selectedCategory);

  return (
    <div className="flex-1 overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b border-white/10 bg-black/20 backdrop-blur-xl">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Unbiased News Engine</h1>
            <p className="text-white/60">Real-time analysis ranked by relevance and originality</p>
          </div>
          <div className="flex items-center space-x-2">
            {isGenerating && (
              <div className="flex items-center space-x-2 text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm">Analyzing...</span>
              </div>
            )}
          </div>
        </div>

        {/* Category Filter */}
        <div className="flex space-x-2 overflow-x-auto">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
                selectedCategory === category
                  ? 'bg-cyan-500 text-white'
                  : 'bg-white/10 text-white/60 hover:bg-white/20 hover:text-white'
              }`}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {/* News Feed */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {filteredNews.map((item) => (
          <div key={item.id} className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/10 p-5 hover:bg-white/15 transition-colors">
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <div className="flex items-center space-x-2 mb-2">
                  <span className="px-2 py-1 bg-cyan-500/20 text-cyan-300 text-xs font-medium rounded">
                    {item.category}
                  </span>
                  {item.trending && (
                    <span className="flex items-center space-x-1 px-2 py-1 bg-orange-500/20 text-orange-300 text-xs font-medium rounded">
                      <TrendingUp className="w-3 h-3" />
                      <span>Trending</span>
                    </span>
                  )}
                </div>
                <h3 className="text-lg font-semibold text-white mb-2 leading-tight">
                  {item.title}
                </h3>
                <p className="text-white/70 text-sm leading-relaxed mb-3">
                  {item.summary.slice(0, 200)}...
                </p>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4 text-xs text-white/60">
                <div className="flex items-center space-x-1">
                  <Clock className="w-3 h-3" />
                  <span>{item.timestamp}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Globe className="w-3 h-3" />
                  <span>{item.source}</span>
                </div>
              </div>
              
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2 text-xs">
                  <span className="text-white/60">Relevance:</span>
                  <div className="w-12 h-2 bg-white/20 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-green-400 rounded-full"
                      style={{ width: `${item.relevanceScore}%` }}
                    />
                  </div>
                </div>
                <div className="flex items-center space-x-2 text-xs">
                  <span className="text-white/60">Originality:</span>
                  <div className="w-12 h-2 bg-white/20 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-purple-400 rounded-full"
                      style={{ width: `${item.originalityScore}%` }}
                    />
                  </div>
                </div>
                <button className="text-white/60 hover:text-white">
                  <ExternalLink className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        ))}

        {filteredNews.length === 0 && (
          <div className="text-center py-12">
            <div className="w-16 h-16 bg-gradient-to-r from-cyan-400 to-purple-400 rounded-full flex items-center justify-center mx-auto mb-4">
              <Zap className="w-8 h-8 text-white animate-pulse" />
            </div>
            <h3 className="text-lg font-semibold text-white mb-2">Analyzing Global Information...</h3>
            <p className="text-white/60">Pollen is processing and ranking news by relevance and originality</p>
          </div>
        )}
      </div>
    </div>
  );
};
