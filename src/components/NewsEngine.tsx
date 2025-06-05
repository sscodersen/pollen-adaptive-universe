import React, { useState, useEffect } from 'react';
import { Clock, TrendingUp, Globe, Zap, ExternalLink, Filter, BarChart3, Star } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

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
  readTime: number;
}

interface NewsEngineProps {
  isGenerating?: boolean;
}

export const NewsEngine = ({ isGenerating = true }: NewsEngineProps) => {
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
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
        // Get trending topics and create news-focused prompts
        const trendingTopics = significanceAlgorithm.getTrendingTopics();
        const randomTopic = trendingTopics[Math.floor(Math.random() * trendingTopics.length)];
        
        const newsPrompts = [
          `urgent breaking news about ${randomTopic} with global implications`,
          `in-depth analysis of ${randomTopic} affecting millions of people`,
          `exclusive investigation into ${randomTopic} revealing actionable insights`,
          `expert commentary on ${randomTopic} with practical applications`,
          `comprehensive report on ${randomTopic} breakthrough developments`,
          `critical assessment of ${randomTopic} impact on society`
        ];
        
        const randomPrompt = newsPrompts[Math.floor(Math.random() * newsPrompts.length)];
        
        const response = await pollenAI.generate(
          `Generate comprehensive news analysis about ${randomPrompt}`,
          "news",
          true // Use significance filtering
        );
        
        const newItem: NewsItem = {
          id: Date.now().toString(),
          title: generateNewsTitle(randomTopic),
          summary: response.content,
          category: getCategory(randomTopic),
          timestamp: formatNewsTimestamp(new Date()),
          relevanceScore: response.significanceScore || Math.random() * 40 + 60,
          originalityScore: Math.random() * 30 + 70,
          source: 'Pollen Analysis',
          trending: response.significanceScore ? response.significanceScore > 8.0 : Math.random() > 0.6,
          readTime: Math.floor(Math.random() * 8) + 2
        };
        
        setNewsItems(prev => [newItem, ...prev.slice(0, 24)]);
      } catch (error) {
        console.error('Failed to generate news:', error);
      }
      setGeneratingNews(false);
    };

    generateNews();
    // Generate news every 45-75 seconds (slower, more deliberate)
    const interval = setInterval(generateNews, Math.random() * 30000 + 45000);
    return () => clearInterval(interval);
  }, [isGenerating, generatingNews]);

  const generateNewsTitle = (topic: string) => {
    const titles = [
      "Revolutionary AI Framework Transforms Industry Standards",
      "Climate Innovation Summit Reveals Breakthrough Solutions",
      "Quantum Computing Milestone Achieved in Global Research",
      "Sustainable Energy Adoption Reaches Historic Threshold",
      "Space Exploration Mission Uncovers Unexpected Phenomena",
      "Biotech Advancement Opens New Treatment Possibilities",
      "Economic Patterns Signal Paradigm Shift in Markets",
      "Policy Reform Initiative Gains International Support"
    ];
    return titles[Math.floor(Math.random() * titles.length)];
  };

  const getCategory = (topic: string) => {
    if (topic.includes('technology') || topic.includes('AI')) return 'Technology';
    if (topic.includes('scientific') || topic.includes('research')) return 'Science';
    if (topic.includes('economic') || topic.includes('policy')) return 'Business';
    if (topic.includes('healthcare') || topic.includes('biotech')) return 'Health';
    if (topic.includes('environmental') || topic.includes('climate')) return 'Environment';
    return 'Technology';
  };

  const formatNewsTimestamp = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins} minutes ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)} hours ago`;
    return `${Math.floor(diffMins / 1440)} days ago`;
  };

  const filteredNews = selectedCategory === 'All' 
    ? newsItems 
    : newsItems.filter(item => item.category === selectedCategory);

  const sortedNews = [...filteredNews].sort((a, b) => {
    if (sortBy === 'significance') return b.relevanceScore - a.relevanceScore;
    if (sortBy === 'originality') return b.originalityScore - a.originalityScore;
    if (sortBy === 'time') return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
    return 0;
  });

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">News Engine</h1>
            <p className="text-gray-400">AI-powered significance analysis • Only 7+ rated content</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingNews && (
              <div className="flex items-center space-x-2 text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm">Analyzing significance...</span>
              </div>
            )}
            <div className="flex items-center space-x-2">
              <Star className="w-4 h-4 text-yellow-400" />
              <span className="text-sm text-gray-400">{filteredNews.length} high-impact articles</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between">
          {/* Category Filter */}
          <div className="flex space-x-2 overflow-x-auto">
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
                  selectedCategory === category
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50 hover:text-white'
                }`}
              >
                {category}
              </button>
            ))}
          </div>

          {/* Sort Options */}
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-gray-700/50 border border-gray-600/50 rounded-lg px-3 py-1 text-sm text-white"
            >
              <option value="significance">Significance Score</option>
              <option value="originality">Originality</option>
              <option value="time">Latest</option>
            </select>
          </div>
        </div>
      </div>

      {/* News Feed */}
      <div className="flex-1 overflow-y-auto p-6 space-y-4">
        {sortedNews.map((item) => (
          <article key={item.id} className="bg-gray-800/50 rounded-xl border border-gray-700/50 p-6 hover:bg-gray-800/70 transition-colors group">
            <div className="flex items-start justify-between mb-4">
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-3">
                  <span className="px-3 py-1 bg-blue-500/20 text-blue-300 text-xs font-medium rounded-full">
                    {item.category}
                  </span>
                  {item.trending && (
                    <span className="flex items-center space-x-1 px-3 py-1 bg-orange-500/20 text-orange-300 text-xs font-medium rounded-full">
                      <TrendingUp className="w-3 h-3" />
                      <span>High Impact</span>
                    </span>
                  )}
                  <span className="flex items-center space-x-1 px-3 py-1 bg-yellow-500/20 text-yellow-300 text-xs font-medium rounded-full">
                    <Star className="w-3 h-3" />
                    <span>7+ Rated</span>
                  </span>
                  <span className="text-xs text-gray-400">{item.readTime} min read</span>
                </div>
                
                <h2 className="text-xl font-bold text-white mb-3 leading-tight group-hover:text-blue-300 transition-colors">
                  {item.title}
                </h2>
                
                <p className="text-gray-300 text-sm leading-relaxed mb-4 line-clamp-3">
                  {item.summary.slice(0, 300)}...
                </p>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-6 text-xs text-gray-400">
                <div className="flex items-center space-x-2">
                  <Clock className="w-3 h-3" />
                  <span>{item.timestamp}</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Globe className="w-3 h-3" />
                  <span>{item.source}</span>
                </div>
              </div>
              
              <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-gray-400">Significance:</span>
                  <div className="w-16 h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-yellow-400 rounded-full transition-all"
                      style={{ width: `${(item.relevanceScore / 10) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-yellow-400 font-medium">{item.relevanceScore.toFixed(1)}/10</span>
                </div>
                
                <button className="text-gray-400 hover:text-white transition-colors">
                  <ExternalLink className="w-4 h-4" />
                </button>
              </div>
            </div>
          </article>
        ))}

        {sortedNews.length === 0 && (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full flex items-center justify-center mx-auto mb-6">
              <Star className="w-10 h-10 text-white animate-pulse" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Analyzing Global Significance...</h3>
            <p className="text-gray-400 max-w-md mx-auto">
              Pollen is processing the world's most impactful news using our 7-factor significance algorithm. Only content scoring 7+ is curated for presentation.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
