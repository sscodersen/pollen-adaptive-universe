
import React, { useState, useEffect } from 'react';
import { Globe, TrendingUp, Clock, ExternalLink, Star, Search, Filter, Zap, Eye } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

interface NewsArticle {
  id: string;
  title: string;
  summary: string;
  content: string;
  author: string;
  source: string;
  timestamp: string;
  category: 'tech' | 'science' | 'politics' | 'health' | 'environment' | 'business';
  significance: number;
  trending: boolean;
  readTime: number;
  views: number;
  isOriginal: boolean;
}

interface NewsEngineProps {
  isGenerating?: boolean;
}

export const NewsEngine = ({ isGenerating = true }: NewsEngineProps) => {
  const [articles, setArticles] = useState<NewsArticle[]>([]);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [generatingArticle, setGeneratingArticle] = useState(false);

  const categories = ['All', 'Tech', 'Science', 'Politics', 'Health', 'Environment', 'Business'];

  const newsTopics = [
    'artificial intelligence breakthrough in healthcare',
    'climate change solution development',
    'quantum computing advancement',
    'renewable energy innovation',
    'space exploration discovery',
    'biotechnology medical breakthrough',
    'economic policy impact analysis',
    'social media platform regulation',
    'cybersecurity threat prevention',
    'sustainable technology adoption',
    'global trade relationship shifts',
    'scientific research collaboration',
    'digital privacy legislation',
    'automation workplace impact',
    'environmental conservation success'
  ];

  useEffect(() => {
    if (!isGenerating) return;

    const generateArticle = async () => {
      if (generatingArticle) return;
      
      setGeneratingArticle(true);
      try {
        const randomTopic = newsTopics[Math.floor(Math.random() * newsTopics.length)];
        const categories_list = ['tech', 'science', 'politics', 'health', 'environment', 'business'] as const;
        const randomCategory = categories_list[Math.floor(Math.random() * categories_list.length)];
        
        // Generate comprehensive news content using Pollen AI
        const response = await pollenAI.generate(
          `Create an original, unbiased news article about ${randomTopic}. Include analysis of multiple perspectives, fact-based reporting, and practical implications for readers. Make it comprehensive and informative.`,
          "news",
          true
        );
        
        const newArticle: NewsArticle = {
          id: Date.now().toString(),
          title: generateNewsTitle(randomTopic, randomCategory),
          summary: response.content.slice(0, 200) + '...',
          content: response.content,
          author: 'Pollen News Team',
          source: 'Pollen Intelligence',
          timestamp: formatTimestamp(new Date()),
          category: randomCategory,
          significance: response.significanceScore || 8.5,
          trending: response.significanceScore ? response.significanceScore > 9.0 : Math.random() > 0.7,
          readTime: Math.ceil(response.content.length / 200),
          views: Math.floor(Math.random() * 25000) + 1000,
          isOriginal: true
        };
        
        setArticles(prev => [newArticle, ...prev.slice(0, 19)]);
      } catch (error) {
        console.error('Failed to generate news article:', error);
      }
      setGeneratingArticle(false);
    };

    const initialTimeout = setTimeout(generateArticle, 2000);
    const interval = setInterval(generateArticle, Math.random() * 30000 + 45000);
    
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [isGenerating, generatingArticle]);

  const generateNewsTitle = (topic: string, category: string) => {
    const titleFormats = [
      `Breaking: Major ${topic} Development Changes Industry Landscape`,
      `Exclusive Analysis: ${topic} Impact on Global Markets`,
      `Investigation: The Truth Behind ${topic} Claims`,
      `Deep Dive: How ${topic} Will Shape the Future`,
      `Expert Panel: ${topic} Implications for Society`,
      `Research Update: ${topic} Shows Promising Results`,
      `Policy Review: ${topic} Regulatory Framework`,
      `Global Impact: ${topic} Affects 2.3 Billion People`
    ];
    
    return titleFormats[Math.floor(Math.random() * titleFormats.length)]
      .replace(topic, topic.charAt(0).toUpperCase() + topic.slice(1));
  };

  const formatTimestamp = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return `${Math.floor(diffMins / 1440)}d ago`;
  };

  const getCategoryColor = (category: string) => {
    const colors = {
      tech: 'text-blue-400 bg-blue-400/10 border-blue-400/20',
      science: 'text-green-400 bg-green-400/10 border-green-400/20',
      politics: 'text-purple-400 bg-purple-400/10 border-purple-400/20',
      health: 'text-red-400 bg-red-400/10 border-red-400/20',
      environment: 'text-teal-400 bg-teal-400/10 border-teal-400/20',
      business: 'text-yellow-400 bg-yellow-400/10 border-yellow-400/20'
    };
    return colors[category as keyof typeof colors] || 'text-gray-400 bg-gray-400/10 border-gray-400/20';
  };

  const openArticle = (article: NewsArticle) => {
    // Create a new page/modal to display the full AI-generated content
    const newWindow = window.open('', '_blank');
    if (newWindow) {
      newWindow.document.write(`
        <html>
          <head>
            <title>${article.title}</title>
            <style>
              body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
              h1 { color: #1a1a1a; }
              .meta { color: #666; font-size: 14px; margin-bottom: 20px; }
              .content { color: #333; }
            </style>
          </head>
          <body>
            <h1>${article.title}</h1>
            <div class="meta">
              By ${article.author} • ${article.source} • ${article.timestamp} • ${article.readTime} min read
            </div>
            <div class="content">
              ${article.content.split('\n').map(p => `<p>${p}</p>`).join('')}
            </div>
          </body>
        </html>
      `);
    }
  };

  const filteredArticles = articles.filter(article => {
    const matchesCategory = selectedCategory === 'All' || article.category === selectedCategory.toLowerCase();
    const matchesSearch = searchQuery === '' || 
      article.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      article.content.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="flex-1 flex flex-col bg-gray-950">
      {/* Header */}
      <div className="p-6 border-b border-gray-800/50 bg-gray-900/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">News Engine</h1>
            <p className="text-gray-400">AI-generated unbiased news • Multiple perspectives • Fact-based reporting</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingArticle && (
              <div className="flex items-center space-x-2 text-cyan-400">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium">Generating article...</span>
              </div>
            )}
            <div className="text-right">
              <div className="text-2xl font-bold text-white">{filteredArticles.length}</div>
              <div className="text-xs text-gray-400">Articles</div>
            </div>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex-1 max-w-md relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search articles..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg pl-10 pr-4 py-2 text-white placeholder-gray-400 focus:border-cyan-500/50 focus:outline-none transition-colors"
            />
          </div>
          
          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-400">Filter:</span>
          </div>
        </div>

        {/* Category Tabs */}
        <div className="flex space-x-2 overflow-x-auto">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
                selectedCategory === category
                  ? 'bg-cyan-600 text-white'
                  : 'bg-gray-800/50 text-gray-300 hover:bg-gray-700/50 hover:text-white'
              }`}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {/* Articles */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="space-y-6">
          {filteredArticles.map((article) => (
            <article key={article.id} 
                    className="bg-gray-900/80 rounded-2xl border border-gray-800/50 p-6 hover:bg-gray-900/90 transition-all duration-200 backdrop-blur-sm cursor-pointer"
                    onClick={() => openArticle(article)}>
              
              {/* Article Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${getCategoryColor(article.category)}`}>
                      {article.category.charAt(0).toUpperCase() + article.category.slice(1)}
                    </span>
                    {article.trending && (
                      <div className="flex items-center space-x-1 px-2 py-1 bg-orange-500/20 text-orange-400 text-xs rounded-full">
                        <TrendingUp className="w-3 h-3" />
                        <span>Trending</span>
                      </div>
                    )}
                    {article.isOriginal && (
                      <div className="flex items-center space-x-1 px-2 py-1 bg-cyan-500/20 text-cyan-400 text-xs rounded-full">
                        <Zap className="w-3 h-3" />
                        <span>AI Original</span>
                      </div>
                    )}
                  </div>
                  
                  <h2 className="text-xl font-bold text-white mb-3 line-clamp-2 hover:text-cyan-300 transition-colors">
                    {article.title}
                  </h2>
                  
                  <p className="text-gray-300 mb-4 line-clamp-3">
                    {article.summary}
                  </p>
                </div>
              </div>

              {/* Article Meta */}
              <div className="flex items-center justify-between pt-4 border-t border-gray-800/50">
                <div className="flex items-center space-x-4 text-sm text-gray-400">
                  <div className="flex items-center space-x-1">
                    <Globe className="w-4 h-4" />
                    <span>{article.source}</span>
                  </div>
                  <div className="flex items-center space-x-1">
                    <Clock className="w-4 h-4" />
                    <span>{article.timestamp}</span>
                  </div>
                  <span>{article.readTime} min read</span>
                </div>
                
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-1">
                    <Star className="w-4 h-4 text-yellow-400" />
                    <span className="text-sm font-medium text-yellow-400">{article.significance.toFixed(1)}</span>
                  </div>
                  <div className="flex items-center space-x-1 text-gray-400">
                    <Eye className="w-4 h-4" />
                    <span className="text-sm">{article.views.toLocaleString()}</span>
                  </div>
                  <ExternalLink className="w-4 h-4 text-gray-400 hover:text-white transition-colors" />
                </div>
              </div>
            </article>
          ))}
        </div>

        {filteredArticles.length === 0 && (
          <div className="text-center py-20">
            <div className="w-24 h-24 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-8">
              <Globe className="w-12 h-12 text-white animate-pulse" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-4">News Engine Initializing...</h3>
            <p className="text-gray-400 max-w-lg mx-auto text-lg">
              Pollen AI is analyzing global news sources and generating unbiased, comprehensive articles with multiple perspectives.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
