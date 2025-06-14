
import React, { useState, useEffect } from 'react';
import { Layout } from '../components/Layout';
import { Search as SearchIcon, Globe, Filter, TrendingUp, ExternalLink, Shield, Clock, Star, Zap, Brain } from 'lucide-react';

interface SearchResult {
  id: string;
  title: string;
  source: string;
  snippet: string;
  url: string;
  bias: 'neutral' | 'left' | 'right' | 'mixed';
  timestamp: string;
  credibility: number;
  category: string;
  relevance: number;
  factCheck?: {
    status: 'verified' | 'disputed' | 'unverified';
    source: string;
  };
}

const Search = () => {
  const [query, setQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('all');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [suggestions, setSuggestions] = useState<string[]>([]);

  const filters = [
    { id: 'all', name: 'All Sources', icon: Globe },
    { id: 'news', name: 'News', icon: TrendingUp },
    { id: 'academic', name: 'Academic', icon: Brain },
    { id: 'social', name: 'Social Media', icon: ExternalLink },
    { id: 'verified', name: 'Verified Only', icon: Shield }
  ];

  const trendingTopics = [
    'AI Model Training Efficiency',
    'Quantum Computing Breakthrough',
    'Neural Architecture Search',
    'Machine Learning Ethics',
    'Autonomous AI Systems',
    'Large Language Model Optimization'
  ];

  const generateMockResults = (searchQuery: string): SearchResult[] => {
    const mockData = [
      {
        title: 'Breakthrough in AI Model Training Efficiency Reduces Costs by 40%',
        source: 'MIT Technology Review',
        snippet: 'Researchers develop new optimization techniques that significantly reduce computational requirements for training large neural networks while maintaining performance...',
        url: 'https://www.technologyreview.com/ai-training-efficiency',
        bias: 'neutral' as const,
        timestamp: '2 hours ago',
        credibility: 9.2,
        category: 'Technology',
        relevance: 0.95,
        factCheck: { status: 'verified' as const, source: 'IEEE Fact Check' }
      },
      {
        title: 'New Quantum-Classical Hybrid Algorithm Shows Promise for Machine Learning',
        source: 'Nature',
        snippet: 'Scientists demonstrate quantum advantage in specific machine learning tasks, opening new possibilities for hybrid quantum-classical computing approaches...',
        url: 'https://www.nature.com/articles/quantum-ml-hybrid',
        bias: 'neutral' as const,
        timestamp: '4 hours ago',
        credibility: 9.8,
        category: 'Research',
        relevance: 0.88,
        factCheck: { status: 'verified' as const, source: 'Peer Review' }
      },
      {
        title: 'AI Ethics Framework Adopted by Major Tech Companies',
        source: 'AI Ethics Institute',
        snippet: 'Industry leaders unite around comprehensive guidelines for responsible AI development, emphasizing transparency, fairness, and human oversight...',
        url: 'https://aiethics.org/framework-2024',
        bias: 'neutral' as const,
        timestamp: '6 hours ago',
        credibility: 8.7,
        category: 'Ethics',
        relevance: 0.82,
        factCheck: { status: 'verified' as const, source: 'Multi-source' }
      },
      {
        title: 'Open Source Language Model Rivals Proprietary Alternatives',
        source: 'ArXiv Preprint',
        snippet: 'Academic consortium releases large language model that matches performance of commercial systems while maintaining full transparency and reproducibility...',
        url: 'https://arxiv.org/abs/2024.01234',
        bias: 'neutral' as const,
        timestamp: '8 hours ago',
        credibility: 8.9,
        category: 'Open Source',
        relevance: 0.79,
        factCheck: { status: 'unverified' as const, source: 'Pending Review' }
      },
      {
        title: 'AutoML Platform Democratizes AI Development for Small Businesses',
        source: 'TechCrunch',
        snippet: 'New automated machine learning platform enables companies without AI expertise to build and deploy custom models, lowering barriers to AI adoption...',
        url: 'https://techcrunch.com/automl-democratization',
        bias: 'neutral' as const,
        timestamp: '12 hours ago',
        credibility: 8.1,
        category: 'Business',
        relevance: 0.75,
        factCheck: { status: 'verified' as const, source: 'Industry Analysis' }
      },
      {
        title: 'Neural Architecture Search Breakthrough Automates AI Design',
        source: 'Google AI Blog',
        snippet: 'Researchers develop system that automatically discovers optimal neural network architectures, potentially revolutionizing how AI models are designed and optimized...',
        url: 'https://ai.googleblog.com/neural-architecture-search',
        bias: 'neutral' as const,
        timestamp: '1 day ago',
        credibility: 9.0,
        category: 'Research',
        relevance: 0.73,
        factCheck: { status: 'verified' as const, source: 'Peer Review' }
      }
    ];

    // Filter based on query relevance and active filter
    let filtered = mockData.filter(result => {
      if (!searchQuery.trim()) return true;
      
      const queryLower = searchQuery.toLowerCase();
      const titleMatch = result.title.toLowerCase().includes(queryLower);
      const snippetMatch = result.snippet.toLowerCase().includes(queryLower);
      const categoryMatch = result.category.toLowerCase().includes(queryLower);
      
      return titleMatch || snippetMatch || categoryMatch;
    });

    if (activeFilter !== 'all') {
      if (activeFilter === 'verified') {
        filtered = filtered.filter(result => result.factCheck?.status === 'verified');
      } else if (activeFilter === 'news') {
        filtered = filtered.filter(result => ['Technology', 'Business'].includes(result.category));
      } else if (activeFilter === 'academic') {
        filtered = filtered.filter(result => ['Research', 'Ethics'].includes(result.category));
      }
    }

    return filtered.map(result => ({
      ...result,
      id: `${result.source}-${Date.now()}-${Math.random()}`
    }));
  };

  useEffect(() => {
    if (query.length > 2) {
      const related = trendingTopics.filter(topic => 
        topic.toLowerCase().includes(query.toLowerCase())
      );
      setSuggestions(related.slice(0, 3));
    } else {
      setSuggestions([]);
    }
  }, [query]);

  const performSearch = async () => {
    if (!query.trim()) return;
    
    setIsSearching(true);
    
    // Simulate search delay
    await new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 500));
    
    const searchResults = generateMockResults(query);
    setResults(searchResults);
    setIsSearching(false);
  };

  const getBiasColor = (bias: string) => {
    const colors = {
      'neutral': 'text-green-400',
      'left': 'text-blue-400',
      'right': 'text-red-400',
      'mixed': 'text-yellow-400'
    };
    return colors[bias as keyof typeof colors] || 'text-slate-400';
  };

  const getCredibilityColor = (credibility: number) => {
    if (credibility >= 9) return 'text-green-400';
    if (credibility >= 8) return 'text-yellow-400';
    if (credibility >= 7) return 'text-orange-400';
    return 'text-red-400';
  };

  const getFactCheckBadge = (factCheck?: { status: string; source: string }) => {
    if (!factCheck) return null;
    
    const badges = {
      'verified': { color: 'bg-green-500/20 text-green-300 border-green-500/30', icon: Shield },
      'disputed': { color: 'bg-red-500/20 text-red-300 border-red-500/30', icon: ExternalLink },
      'unverified': { color: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30', icon: Clock }
    };
    
    const badge = badges[factCheck.status as keyof typeof badges];
    if (!badge) return null;
    
    const IconComponent = badge.icon;
    
    return (
      <div className={`inline-flex items-center space-x-1 px-2 py-1 rounded border text-xs ${badge.color}`}>
        <IconComponent className="w-3 h-3" />
        <span>{factCheck.status}</span>
      </div>
    );
  };

  return (
    <Layout>
      <div className="p-6 max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Unbiased AI-Powered Search</h1>
          <p className="text-slate-400">Advanced search with real-time fact-checking and bias detection</p>
        </div>

        {/* Search Interface */}
        <div className="bg-slate-800/50 backdrop-blur-xl rounded-xl border border-slate-700/50 p-6 mb-6">
          <div className="flex space-x-4 mb-4">
            <div className="flex-1 relative">
              <SearchIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-5 h-5" />
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && performSearch()}
                placeholder="Search for verified, unbiased information..."
                className="w-full bg-slate-700/50 border border-slate-600/50 rounded-lg pl-10 pr-4 py-3 text-white placeholder-slate-400 focus:border-cyan-500/50 focus:outline-none"
              />
              
              {/* Search Suggestions */}
              {suggestions.length > 0 && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-slate-800 border border-slate-600/50 rounded-lg z-10">
                  {suggestions.map((suggestion, index) => (
                    <button
                      key={index}
                      onClick={() => setQuery(suggestion)}
                      className="w-full text-left px-4 py-2 hover:bg-slate-700/50 text-slate-300 first:rounded-t-lg last:rounded-b-lg"
                    >
                      <Zap className="inline w-4 h-4 mr-2 text-cyan-400" />
                      {suggestion}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <button
              onClick={performSearch}
              disabled={isSearching}
              className="bg-gradient-to-r from-cyan-500 to-purple-500 px-6 py-3 rounded-lg font-medium transition-all disabled:opacity-50 hover:from-cyan-600 hover:to-purple-600"
            >
              {isSearching ? 'Searching...' : 'Search'}
            </button>
          </div>

          {/* Filters */}
          <div className="flex space-x-2 overflow-x-auto pb-2">
            {filters.map((filter) => {
              const IconComponent = filter.icon;
              return (
                <button
                  key={filter.id}
                  onClick={() => setActiveFilter(filter.id)}
                  className={`flex items-center space-x-2 px-3 py-2 rounded-lg text-sm transition-all whitespace-nowrap ${
                    activeFilter === filter.id
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'bg-slate-700/50 text-slate-400 hover:bg-slate-600/50'
                  }`}
                >
                  <IconComponent className="w-4 h-4" />
                  <span>{filter.name}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Search Results */}
        {results.length > 0 && (
          <div className="space-y-4 mb-8">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Search Results ({results.length})</h2>
              <div className="text-sm text-slate-400">
                Ranked by relevance and credibility
              </div>
            </div>
            
            {results.map((result) => (
              <div key={result.id} className="bg-slate-800/50 rounded-lg p-6 border border-slate-700/50 hover:bg-slate-800/70 transition-all">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <a 
                      href={result.url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="text-lg font-medium text-white hover:text-cyan-300 cursor-pointer block mb-2"
                    >
                      {result.title}
                    </a>
                    <div className="flex items-center space-x-4 mb-3">
                      <span className="text-sm text-slate-400">{result.source}</span>
                      <span className="text-xs text-slate-500">{result.timestamp}</span>
                      <span className="text-xs px-2 py-1 bg-slate-700/50 rounded text-slate-300">
                        {result.category}
                      </span>
                      <div className="flex items-center space-x-1">
                        <Star className={`w-3 h-3 ${getCredibilityColor(result.credibility)}`} />
                        <span className={`text-xs font-medium ${getCredibilityColor(result.credibility)}`}>
                          {result.credibility}/10
                        </span>
                      </div>
                    </div>
                  </div>
                  <ExternalLink className="w-5 h-5 text-slate-400 ml-4" />
                </div>
                
                <p className="text-slate-300 mb-4 leading-relaxed">{result.snippet}</p>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      <div className={`w-2 h-2 rounded-full ${
                        result.bias === 'neutral' ? 'bg-green-400' :
                        result.bias === 'mixed' ? 'bg-yellow-400' :
                        'bg-slate-400'
                      }`} />
                      <span className={`text-xs font-medium ${getBiasColor(result.bias)}`}>
                        {result.bias} bias
                      </span>
                    </div>
                    {getFactCheckBadge(result.factCheck)}
                  </div>
                  <div className="flex items-center space-x-1 text-xs text-slate-400">
                    <span>Relevance:</span>
                    <span className="text-cyan-400 font-medium">
                      {Math.round(result.relevance * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Trending Topics */}
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
            <TrendingUp className="w-5 h-5 text-red-400" />
            <span>Trending in AI & Technology</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {trendingTopics.map((topic, index) => (
              <button
                key={index}
                onClick={() => setQuery(topic)}
                className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50 hover:bg-slate-700/50 transition-all text-left group"
              >
                <div className="flex items-center space-x-2 mb-2">
                  <TrendingUp className="w-4 h-4 text-red-400" />
                  <span className="text-sm text-red-400 font-medium">Trending #{index + 1}</span>
                </div>
                <h3 className="font-medium text-white group-hover:text-cyan-300 transition-colors">{topic}</h3>
                <p className="text-sm text-slate-400 mt-1">
                  {Math.floor(Math.random() * 500) + 100} verified sources
                </p>
              </button>
            ))}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Search;
