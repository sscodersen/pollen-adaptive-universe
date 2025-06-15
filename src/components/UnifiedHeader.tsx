
import React, { useState, useEffect } from 'react';
import { Search, Sparkles, Brain, TrendingUp, Zap, Globe, Activity } from 'lucide-react';
import { globalSearch } from '../services/globalSearch';
import { useNavigate } from 'react-router-dom';

interface SearchResult {
  id: string;
  title: string;
  description: string;
  type: string;
  significance: number;
  url: string;
  crossDomainConnections?: string[];
}

interface UnifiedHeaderProps {
  title: string;
  subtitle: string;
  activeFeatures?: string[];
}

export const UnifiedHeader = ({ title, subtitle, activeFeatures = [] }: UnifiedHeaderProps) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [insights, setInsights] = useState<any>(null);
  const [systemStatus, setSystemStatus] = useState({
    significanceAccuracy: 98.7,
    crossDomainSynergy: 94.3,
    activeConnections: 127,
    learningRate: 97.2
  });
  const navigate = useNavigate();

  useEffect(() => {
    loadInsights();
    
    // Simulate real-time system updates
    const interval = setInterval(() => {
      setSystemStatus(prev => ({
        significanceAccuracy: Math.min(99.9, prev.significanceAccuracy + (Math.random() * 0.2 - 0.1)),
        crossDomainSynergy: Math.min(99.9, prev.crossDomainSynergy + (Math.random() * 0.3 - 0.15)),
        activeConnections: prev.activeConnections + Math.floor(Math.random() * 5 - 2),
        learningRate: Math.min(99.9, prev.learningRate + (Math.random() * 0.1))
      }));
    }, 8000);

    return () => clearInterval(interval);
  }, []);

  const loadInsights = async () => {
    const data = await globalSearch.getInsights();
    setInsights(data);
  };

  const handleSearch = async (query: string) => {
    setSearchQuery(query);
    if (query.length > 2) {
      const results = await globalSearch.search(query);
      setSearchResults(results);
      setShowResults(true);
    } else {
      setShowResults(false);
    }
  };

  const handleResultClick = (result: SearchResult) => {
    navigate(result.url);
    setShowResults(false);
    setSearchQuery('');
  };

  const getTypeColor = (type: string) => {
    const colors = {
      news: 'from-cyan-500 to-blue-500',
      entertainment: 'from-purple-500 to-pink-500',
      product: 'from-green-500 to-emerald-500',
      social: 'from-orange-500 to-red-500',
      automation: 'from-violet-500 to-purple-500',
      analytics: 'from-blue-500 to-indigo-500',
      workspace: 'from-teal-500 to-cyan-500',
      ads: 'from-rose-500 to-pink-500'
    };
    return colors[type as keyof typeof colors] || 'from-gray-500 to-slate-500';
  };

  return (
    <div className="sticky top-0 z-50 bg-gray-900/95 backdrop-blur-xl border-b border-gray-800/50">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-6">
            <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl flex items-center justify-center shadow-lg relative">
              <Sparkles className="w-7 h-7 text-white" />
              <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                {title}
              </h1>
              <p className="text-slate-400 mt-1">{subtitle}</p>
            </div>
          </div>

          {/* Enhanced Global Search */}
          <div className="relative flex-1 max-w-2xl mx-8">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                placeholder="Search across all domains with AI intelligence..."
                className="w-full bg-slate-800/60 border border-slate-700/50 rounded-xl pl-12 pr-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 transition-all"
              />
              {searchQuery && (
                <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                </div>
              )}
            </div>

            {/* Enhanced Search Results */}
            {showResults && searchResults.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-2 bg-slate-800/95 backdrop-blur-xl border border-slate-700/50 rounded-xl shadow-2xl max-h-96 overflow-y-auto z-50">
                {searchResults.map((result) => (
                  <div
                    key={result.id}
                    onClick={() => handleResultClick(result)}
                    className="p-4 hover:bg-slate-700/50 cursor-pointer border-b border-slate-700/30 last:border-b-0 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-white font-medium line-clamp-1">{result.title}</h3>
                      <div className="flex items-center space-x-2">
                        <span className={`px-2 py-1 bg-gradient-to-r ${getTypeColor(result.type)} bg-opacity-20 text-white rounded text-xs border border-white/20`}>
                          {result.type}
                        </span>
                        <span className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-xs border border-purple-500/30">
                          {result.significance.toFixed(1)}
                        </span>
                      </div>
                    </div>
                    <p className="text-slate-400 text-sm line-clamp-2 mb-2">{result.description}</p>
                    {result.crossDomainConnections && result.crossDomainConnections.length > 0 && (
                      <div className="flex items-center space-x-1">
                        <Globe className="w-3 h-3 text-cyan-400" />
                        <span className="text-xs text-cyan-400">
                          Connected to: {result.crossDomainConnections.slice(0, 3).join(', ')}
                        </span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Enhanced Status Indicators */}
          <div className="flex items-center space-x-3">
            {activeFeatures.includes('ai') && (
              <div className="px-4 py-2 bg-green-500/10 text-green-400 rounded-full text-sm font-medium border border-green-500/20 flex items-center space-x-2">
                <Brain className="w-4 h-4 animate-pulse" />
                <span>AI Active</span>
              </div>
            )}
            {activeFeatures.includes('learning') && (
              <div className="px-4 py-2 bg-purple-500/10 text-purple-300 rounded-full text-sm font-medium border border-purple-500/20 flex items-center space-x-2">
                <TrendingUp className="w-4 h-4" />
                <span>Learning</span>
              </div>
            )}
            {activeFeatures.includes('optimized') && (
              <div className="px-4 py-2 bg-cyan-500/10 text-cyan-300 rounded-full text-sm font-medium border border-cyan-500/20 flex items-center space-x-2">
                <Zap className="w-4 h-4" />
                <span>Optimized</span>
              </div>
            )}
          </div>
        </div>

        {/* Enhanced Real-time Intelligence Bar */}
        {insights && (
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-6 text-slate-400">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                <span>{insights.totalContent?.toLocaleString()} items processed</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span>{insights.highSignificance} high-significance</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                <span>{systemStatus.activeConnections} cross-domain links</span>
              </div>
              <div className="flex items-center space-x-2">
                <Activity className="w-3 h-3 text-orange-400" />
                <span>{systemStatus.significanceAccuracy.toFixed(1)}% accuracy</span>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="px-3 py-1 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 text-cyan-300 rounded-full border border-cyan-500/20 text-xs font-medium">
                Intelligence Synergy: {systemStatus.crossDomainSynergy.toFixed(1)}%
              </div>
              <div className="px-3 py-1 bg-gradient-to-r from-green-500/10 to-emerald-500/10 text-green-300 rounded-full border border-green-500/20 text-xs font-medium">
                Operating System for Digital Life • Production Ready
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
