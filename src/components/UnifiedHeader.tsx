
import React, { useState, useEffect } from 'react';
import { Search, Sparkles, Brain, TrendingUp, Zap } from 'lucide-react';
import { globalSearch } from '../services/globalSearch';
import { useNavigate } from 'react-router-dom';

interface SearchResult {
  id: string;
  title: string;
  description: string;
  type: string;
  significance: number;
  url: string;
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
  const navigate = useNavigate();

  useEffect(() => {
    loadInsights();
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

  return (
    <div className="sticky top-0 z-50 bg-gray-900/95 backdrop-blur-xl border-b border-gray-800/50">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-6">
            <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl flex items-center justify-center shadow-lg">
              <Sparkles className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                {title}
              </h1>
              <p className="text-slate-400 mt-1">{subtitle}</p>
            </div>
          </div>

          {/* Global Search */}
          <div className="relative flex-1 max-w-2xl mx-8">
            <div className="relative">
              <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                placeholder="Search across all domains..."
                className="w-full bg-slate-800/60 border border-slate-700/50 rounded-xl pl-12 pr-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:border-cyan-500/50 transition-all"
              />
            </div>

            {/* Search Results Dropdown */}
            {showResults && searchResults.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-2 bg-slate-800/95 backdrop-blur-xl border border-slate-700/50 rounded-xl shadow-2xl max-h-96 overflow-y-auto z-50">
                {searchResults.map((result) => (
                  <div
                    key={result.id}
                    onClick={() => handleResultClick(result)}
                    className="p-4 hover:bg-slate-700/50 cursor-pointer border-b border-slate-700/30 last:border-b-0 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="text-white font-medium">{result.title}</h3>
                      <div className="flex items-center space-x-2">
                        <span className="px-2 py-1 bg-cyan-500/20 text-cyan-300 rounded text-xs border border-cyan-500/30">
                          {result.type}
                        </span>
                        <span className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-xs border border-purple-500/30">
                          {result.significance.toFixed(1)}
                        </span>
                      </div>
                    </div>
                    <p className="text-slate-400 text-sm line-clamp-2">{result.description}</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Status Indicators */}
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

        {/* Real-time Insights Bar */}
        {insights && (
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center space-x-6 text-slate-400">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse"></div>
                <span>{insights.totalContent.toLocaleString()} items processed</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span>{insights.highSignificance} high-significance</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                <span>Cross-domain intelligence active</span>
              </div>
            </div>
            
            <div className="px-3 py-1 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 text-cyan-300 rounded-full border border-cyan-500/20 text-xs font-medium">
              Next-Gen Platform • Production Ready
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
