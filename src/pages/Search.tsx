
import React, { useState } from 'react';
import { Layout } from '../components/Layout';
import { Search as SearchIcon, Globe, Filter, TrendingUp, ExternalLink, Shield } from 'lucide-react';

const Search = () => {
  const [query, setQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('all');
  const [results, setResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);

  const filters = [
    { id: 'all', name: 'All Sources' },
    { id: 'news', name: 'News' },
    { id: 'academic', name: 'Academic' },
    { id: 'social', name: 'Social Media' }
  ];

  const performSearch = async () => {
    if (!query.trim()) return;
    setIsSearching(true);
    
    // Simulate search delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const mockResults = [
      {
        id: '1',
        title: 'AI Model Training Breakthrough',
        source: 'Tech Research Journal',
        snippet: 'Recent advances in AI training methodologies show promising results...',
        url: '#',
        bias: 'neutral',
        timestamp: '2 hours ago'
      },
      {
        id: '2',
        title: 'Ethical AI Development Guidelines',
        source: 'AI Ethics Institute',
        snippet: 'New framework for responsible AI development released...',
        url: '#',
        bias: 'neutral',
        timestamp: '5 hours ago'
      }
    ];
    
    setResults(mockResults);
    setIsSearching(false);
  };

  return (
    <Layout>
      <div className="p-6 max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Unbiased Search & News</h1>
          <p className="text-slate-400">AI-powered search with source transparency and bias detection</p>
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
                placeholder="Search for unbiased information..."
                className="w-full bg-slate-700/50 border border-slate-600/50 rounded-lg pl-10 pr-4 py-3 text-white placeholder-slate-400"
              />
            </div>
            <button
              onClick={performSearch}
              disabled={isSearching}
              className="bg-gradient-to-r from-cyan-500 to-purple-500 px-6 py-3 rounded-lg font-medium transition-all disabled:opacity-50"
            >
              {isSearching ? 'Searching...' : 'Search'}
            </button>
          </div>

          {/* Filters */}
          <div className="flex space-x-2">
            {filters.map((filter) => (
              <button
                key={filter.id}
                onClick={() => setActiveFilter(filter.id)}
                className={`px-3 py-1 rounded-lg text-sm transition-all ${
                  activeFilter === filter.id
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                    : 'bg-slate-700/50 text-slate-400 hover:bg-slate-600/50'
                }`}
              >
                {filter.name}
              </button>
            ))}
          </div>
        </div>

        {/* Results */}
        {results.length > 0 && (
          <div className="space-y-4">
            <h2 className="text-xl font-semibold">Search Results</h2>
            {results.map((result) => (
              <div key={result.id} className="bg-slate-800/50 rounded-lg p-6 border border-slate-700/50">
                <div className="flex items-start justify-between mb-2">
                  <h3 className="text-lg font-medium text-white hover:text-cyan-300 cursor-pointer">
                    {result.title}
                  </h3>
                  <ExternalLink className="w-4 h-4 text-slate-400" />
                </div>
                <p className="text-slate-300 mb-3">{result.snippet}</p>
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <span className="text-sm text-slate-400">{result.source}</span>
                    <span className="text-xs text-slate-500">{result.timestamp}</span>
                    <div className="flex items-center space-x-1">
                      <Shield className="w-3 h-3 text-green-400" />
                      <span className="text-xs text-green-400">Verified Neutral</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Trending Topics */}
        <div className="mt-8">
          <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
            <TrendingUp className="w-5 h-5" />
            <span>Trending Topics</span>
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {['AI Safety Research', 'Quantum Computing Advances', 'Climate Tech Innovation'].map((topic) => (
              <div key={topic} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
                <h3 className="font-medium mb-2">{topic}</h3>
                <p className="text-sm text-slate-400">Multiple verified sources</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default Search;
