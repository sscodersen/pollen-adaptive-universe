import { useState, useEffect } from 'react';
import { 
  ArrowLeft, Search as SearchIcon, Sparkles, TrendingUp, 
  Lightbulb, Zap, Globe, Briefcase, Home, ShoppingBag,
  MapPin, Target, Award, Clock, ChevronRight, Bot
} from "lucide-react";
import { BottomNav } from "./BottomNav";
import { sseWorkerBot, ContentSuggestion, UGCAdContent } from '../services/sseWorkerBot';
import { opportunityCurationService, TrendOpportunity, RealEstateOpportunity } from '../services/opportunityCuration';

interface EnhancedExploreProps {
  onNavigate: (screen: 'feed' | 'explore' | 'shop') => void;
}

export function EnhancedExplore({ onNavigate }: EnhancedExploreProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeSection, setActiveSection] = useState<'trending' | 'real-estate' | 'content-creator' | 'all'>('all');
  const [contentSuggestions, setContentSuggestions] = useState<ContentSuggestion[]>([]);
  const [trendingOpportunities, setTrendingOpportunities] = useState<TrendOpportunity[]>([]);
  const [realEstateOpps, setRealEstateOpps] = useState<RealEstateOpportunity[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    loadOpportunities();
  }, []);

  const loadOpportunities = async () => {
    const [trending, realEstate] = await Promise.all([
      opportunityCurationService.getTrendingOpportunities(),
      opportunityCurationService.getRealEstateOpportunities()
    ]);
    
    setTrendingOpportunities(trending);
    setRealEstateOpps(realEstate);
  };

  const handleGenerateContent = async () => {
    if (!searchQuery.trim()) return;
    
    setIsGenerating(true);
    try {
      const suggestions = await sseWorkerBot.generateContentSuggestions(searchQuery, 3);
      setContentSuggestions(suggestions);
    } catch (error) {
      console.error('Failed to generate content:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    handleGenerateContent();
  };

  return (
    <div className="relative min-h-screen pb-32 bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50">
      {/* Header */}
      <div className="p-4 sm:p-6 pt-6 sm:pt-8">
        <div className="flex items-center gap-3 mb-6">
          <button 
            onClick={() => onNavigate('feed')} 
            className="w-10 h-10 rounded-full bg-white/80 backdrop-blur-sm flex items-center justify-center hover:bg-white transition-all"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <form onSubmit={handleSearch} className="flex-1 glass-card px-4 py-3 rounded-2xl flex items-center gap-3">
            <SearchIcon className="w-5 h-5 text-gray-400" />
            <input 
              type="text" 
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="What would you like to create or discover?"
              className="flex-1 bg-transparent outline-none text-sm text-gray-600"
            />
            {searchQuery && (
              <button 
                type="submit"
                className="bg-purple-600 text-white p-2 rounded-lg hover:bg-purple-700 transition-all"
              >
                <Sparkles className="w-4 h-4" />
              </button>
            )}
          </form>
        </div>

        {/* Section Tabs */}
        <div className="flex gap-2 mb-6 overflow-x-auto scrollbar-thin pb-2">
          <button 
            onClick={() => setActiveSection('all')}
            className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap transition-all ${
              activeSection === 'all' ? 'bg-white/90' : 'bg-white/50 text-gray-600'
            }`}
          >
            All
          </button>
          <button 
            onClick={() => setActiveSection('trending')}
            className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap flex items-center gap-2 transition-all ${
              activeSection === 'trending' ? 'bg-white/90' : 'bg-white/50 text-gray-600'
            }`}
          >
            <TrendingUp className="w-4 h-4" />
            Trending
          </button>
          <button 
            onClick={() => setActiveSection('real-estate')}
            className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap flex items-center gap-2 transition-all ${
              activeSection === 'real-estate' ? 'bg-white/90' : 'bg-white/50 text-gray-600'
            }`}
          >
            <Home className="w-4 h-4" />
            Real Estate
          </button>
          <button 
            onClick={() => setActiveSection('content-creator')}
            className={`px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap flex items-center gap-2 transition-all ${
              activeSection === 'content-creator' ? 'bg-white/90' : 'bg-white/50 text-gray-600'
            }`}
          >
            <Bot className="w-4 h-4" />
            AI Creator
          </button>
        </div>

        <div className="space-y-4">
          {/* SSE Worker Bot - Content Creation */}
          {(activeSection === 'all' || activeSection === 'content-creator') && (
            <div className="glass-card p-6 shadow-lg border-2 border-purple-200">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                  <Bot className="w-6 h-6 text-white" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-lg">AI Content Creator</h3>
                  <p className="text-xs text-gray-600">Powered by SSE Worker Bot</p>
                </div>
              </div>
              
              <p className="text-gray-600 text-sm mb-4">
                Generate UGC ads, social posts, and creative content with AI assistance
              </p>

              {isGenerating && (
                <div className="bg-purple-50 rounded-xl p-4 mb-4">
                  <div className="flex items-center gap-3">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-purple-600"></div>
                    <span className="text-sm text-gray-600">AI is crafting your content...</span>
                  </div>
                </div>
              )}

              {contentSuggestions.length > 0 && (
                <div className="space-y-3 mb-4">
                  {contentSuggestions.map((suggestion) => (
                    <div key={suggestion.id} className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-4">
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex-1">
                          <span className="text-xs bg-purple-200 text-purple-800 px-2 py-1 rounded-full">
                            {suggestion.type}
                          </span>
                          <h4 className="font-semibold text-gray-800 mt-2">{suggestion.title}</h4>
                        </div>
                        <Clock className="w-4 h-4 text-gray-400" />
                      </div>
                      <p className="text-xs text-gray-600 mb-2">{suggestion.description}</p>
                      <div className="flex items-center justify-between mt-3">
                        <span className="text-xs text-gray-500">{suggestion.estimatedTime}</span>
                        <button className="text-purple-600 text-xs font-medium flex items-center gap-1">
                          Use Template <ChevronRight className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <div className="grid grid-cols-2 gap-3">
                <button className="bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 rounded-xl font-medium hover:shadow-lg transition-all text-sm">
                  Create UGC Ad
                </button>
                <button className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white py-3 rounded-xl font-medium hover:shadow-lg transition-all text-sm">
                  Social Post
                </button>
              </div>
            </div>
          )}

          {/* Trending Opportunities */}
          {(activeSection === 'all' || activeSection === 'trending') && trendingOpportunities.length > 0 && (
            <div className="glass-card p-6 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-6 h-6 text-green-600" />
                  <h3 className="font-semibold text-lg">Trending Now</h3>
                </div>
                <Zap className="w-5 h-5 text-yellow-500" />
              </div>
              
              <div className="space-y-3">
                {trendingOpportunities.map((trend) => (
                  <div key={trend.id} className="bg-gradient-to-r from-green-50 to-emerald-50 rounded-xl p-4">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-semibold text-gray-800 flex-1">{trend.title}</h4>
                      <div className="flex items-center gap-1 ml-2">
                        <TrendingUp className="w-4 h-4 text-green-600" />
                        <span className="text-xs font-bold text-green-600">
                          {trend.trendingScore.toFixed(1)}
                        </span>
                      </div>
                    </div>
                    <p className="text-xs text-gray-600 mb-3 line-clamp-2">{trend.description}</p>
                    
                    <div className="flex items-center justify-between mb-3">
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        trend.momentum === 'rising' ? 'bg-green-200 text-green-800' :
                        trend.momentum === 'stable' ? 'bg-blue-200 text-blue-800' :
                        'bg-gray-200 text-gray-800'
                      }`}>
                        {trend.momentum}
                      </span>
                      <span className="text-xs text-gray-500">Peak: {trend.predictedPeak}</span>
                    </div>

                    {trend.aiInsights && trend.aiInsights.length > 0 && (
                      <div className="bg-white/60 rounded-lg p-2 mb-2">
                        <p className="text-xs text-purple-600 flex items-start gap-1">
                          <Lightbulb className="w-3 h-3 mt-0.5 flex-shrink-0" />
                          {trend.aiInsights[0]}
                        </p>
                      </div>
                    )}

                    <div className="flex gap-2 mt-2">
                      {trend.relatedTopics.slice(0, 3).map(topic => (
                        <span key={topic} className="text-xs bg-white/60 px-2 py-1 rounded-full">
                          {topic}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Real Estate Opportunities */}
          {(activeSection === 'all' || activeSection === 'real-estate') && realEstateOpps.length > 0 && (
            <div className="glass-card p-6 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Home className="w-6 h-6 text-blue-600" />
                  <h3 className="font-semibold text-lg">Real Estate</h3>
                </div>
                <Award className="w-5 h-5 text-yellow-500" />
              </div>
              
              <div className="space-y-3">
                {realEstateOpps.map((property) => (
                  <div key={property.id} className="bg-gradient-to-r from-blue-50 to-cyan-50 rounded-xl p-4">
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex-1">
                        <h4 className="font-semibold text-gray-800">{property.title}</h4>
                        <div className="flex items-center gap-2 mt-1">
                          <MapPin className="w-3 h-3 text-gray-500" />
                          <span className="text-xs text-gray-600">{property.location}</span>
                        </div>
                      </div>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        property.marketTrend === 'bullish' ? 'bg-green-200 text-green-800' :
                        property.marketTrend === 'neutral' ? 'bg-blue-200 text-blue-800' :
                        'bg-red-200 text-red-800'
                      }`}>
                        {property.marketTrend}
                      </span>
                    </div>

                    <p className="text-xs text-gray-600 mb-3">{property.description}</p>

                    <div className="grid grid-cols-2 gap-2 mb-3">
                      <div className="bg-white/60 rounded-lg p-2">
                        <p className="text-xs text-gray-500">Price Range</p>
                        <p className="text-sm font-semibold text-gray-800">{property.priceRange}</p>
                      </div>
                      <div className="bg-white/60 rounded-lg p-2">
                        <p className="text-xs text-gray-500">ROI</p>
                        <p className="text-sm font-semibold text-green-600">{property.roi}%</p>
                      </div>
                    </div>

                    {property.aiInsights && property.aiInsights.length > 0 && (
                      <div className="bg-white/60 rounded-lg p-2">
                        <p className="text-xs text-blue-600 flex items-start gap-1">
                          <Target className="w-3 h-3 mt-0.5 flex-shrink-0" />
                          {property.aiInsights[0]}
                        </p>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Discover More Section */}
          {activeSection === 'all' && (
            <div className="glass-card p-6 shadow-lg">
              <h3 className="font-semibold text-lg mb-4">Explore Categories</h3>
              <div className="grid grid-cols-2 gap-3">
                <button className="bg-gradient-to-br from-purple-100 to-pink-100 p-4 rounded-xl text-left hover:shadow-md transition-all">
                  <ShoppingBag className="w-6 h-6 text-purple-600 mb-2" />
                  <p className="font-medium text-sm">Products</p>
                  <p className="text-xs text-gray-600">Latest finds</p>
                </button>
                <button className="bg-gradient-to-br from-blue-100 to-cyan-100 p-4 rounded-xl text-left hover:shadow-md transition-all">
                  <Globe className="w-6 h-6 text-blue-600 mb-2" />
                  <p className="font-medium text-sm">Travel</p>
                  <p className="text-xs text-gray-600">Best deals</p>
                </button>
                <button className="bg-gradient-to-br from-green-100 to-emerald-100 p-4 rounded-xl text-left hover:shadow-md transition-all">
                  <Briefcase className="w-6 h-6 text-green-600 mb-2" />
                  <p className="font-medium text-sm">Apps</p>
                  <p className="text-xs text-gray-600">Trending apps</p>
                </button>
                <button className="bg-gradient-to-br from-yellow-100 to-orange-100 p-4 rounded-xl text-left hover:shadow-md transition-all">
                  <Lightbulb className="w-6 h-6 text-yellow-600 mb-2" />
                  <p className="font-medium text-sm">Lifestyle</p>
                  <p className="text-xs text-gray-600">Curated tips</p>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
      
      <BottomNav currentScreen="explore" onNavigate={onNavigate} />
    </div>
  );
}
