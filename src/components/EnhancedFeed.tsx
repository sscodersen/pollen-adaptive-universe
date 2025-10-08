import { useState, useEffect } from 'react';
import { 
  Clock, Sun, Search, TrendingUp, Flame, Shield, Heart, 
  Sprout, HandHeart, Sparkles, ChevronRight, CheckCircle, 
  AlertTriangle, Droplets, ThermometerSun, Vote
} from "lucide-react";
import { BottomNav } from "./BottomNav";
import { wellnessContentService, WellnessTip } from '../services/wellnessContent';
import { socialImpactService, SocialInitiative } from '../services/socialImpact';
import { opportunityCurationService, Opportunity } from '../services/opportunityCuration';

interface EnhancedFeedProps {
  onNavigate: (screen: 'feed' | 'explore' | 'shop') => void;
}

export function EnhancedFeed({ onNavigate }: EnhancedFeedProps) {
  const [activeTab, setActiveTab] = useState<'all' | 'wellness' | 'agriculture' | 'social-impact' | 'opportunities'>('all');
  const [dailyTip, setDailyTip] = useState<WellnessTip | null>(null);
  const [initiatives, setInitiatives] = useState<SocialInitiative[]>([]);
  const [opportunities, setOpportunities] = useState<Opportunity[]>([]);
  const [votedInitiatives, setVotedInitiatives] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadContent();
  }, []);

  const loadContent = async () => {
    const [tip, socialInitiatives, curatedOpps] = await Promise.all([
      wellnessContentService.getDailyTip(),
      socialImpactService.getCuratedInitiatives(3),
      opportunityCurationService.getCuratedOpportunities(undefined, 4)
    ]);
    
    setDailyTip(tip);
    setInitiatives(socialInitiatives);
    setOpportunities(curatedOpps);
  };

  const handleVote = async (initiativeId: string) => {
    const success = await socialImpactService.voteForInitiative('user-1', initiativeId);
    if (success) {
      setVotedInitiatives(prev => new Set([...prev, initiativeId]));
    }
  };

  return (
    <div className="relative min-h-screen pb-32 bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      {/* Header */}
      <div className="p-4 sm:p-6 pt-6 sm:pt-8">
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-gradient-to-br from-purple-400 to-pink-400 flex items-center justify-center text-white font-semibold text-base sm:text-lg">
              J
            </div>
            <div className="flex items-center gap-2 glass-card px-3 sm:px-4 py-1.5 sm:py-2">
              <Clock className="w-4 h-4 text-gray-600" />
              <span className="text-xl sm:text-2xl font-semibold">Hey Jane,</span>
            </div>
          </div>
          <div className="glass-card px-3 sm:px-4 py-1.5 sm:py-2 flex items-center gap-2">
            <Sun className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-500" />
            <span className="text-base sm:text-lg font-medium">11°C</span>
          </div>
        </div>
        
        <h2 className="text-lg sm:text-xl font-medium text-gray-800 mb-4">Your AI-Powered Feed</h2>
        
        {/* Search Bar */}
        <button 
          onClick={() => onNavigate('explore')}
          className="w-full glass-card p-4 rounded-2xl mb-6 text-left flex items-center gap-3 hover:bg-white/60 transition-all"
        >
          <Search className="w-5 h-5 text-gray-400" />
          <span className="text-gray-500">Ask Pollen anything...</span>
        </button>

        {/* Feed Categories */}
        <div className="flex gap-2 mb-6 overflow-x-auto scrollbar-thin pb-2">
          <button 
            onClick={() => setActiveTab('all')}
            className={`px-6 py-2 rounded-full text-sm font-medium whitespace-nowrap flex items-center gap-2 transition-all ${
              activeTab === 'all' ? 'bg-white/90' : 'bg-white/50 text-gray-600'
            }`}
          >
            <Flame className="w-4 h-4" />
            All Posts
          </button>
          <button 
            onClick={() => setActiveTab('wellness')}
            className={`px-6 py-2 rounded-full text-sm font-medium whitespace-nowrap flex items-center gap-2 transition-all ${
              activeTab === 'wellness' ? 'bg-white/90' : 'bg-white/50 text-gray-600'
            }`}
          >
            <Heart className="w-4 h-4" />
            Wellness
          </button>
          <button 
            onClick={() => setActiveTab('agriculture')}
            className={`px-6 py-2 rounded-full text-sm font-medium whitespace-nowrap flex items-center gap-2 transition-all ${
              activeTab === 'agriculture' ? 'bg-white/90' : 'bg-white/50 text-gray-600'
            }`}
          >
            <Sprout className="w-4 h-4" />
            Agriculture
          </button>
          <button 
            onClick={() => setActiveTab('social-impact')}
            className={`px-6 py-2 rounded-full text-sm font-medium whitespace-nowrap flex items-center gap-2 transition-all ${
              activeTab === 'social-impact' ? 'bg-white/90' : 'bg-white/50 text-gray-600'
            }`}
          >
            <HandHeart className="w-4 h-4" />
            Social Impact
          </button>
          <button 
            onClick={() => setActiveTab('opportunities')}
            className={`px-6 py-2 rounded-full text-sm font-medium whitespace-nowrap flex items-center gap-2 transition-all ${
              activeTab === 'opportunities' ? 'bg-white/90' : 'bg-white/50 text-gray-600'
            }`}
          >
            <Sparkles className="w-4 h-4" />
            Opportunities
          </button>
        </div>

        {/* Content Sections */}
        <div className="space-y-4">
          {/* Content Verification Feature */}
          {(activeTab === 'all') && (
            <div className="glass-card p-6 shadow-lg border-2 border-purple-200">
              <div className="flex items-center gap-3 mb-4">
                <Shield className="w-6 h-6 text-purple-600" />
                <h3 className="font-semibold text-lg">Content Verification</h3>
              </div>
              <p className="text-gray-600 text-sm mb-4">
                Verify content authenticity with AI-powered deepfake detection
              </p>
              <button className="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 rounded-xl font-medium hover:shadow-lg transition-all">
                Upload Content to Verify
              </button>
              <div className="mt-4 flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-4 h-4 text-green-500" />
                  <span className="text-gray-600">94% Accuracy</span>
                </div>
                <div className="flex items-center gap-2">
                  <Shield className="w-4 h-4 text-purple-500" />
                  <span className="text-gray-600">Secure Analysis</span>
                </div>
              </div>
            </div>
          )}

          {/* Daily Wellness Tip */}
          {(activeTab === 'all' || activeTab === 'wellness') && dailyTip && (
            <div className="gradient-card-pink p-6 shadow-lg">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Heart className="w-5 h-5 text-pink-600" />
                  <h3 className="font-semibold text-lg">Daily Wellness Tip</h3>
                </div>
                <span className="text-xs bg-pink-200 text-pink-800 px-3 py-1 rounded-full">
                  {dailyTip.duration}
                </span>
              </div>
              <h4 className="font-semibold text-gray-800 mb-2">{dailyTip.title}</h4>
              <p className="text-gray-600 text-sm mb-4">{dailyTip.content}</p>
              <div className="flex items-center justify-between">
                <div className="flex gap-2">
                  {dailyTip.tags.slice(0, 2).map(tag => (
                    <span key={tag} className="text-xs bg-white/60 px-3 py-1 rounded-full">
                      #{tag}
                    </span>
                  ))}
                </div>
                <button className="text-purple-600 text-sm font-medium flex items-center gap-1">
                  Read More <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* Agriculture Tools */}
          {(activeTab === 'all' || activeTab === 'agriculture') && (
            <div className="gradient-card-green p-6 shadow-lg">
              <div className="flex items-center gap-3 mb-4">
                <Sprout className="w-6 h-6 text-green-700" />
                <h3 className="font-semibold text-lg">Smart Farming Tools</h3>
              </div>
              <p className="text-gray-700 text-sm mb-4">
                AI-powered insights for precision agriculture
              </p>
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="bg-white/60 rounded-xl p-3">
                  <Droplets className="w-5 h-5 text-blue-600 mb-2" />
                  <p className="text-sm font-medium">Soil Analysis</p>
                  <p className="text-xs text-gray-600">pH 6.5, Medium N</p>
                </div>
                <div className="bg-white/60 rounded-xl p-3">
                  <ThermometerSun className="w-5 h-5 text-orange-600 mb-2" />
                  <p className="text-sm font-medium">Weather</p>
                  <p className="text-xs text-gray-600">Rain in 3 days</p>
                </div>
              </div>
              <button className="w-full bg-green-600 text-white py-3 rounded-xl font-medium hover:bg-green-700 transition-all">
                Get Crop Recommendations
              </button>
            </div>
          )}

          {/* Social Impact Initiatives */}
          {(activeTab === 'all' || activeTab === 'social-impact') && (
            <div className="glass-card p-6 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <HandHeart className="w-6 h-6 text-purple-600" />
                  <h3 className="font-semibold text-lg">Social Impact</h3>
                </div>
                <span className="text-xs bg-purple-100 text-purple-800 px-3 py-1 rounded-full">
                  AI Curated
                </span>
              </div>
              <div className="space-y-3">
                {initiatives.map((initiative) => (
                  <div key={initiative.id} className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-4">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-semibold text-gray-800 text-sm flex-1">{initiative.title}</h4>
                      <span className="text-xs bg-green-100 text-green-800 px-2 py-1 rounded-full ml-2">
                        {initiative.aiQualityScore.toFixed(1)}/10
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 mb-3 line-clamp-2">{initiative.description}</p>
                    <div className="flex items-center justify-between">
                      <div className="text-xs text-gray-500">
                        ${initiative.currentFunding.toLocaleString()} / ${initiative.fundingGoal.toLocaleString()}
                      </div>
                      <button 
                        onClick={() => handleVote(initiative.id)}
                        disabled={votedInitiatives.has(initiative.id)}
                        className={`flex items-center gap-1 px-3 py-1 rounded-full text-xs font-medium transition-all ${
                          votedInitiatives.has(initiative.id)
                            ? 'bg-gray-200 text-gray-500 cursor-not-allowed'
                            : 'bg-purple-600 text-white hover:bg-purple-700'
                        }`}
                      >
                        <Vote className="w-3 h-3" />
                        {votedInitiatives.has(initiative.id) ? 'Voted' : 'Vote'}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Curated Opportunities */}
          {(activeTab === 'all' || activeTab === 'opportunities') && opportunities.length > 0 && (
            <div className="glass-card p-6 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-6 h-6 text-yellow-600" />
                  <h3 className="font-semibold text-lg">Curated Opportunities</h3>
                </div>
                <TrendingUp className="w-5 h-5 text-green-600" />
              </div>
              <div className="space-y-3">
                {opportunities.slice(0, 3).map((opp) => (
                  <div key={opp.id} className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-xl p-4">
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs bg-yellow-200 text-yellow-800 px-2 py-0.5 rounded-full">
                            {opp.type}
                          </span>
                          {opp.urgency === 'high' && (
                            <AlertTriangle className="w-3 h-3 text-orange-600" />
                          )}
                        </div>
                        <h4 className="font-semibold text-gray-800 text-sm">{opp.title}</h4>
                      </div>
                      <span className="text-xs font-bold text-purple-600 ml-2">
                        {opp.relevanceScore.toFixed(1)}
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 mb-2 line-clamp-2">{opp.description}</p>
                    {opp.aiInsights && opp.aiInsights.length > 0 && (
                      <p className="text-xs text-purple-600 italic">💡 {opp.aiInsights[0]}</p>
                    )}
                  </div>
                ))}
              </div>
              <button 
                onClick={() => onNavigate('explore')}
                className="w-full mt-4 text-purple-600 font-medium text-sm flex items-center justify-center gap-2 hover:text-purple-700 transition-all"
              >
                Explore All Opportunities <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          )}
        </div>
      </div>
      
      <BottomNav currentScreen="feed" onNavigate={onNavigate} />
    </div>
  );
}
