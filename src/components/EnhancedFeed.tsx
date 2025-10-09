import { useState, useEffect, useCallback, memo } from 'react';
import { 
  Clock, Sun, Search, TrendingUp, Flame, Shield, Heart, 
  Sprout, HandHeart, Sparkles, ChevronRight, CheckCircle, 
  AlertTriangle, Droplets, ThermometerSun, Vote
} from "lucide-react";
import { BottomNav } from "./BottomNav";
import { wellnessContentService, WellnessTip } from '../services/wellnessContent';
import { socialImpactService, SocialInitiative } from '../services/socialImpact';
import { opportunityCurationService, Opportunity } from '../services/opportunityCuration';
import type { FeedTab } from '../types/feed';

interface EnhancedFeedProps {
  onNavigate: (screen: 'feed' | 'explore' | 'shop' | 'community') => void;
}

interface TabConfig {
  id: FeedTab;
  label: string;
  icon: typeof Flame;
}

const FEED_TABS: readonly TabConfig[] = [
  { id: 'all', label: 'All Posts', icon: Flame },
  { id: 'wellness', label: 'Wellness', icon: Heart },
  { id: 'agriculture', label: 'Agriculture', icon: Sprout },
  { id: 'social-impact', label: 'Social Impact', icon: HandHeart },
  { id: 'opportunities', label: 'Opportunities', icon: Sparkles },
] as const;

export const EnhancedFeed = memo(({ onNavigate }: EnhancedFeedProps) => {
  const [activeTab, setActiveTab] = useState<FeedTab>('all');
  const [dailyTip, setDailyTip] = useState<WellnessTip | null>(null);
  const [initiatives, setInitiatives] = useState<SocialInitiative[]>([]);
  const [opportunities, setOpportunities] = useState<Opportunity[]>([]);
  const [votedInitiatives, setVotedInitiatives] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    loadContent();
  }, []);

  const loadContent = useCallback(async (): Promise<void> => {
    try {
      setIsLoading(true);
      const [tip, socialInitiatives, curatedOpps] = await Promise.all([
        wellnessContentService.getDailyTip(),
        socialImpactService.getCuratedInitiatives(3),
        opportunityCurationService.getCuratedOpportunities(undefined, 4)
      ]);
      
      setDailyTip(tip);
      setInitiatives(socialInitiatives);
      setOpportunities(curatedOpps);
    } catch (error) {
      console.error('Failed to load feed content:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleVote = useCallback(async (initiativeId: string): Promise<void> => {
    const success = await socialImpactService.voteForInitiative('user-1', initiativeId);
    if (success) {
      setVotedInitiatives(prev => new Set([...prev, initiativeId]));
      setInitiatives(prev => 
        prev.map(init => 
          init.id === initiativeId 
            ? { ...init, votes: init.votes + 1 }
            : init
        )
      );
    }
  }, []);

  return (
    <div className="relative min-h-screen pb-32">
      {/* Header */}
      <div className="p-4 sm:p-6 pt-6 sm:pt-8">
        <div className="flex items-start justify-between mb-6">
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-black dark:bg-white flex items-center justify-center text-white dark:text-black font-semibold text-base sm:text-lg shadow-md">
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
          {FEED_TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveTab(id)}
              className={activeTab === id ? 'professional-tab-active' : 'professional-tab-inactive'}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>

        {/* Content Sections */}
        <div className="space-y-4">
          {/* Content Verification Feature */}
          {activeTab === 'all' && (
            <div className="card-elevated p-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-12 h-12 rounded-lg bg-black dark:bg-white flex items-center justify-center shadow-md">
                  <Shield className="w-6 h-6 text-white dark:text-black" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-lg text-gray-900">Content Verification</h3>
                  <p className="text-xs text-gray-500">AI-Powered Analysis</p>
                </div>
              </div>
              <p className="text-gray-600 text-sm mb-5 leading-relaxed">
                Verify content authenticity with advanced deepfake detection technology
              </p>
              <button className="w-full professional-button-primary">
                Upload Content to Verify
              </button>
              <div className="mt-5 pt-5 border-t border-gray-200/50 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <CheckCircle className="w-5 h-5 text-green-500" />
                  <span className="text-sm font-medium text-gray-700">94% Accuracy</span>
                </div>
                <div className="flex items-center gap-2">
                  <Shield className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Secure Analysis</span>
                </div>
              </div>
            </div>
          )}

          {/* Daily Wellness Tip */}
          {(activeTab === 'all' || activeTab === 'wellness') && dailyTip && (
            <div className="card-elevated p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-red-50 dark:bg-red-950 flex items-center justify-center">
                    <Heart className="w-5 h-5 text-red-600 dark:text-red-400" />
                  </div>
                  <h3 className="font-semibold text-lg text-gray-900 dark:text-gray-100">Daily Wellness Tip</h3>
                </div>
                <span className="badge-secondary">
                  {dailyTip.duration}
                </span>
              </div>
              <h4 className="font-semibold text-gray-900 dark:text-gray-100 mb-2 text-base">{dailyTip.title}</h4>
              <p className="text-gray-600 dark:text-gray-300 text-sm mb-5 leading-relaxed">{dailyTip.content}</p>
              <div className="flex items-center justify-between pt-4 border-t border-gray-200 dark:border-gray-700">
                <div className="flex gap-2">
                  {dailyTip.tags.slice(0, 2).map(tag => (
                    <span key={tag} className="professional-badge bg-white/70 text-gray-600">
                      #{tag}
                    </span>
                  ))}
                </div>
                <button className="text-gray-900 dark:text-gray-100 text-sm font-semibold flex items-center gap-1 hover:text-gray-700 dark:hover:text-gray-300 transition-colors">
                  Read More <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* Agriculture Tools */}
          {(activeTab === 'all' || activeTab === 'agriculture') && (
            <div className="card-elevated p-6">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-12 h-12 rounded-xl bg-green-500/10 flex items-center justify-center">
                  <Sprout className="w-6 h-6 text-green-600" />
                </div>
                <div className="flex-1">
                  <h3 className="font-semibold text-lg text-gray-900">Smart Farming Tools</h3>
                  <p className="text-xs text-gray-500">Precision Agriculture</p>
                </div>
              </div>
              <p className="text-gray-600 text-sm mb-5 leading-relaxed">
                AI-powered insights for optimized crop management
              </p>
              <div className="grid grid-cols-2 gap-3 mb-5">
                <div className="stat-card">
                  <Droplets className="w-5 h-5 text-blue-600 mb-2" />
                  <p className="text-sm font-semibold text-gray-900">Soil Analysis</p>
                  <p className="text-xs text-gray-600 mt-1">pH 6.5, Medium N</p>
                </div>
                <div className="stat-card">
                  <ThermometerSun className="w-5 h-5 text-orange-600 mb-2" />
                  <p className="text-sm font-semibold text-gray-900">Weather</p>
                  <p className="text-xs text-gray-600 mt-1">Rain in 3 days</p>
                </div>
              </div>
              <button className="w-full professional-button bg-green-600 text-white hover:bg-green-700">
                Get Crop Recommendations
              </button>
            </div>
          )}

          {/* Social Impact Initiatives */}
          {(activeTab === 'all' || activeTab === 'social-impact') && (
            <div className="glass-card p-6 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <HandHeart className="w-6 h-6 text-gray-900 dark:text-gray-100" />
                  <h3 className="font-semibold text-lg">Social Impact</h3>
                </div>
                <span className="badge-accent">
                  AI Curated
                </span>
              </div>
              <div className="space-y-3">
                {initiatives.map((initiative) => (
                  <div key={initiative.id} className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
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
                            : 'bg-black dark:bg-white text-white dark:text-black hover:bg-gray-800 dark:hover:bg-gray-200'
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
                      <span className="text-xs font-bold accent-blue ml-2">
                        {opp.relevanceScore.toFixed(1)}
                      </span>
                    </div>
                    <p className="text-xs text-gray-600 mb-2 line-clamp-2">{opp.description}</p>
                    {opp.aiInsights && opp.aiInsights.length > 0 && (
                      <p className="text-xs accent-blue italic">💡 {opp.aiInsights[0]}</p>
                    )}
                  </div>
                ))}
              </div>
              <button 
                onClick={() => onNavigate('explore')}
                className="w-full mt-4 text-gray-900 dark:text-gray-100 font-medium text-sm flex items-center justify-center gap-2 hover:text-gray-700 dark:hover:text-gray-300 transition-all"
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
});

EnhancedFeed.displayName = 'EnhancedFeed';
