
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { 
  Brain, 
  Zap, 
  Globe, 
  TrendingUp, 
  Target, 
  Sparkles,
  ArrowRight,
  Clock,
  Bot
} from 'lucide-react';
import { useIntelligenceEngine } from '../hooks/useIntelligenceEngine';
import { PLATFORM_CONFIG } from '../lib/platformConfig';

export const OptimizedActivityFeed: React.FC = () => {
  const { insights, isGenerating, generateInsights } = useIntelligenceEngine();

  const getTypeIcon = (type: string) => {
    const icons = {
      breakthrough: Brain,
      optimization: Zap,
      prediction: TrendingUp,
      correlation: Globe,
      anomaly: Target
    };
    return icons[type as keyof typeof icons] || Brain;
  };

  const getTypeGradient = (type: string) => {
    const gradients = {
      breakthrough: PLATFORM_CONFIG.ui.gradients.intelligence,
      optimization: PLATFORM_CONFIG.ui.gradients.system,
      prediction: PLATFORM_CONFIG.ui.gradients.analytics,
      correlation: PLATFORM_CONFIG.ui.gradients.warning,
      anomaly: PLATFORM_CONFIG.ui.gradients.primary
    };
    return gradients[type as keyof typeof gradients] || PLATFORM_CONFIG.ui.gradients.primary;
  };

  const getPriorityColor = (significance: number) => {
    if (significance >= 9) return 'bg-red-500/20 text-red-300 border-red-500/30';
    if (significance >= 8) return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
    return 'bg-green-500/20 text-green-300 border-green-500/30';
  };

  return (
    <Card className="bg-gradient-to-br from-slate-800/50 to-purple-900/20 border-purple-500/30 hover:border-purple-400/50 transition-all duration-300">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-purple-300 flex items-center">
            <Brain className="w-5 h-5 mr-2 animate-pulse" />
            Intelligent Activity Stream
            <Badge className="ml-3 bg-cyan-500/20 text-cyan-300 border-cyan-500/30">
              Real-time
            </Badge>
          </CardTitle>
          <Button
            onClick={generateInsights}
            disabled={isGenerating}
            variant="outline"
            size="sm"
            className="border-purple-500/30 text-purple-300 hover:bg-purple-500/10"
          >
            {isGenerating ? (
              <>
                <Bot className="w-4 h-4 mr-2 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4 mr-2" />
                Generate Insights
              </>
            )}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4 max-h-96 overflow-y-auto scrollbar-thin scrollbar-track-slate-800 scrollbar-thumb-purple-500/30">
        {isGenerating && (
          <div className="flex items-center justify-center py-8">
            <div className="text-center">
              <div className="w-12 h-12 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-xl flex items-center justify-center mx-auto mb-3 animate-pulse">
                <Bot className="w-6 h-6 text-white animate-spin" />
              </div>
              <div className="text-sm text-purple-300">AI is generating new insights...</div>
            </div>
          </div>
        )}
        
        {insights.map((insight) => {
          const IconComponent = getTypeIcon(insight.type);
          const gradient = getTypeGradient(insight.type);
          
          return (
            <div 
              key={insight.id} 
              className="bg-gradient-to-br from-slate-700/30 to-slate-800/30 rounded-lg p-4 border border-slate-600/30 hover:border-cyan-500/30 transition-all duration-300 group animate-fade-in"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-start space-x-3">
                  <div className={`w-10 h-10 bg-gradient-to-r ${gradient} bg-opacity-20 rounded-lg flex items-center justify-center border border-white/20 flex-shrink-0 mt-0.5 group-hover:scale-110 transition-transform duration-300`}>
                    <IconComponent className="w-5 h-5 text-white" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-white font-semibold text-base group-hover:text-cyan-300 transition-colors line-clamp-1">
                      {insight.title}
                    </h3>
                    <div className="flex items-center space-x-2 mt-1">
                      <Badge 
                        variant="outline" 
                        className="bg-purple-500/20 text-purple-300 border-purple-500/30 text-xs"
                      >
                        {insight.significance.toFixed(1)}
                      </Badge>
                      <Badge 
                        variant="outline" 
                        className={`${getPriorityColor(insight.significance)} text-xs`}
                      >
                        {insight.significance >= 9 ? 'CRITICAL' : insight.significance >= 8 ? 'HIGH' : 'MEDIUM'}
                      </Badge>
                      <Badge 
                        variant="outline" 
                        className="bg-cyan-500/20 text-cyan-300 border-cyan-500/30 text-xs"
                      >
                        {insight.confidence.toFixed(1)}% confidence
                      </Badge>
                    </div>
                  </div>
                </div>
                <div className="flex items-center space-x-2 flex-shrink-0">
                  <div className="flex items-center space-x-1 text-slate-400 text-sm">
                    <Clock className="w-3 h-3" />
                    <span>{insight.timestamp}</span>
                  </div>
                  {insight.actionable && (
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  )}
                </div>
              </div>
              
              <p className="text-slate-300 mb-4 leading-relaxed text-sm line-clamp-2">
                {insight.description}
              </p>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Globe className="w-4 h-4 text-cyan-400" />
                  <span className="text-sm text-cyan-400">
                    Domains: {insight.domains.slice(0, 2).join(', ')}
                    {insight.domains.length > 2 && ` +${insight.domains.length - 2}`}
                  </span>
                </div>
                {insight.actionable && (
                  <Button 
                    variant="outline" 
                    size="sm"
                    className="border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10 h-7 text-xs"
                  >
                    <ArrowRight className="w-3 h-3 mr-1" />
                    Explore
                  </Button>
                )}
              </div>
            </div>
          );
        })}
        
        {insights.length === 0 && !isGenerating && (
          <div className="text-center py-8">
            <Brain className="w-12 h-12 text-slate-400 mx-auto mb-3" />
            <div className="text-slate-400">No insights available. Generate some to get started!</div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
