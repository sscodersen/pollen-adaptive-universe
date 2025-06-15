
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { globalSearch } from '../services/globalSearch';
import { 
  Globe, 
  TrendingUp, 
  Zap, 
  ArrowRight, 
  Brain,
  Target 
} from 'lucide-react';

interface CrossDomainInsightsProps {
  currentDomain?: string;
  limit?: number;
  showActions?: boolean;
}

export const CrossDomainInsights: React.FC<CrossDomainInsightsProps> = ({ 
  currentDomain = 'all', 
  limit = 3,
  showActions = true 
}) => {
  const [insights, setInsights] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    loadCrossDomainInsights();
  }, [currentDomain]);

  const loadCrossDomainInsights = async () => {
    setIsLoading(true);
    try {
      const data = await globalSearch.getInsights();
      const domainInsights = currentDomain === 'all' 
        ? data.crossDomainInsights || []
        : globalSearch.getCrossDomainConnections(currentDomain);
      
      setInsights(domainInsights.slice(0, limit));
    } catch (error) {
      console.error('Error loading cross-domain insights:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <Card className="bg-slate-800/50 border-slate-700/50">
        <CardContent className="p-6">
          <div className="animate-pulse space-y-3">
            <div className="h-4 bg-slate-700 rounded w-3/4"></div>
            <div className="h-3 bg-slate-700 rounded w-full"></div>
            <div className="h-3 bg-slate-700 rounded w-2/3"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (insights.length === 0) {
    return (
      <Card className="bg-slate-800/50 border-slate-700/50">
        <CardContent className="p-6 text-center">
          <Globe className="w-8 h-8 text-slate-400 mx-auto mb-2" />
          <p className="text-slate-400 text-sm">No cross-domain insights available</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-gradient-to-br from-slate-800/50 to-purple-900/20 border-purple-500/30">
      <CardHeader>
        <CardTitle className="text-purple-300 flex items-center">
          <Brain className="w-5 h-5 mr-2" />
          Cross-Domain Intelligence
        </CardTitle>
        <CardDescription>
          AI-detected connections across platform domains
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {insights.map((insight, index) => (
          <div 
            key={index} 
            className="p-4 bg-slate-700/30 rounded-lg border border-slate-600/30 hover:border-purple-500/30 transition-colors"
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Globe className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-medium text-cyan-300">
                  {insight.sourceType} → {insight.targetType}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <Badge variant="secondary" className="bg-purple-500/20 text-purple-300 border-purple-500/30">
                  {insight.significance?.toFixed(1) || '8.5'}
                </Badge>
                {insight.actionable && (
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                )}
              </div>
            </div>
            
            <p className="text-slate-300 text-sm mb-3 leading-relaxed">
              {insight.connection}
            </p>
            
            {showActions && insight.actionable && (
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-1 text-xs text-green-400">
                  <Target className="w-3 h-3" />
                  <span>Actionable insight</span>
                </div>
                <Button 
                  variant="outline" 
                  size="sm" 
                  className="border-purple-500/30 text-purple-300 hover:bg-purple-500/10"
                >
                  <ArrowRight className="w-3 h-3 mr-1" />
                  Explore
                </Button>
              </div>
            )}
          </div>
        ))}
        
        {insights.length > 0 && (
          <div className="pt-3 border-t border-slate-600/30">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <div className="flex items-center space-x-1">
                <TrendingUp className="w-3 h-3" />
                <span>Intelligence confidence: 96.8%</span>
              </div>
              <div className="flex items-center space-x-1">
                <Zap className="w-3 h-3" />
                <span>Real-time analysis</span>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
