
import React, { useState, useEffect } from 'react';
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
  Activity,
  Users,
  Bot
} from 'lucide-react';

interface ActivityItem {
  id: string;
  type: 'ai_insight' | 'system' | 'community' | 'cross_domain' | 'optimization';
  title: string;
  description: string;
  significance: number;
  timestamp: string;
  domain: string;
  connections?: string[];
  actionable?: boolean;
}

interface EnhancedActivityFeedProps {
  activities?: ActivityItem[];
  showFilters?: boolean;
}

export const EnhancedActivityFeed: React.FC<EnhancedActivityFeedProps> = ({ 
  activities: propActivities,
  showFilters = true 
}) => {
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [filter, setFilter] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (propActivities) {
      setActivities(propActivities);
    } else {
      generateIntelligentActivities();
    }
  }, [propActivities]);

  const generateIntelligentActivities = () => {
    setIsLoading(true);
    
    const intelligentActivities: ActivityItem[] = [
      {
        id: '1',
        type: 'ai_insight',
        title: 'Cross-Domain Intelligence Synthesis',
        description: 'AI detected breakthrough connections between quantum computing research and entertainment content optimization, leading to 340% improvement in personalized content delivery across all platform domains.',
        significance: 9.4,
        timestamp: '2m ago',
        domain: 'Global Intelligence',
        connections: ['news', 'entertainment', 'analytics'],
        actionable: true
      },
      {
        id: '2',
        type: 'system',
        title: 'Platform Neural Network Evolution',
        description: 'System autonomously optimized significance algorithm weights based on 50M+ user interactions, achieving 98.7% accuracy in content relevance prediction across social, commerce, and workspace domains.',
        significance: 9.1,
        timestamp: '8m ago',
        domain: 'Platform Intelligence',
        connections: ['social', 'commerce', 'workspace'],
        actionable: true
      },
      {
        id: '3',
        type: 'cross_domain',
        title: 'Unified Workflow Intelligence',
        description: 'Cross-domain analysis revealed optimal productivity patterns: news consumption at 9AM increases creative output by 67%, while entertainment breaks boost analytical performance by 45%.',
        significance: 8.8,
        timestamp: '15m ago',
        domain: 'Behavioral Intelligence',
        connections: ['news', 'entertainment', 'workspace'],
        actionable: true
      },
      {
        id: '4',
        type: 'community',
        title: 'Global Intelligence Network Expansion',
        description: 'Connected with 23,000+ verified intelligence sources worldwide. Real-time collaboration protocols now enable instant knowledge synthesis across time zones and domains.',
        significance: 8.6,
        timestamp: '1h ago',
        domain: 'Network Intelligence',
        connections: ['community', 'analytics', 'automation'],
        actionable: false
      },
      {
        id: '5',
        type: 'optimization',
        title: 'Predictive Content Pipeline',
        description: 'AI successfully predicted viral content trends 72 hours in advance with 94% accuracy, automatically optimizing content distribution across social and entertainment platforms.',
        significance: 8.9,
        timestamp: '2h ago',
        domain: 'Predictive Intelligence',
        connections: ['social', 'entertainment', 'analytics'],
        actionable: true
      }
    ];

    setTimeout(() => {
      setActivities(intelligentActivities);
      setIsLoading(false);
    }, 1000);
  };

  const getTypeIcon = (type: string) => {
    const icons = {
      ai_insight: Brain,
      system: Zap,
      community: Users,
      cross_domain: Globe,
      optimization: Target
    };
    return icons[type as keyof typeof icons] || Activity;
  };

  const getTypeColor = (type: string) => {
    const colors = {
      ai_insight: 'from-purple-500/20 to-pink-500/20 border-purple-500/30',
      system: 'from-cyan-500/20 to-blue-500/20 border-cyan-500/30',
      community: 'from-green-500/20 to-emerald-500/20 border-green-500/30',
      cross_domain: 'from-orange-500/20 to-red-500/20 border-orange-500/30',
      optimization: 'from-violet-500/20 to-purple-500/20 border-violet-500/30'
    };
    return colors[type as keyof typeof colors] || 'from-slate-500/20 to-gray-500/20 border-slate-500/30';
  };

  const filteredActivities = filter === 'all' 
    ? activities 
    : activities.filter(activity => activity.type === filter);

  return (
    <div className="space-y-6">
      {showFilters && (
        <div className="flex items-center justify-between">
          <h3 className="text-xl font-bold text-cyan-300">Intelligence Activity Stream</h3>
          <div className="flex items-center space-x-2">
            {['all', 'ai_insight', 'cross_domain', 'optimization'].map((filterType) => (
              <Button
                key={filterType}
                variant={filter === filterType ? 'default' : 'outline'}
                size="sm"
                onClick={() => setFilter(filterType)}
                className={filter === filterType 
                  ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white'
                  : 'border-slate-600 text-slate-400 hover:text-white hover:border-cyan-500/50'
                }
              >
                {filterType.replace('_', ' ').toUpperCase()}
              </Button>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-4">
        {isLoading ? (
          Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} className="bg-slate-800/50 border-slate-700/50 animate-pulse">
              <CardContent className="p-6">
                <div className="space-y-3">
                  <div className="h-4 bg-slate-700 rounded w-3/4"></div>
                  <div className="h-3 bg-slate-700 rounded w-full"></div>
                  <div className="h-3 bg-slate-700 rounded w-2/3"></div>
                </div>
              </CardContent>
            </Card>
          ))
        ) : (
          filteredActivities.map((activity) => {
            const IconComponent = getTypeIcon(activity.type);
            return (
              <Card 
                key={activity.id} 
                className={`bg-gradient-to-br ${getTypeColor(activity.type)} hover:border-opacity-60 transition-all duration-300 group`}
              >
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-lg flex items-center justify-center border border-cyan-500/30">
                        <IconComponent className="w-5 h-5 text-cyan-400" />
                      </div>
                      <div>
                        <CardTitle className="text-white text-lg group-hover:text-cyan-300 transition-colors">
                          {activity.title}
                        </CardTitle>
                        <div className="flex items-center space-x-2 mt-1">
                          <Badge variant="secondary" className="bg-purple-500/20 text-purple-300 border-purple-500/30">
                            {activity.significance.toFixed(1)}
                          </Badge>
                          <span className="text-slate-400 text-sm">{activity.domain}</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-slate-400 text-sm">{activity.timestamp}</span>
                      {activity.actionable && (
                        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                      )}
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-slate-300 mb-4 leading-relaxed">
                    {activity.description}
                  </p>
                  
                  {activity.connections && activity.connections.length > 0 && (
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Globe className="w-4 h-4 text-cyan-400" />
                        <span className="text-sm text-cyan-400">
                          Connected: {activity.connections.join(', ')}
                        </span>
                      </div>
                      {activity.actionable && (
                        <Button 
                          variant="outline" 
                          size="sm"
                          className="border-cyan-500/30 text-cyan-300 hover:bg-cyan-500/10"
                        >
                          <ArrowRight className="w-3 h-3 mr-1" />
                          Explore
                        </Button>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })
        )}
      </div>

      <div className="text-center pt-4">
        <Button 
          variant="outline" 
          onClick={generateIntelligentActivities}
          disabled={isLoading}
          className="border-purple-500/30 text-purple-300 hover:bg-purple-500/10"
        >
          {isLoading ? (
            <>
              <Bot className="w-4 h-4 mr-2 animate-spin" />
              Generating Intelligence...
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4 mr-2" />
              Generate New Insights
            </>
          )}
        </Button>
      </div>
    </div>
  );
};
