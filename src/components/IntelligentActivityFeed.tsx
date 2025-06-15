
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
  Bot,
  Clock
} from 'lucide-react';

interface ActivityItem {
  id: string;
  type: 'ai_insight' | 'system' | 'community' | 'cross_domain' | 'optimization' | 'learning';
  title: string;
  description: string;
  significance: number;
  timestamp: string;
  domain: string;
  connections?: string[];
  actionable?: boolean;
  priority?: 'high' | 'medium' | 'low';
}

export const IntelligentActivityFeed: React.FC = () => {
  const [activities, setActivities] = useState<ActivityItem[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [filter, setFilter] = useState<string>('all');

  useEffect(() => {
    generateIntelligentActivities();
    
    // Real-time activity simulation
    const interval = setInterval(() => {
      if (Math.random() > 0.7) {
        addRandomActivity();
      }
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  const generateIntelligentActivities = () => {
    setIsGenerating(true);
    
    const intelligentActivities: ActivityItem[] = [
      {
        id: '1',
        type: 'ai_insight',
        title: 'Cross-Domain Pattern Recognition Breakthrough',
        description: 'AI detected unprecedented correlations between user productivity cycles, entertainment preferences, and social engagement patterns. This insight enables 340% improvement in personalized content timing across all platform domains.',
        significance: 9.6,
        timestamp: '2m ago',
        domain: 'Global Intelligence',
        connections: ['workspace', 'entertainment', 'social', 'analytics'],
        actionable: true,
        priority: 'high'
      },
      {
        id: '2',
        type: 'system',
        title: 'Neural Network Evolution Complete',
        description: 'Platform AI autonomously upgraded its significance algorithm using quantum-inspired optimization. New architecture processes 500M+ data points with 99.2% accuracy while reducing computational overhead by 67%.',
        significance: 9.4,
        timestamp: '8m ago',
        domain: 'Platform Intelligence',
        connections: ['analytics', 'automation'],
        actionable: true,
        priority: 'high'
      },
      {
        id: '3',
        type: 'cross_domain',
        title: 'Unified Workflow Intelligence Synthesis',
        description: 'Cross-domain analysis revealed optimal productivity patterns: morning news consumption increases creative output by 67%, strategic entertainment breaks boost analytical performance by 45%, and social interactions enhance problem-solving by 38%.',
        significance: 9.1,
        timestamp: '15m ago',
        domain: 'Behavioral Intelligence',
        connections: ['news', 'entertainment', 'workspace', 'social'],
        actionable: true,
        priority: 'medium'
      },
      {
        id: '4',
        type: 'learning',
        title: 'Adaptive Learning Network Expansion',
        description: 'Connected with 47,000+ verified intelligence sources globally. Real-time learning protocols now enable instant knowledge synthesis across 127 domains and 34 languages with 96.8% accuracy.',
        significance: 8.9,
        timestamp: '1h ago',
        domain: 'Learning Intelligence',
        connections: ['community', 'analytics', 'automation'],
        actionable: false,
        priority: 'medium'
      },
      {
        id: '5',
        type: 'optimization',
        title: 'Predictive Content Intelligence Pipeline',
        description: 'AI successfully predicted viral content trends 96 hours in advance with 97% accuracy. Automated content optimization now runs across social, entertainment, and news platforms with real-time adaptation.',
        significance: 9.2,
        timestamp: '2h ago',
        domain: 'Predictive Intelligence',
        connections: ['social', 'entertainment', 'news'],
        actionable: true,
        priority: 'high'
      }
    ];

    setTimeout(() => {
      setActivities(intelligentActivities);
      setIsGenerating(false);
    }, 1200);
  };

  const addRandomActivity = () => {
    const newActivity: ActivityItem = {
      id: Date.now().toString(),
      type: Math.random() > 0.5 ? 'ai_insight' : 'cross_domain',
      title: 'Real-time Intelligence Discovery',
      description: `AI detected new pattern correlation with ${(Math.random() * 40 + 60).toFixed(1)}% confidence. Cross-domain impact across ${Math.floor(Math.random() * 4 + 2)} platform areas.`,
      significance: Math.random() * 2 + 7.5,
      timestamp: 'now',
      domain: 'Live Intelligence',
      connections: ['analytics'],
      actionable: Math.random() > 0.3,
      priority: 'medium'
    };

    setActivities(prev => [newActivity, ...prev.slice(0, 9)]);
  };

  const getTypeIcon = (type: string) => {
    const icons = {
      ai_insight: Brain,
      system: Zap,
      community: Users,
      cross_domain: Globe,
      optimization: Target,
      learning: Bot
    };
    return icons[type as keyof typeof icons] || Activity;
  };

  const getTypeColor = (type: string) => {
    const colors = {
      ai_insight: 'from-purple-500/20 to-pink-500/20 border-purple-500/30',
      system: 'from-cyan-500/20 to-blue-500/20 border-cyan-500/30',
      community: 'from-green-500/20 to-emerald-500/20 border-green-500/30',
      cross_domain: 'from-orange-500/20 to-red-500/20 border-orange-500/30',
      optimization: 'from-violet-500/20 to-purple-500/20 border-violet-500/30',
      learning: 'from-yellow-500/20 to-orange-500/20 border-yellow-500/30'
    };
    return colors[type as keyof typeof colors] || 'from-slate-500/20 to-gray-500/20 border-slate-500/30';
  };

  const getPriorityColor = (priority?: string) => {
    switch (priority) {
      case 'high': return 'bg-red-500/20 text-red-300 border-red-500/30';
      case 'medium': return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30';
      case 'low': return 'bg-green-500/20 text-green-300 border-green-500/30';
      default: return 'bg-gray-500/20 text-gray-300 border-gray-500/30';
    }
  };

  const filteredActivities = filter === 'all' 
    ? activities 
    : activities.filter(activity => activity.type === filter);

  return (
    <Card className="bg-gradient-to-br from-slate-800/50 to-purple-900/20 border-purple-500/30">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-purple-300 flex items-center">
            <Brain className="w-5 h-5 mr-2" />
            Intelligent Activity Stream
          </CardTitle>
          <div className="flex items-center space-x-2">
            {['all', 'ai_insight', 'cross_domain', 'optimization'].map((filterType) => (
              <Button
                key={filterType}
                variant={filter === filterType ? 'default' : 'outline'}
                size="sm"
                onClick={() => setFilter(filterType)}
                className={filter === filterType 
                  ? 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white h-7 text-xs'
                  : 'border-slate-600 text-slate-400 hover:text-white hover:border-cyan-500/50 h-7 text-xs'
                }
              >
                {filterType.replace('_', ' ').toUpperCase()}
              </Button>
            ))}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {isGenerating ? (
          Array.from({ length: 3 }).map((_, i) => (
            <div key={i} className="bg-slate-700/30 rounded-lg p-4 animate-pulse">
              <div className="space-y-3">
                <div className="h-4 bg-slate-600 rounded w-3/4"></div>
                <div className="h-3 bg-slate-600 rounded w-full"></div>
                <div className="h-3 bg-slate-600 rounded w-2/3"></div>
              </div>
            </div>
          ))
        ) : (
          filteredActivities.map((activity) => {
            const IconComponent = getTypeIcon(activity.type);
            return (
              <div 
                key={activity.id} 
                className={`bg-gradient-to-br ${getTypeColor(activity.type)} rounded-lg p-4 hover:border-opacity-60 transition-all duration-300 group`}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-start space-x-3">
                    <div className="w-10 h-10 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-lg flex items-center justify-center border border-cyan-500/30 flex-shrink-0 mt-0.5">
                      <IconComponent className="w-5 h-5 text-cyan-400" />
                    </div>
                    <div className="flex-1">
                      <h3 className="text-white font-semibold text-lg group-hover:text-cyan-300 transition-colors">
                        {activity.title}
                      </h3>
                      <div className="flex items-center space-x-2 mt-1">
                        <Badge variant="secondary" className="bg-purple-500/20 text-purple-300 border-purple-500/30 text-xs">
                          {activity.significance.toFixed(1)}
                        </Badge>
                        {activity.priority && (
                          <Badge variant="outline" className={`${getPriorityColor(activity.priority)} text-xs`}>
                            {activity.priority.toUpperCase()}
                          </Badge>
                        )}
                        <span className="text-slate-400 text-sm">{activity.domain}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2 flex-shrink-0">
                    <div className="flex items-center space-x-1 text-slate-400 text-sm">
                      <Clock className="w-3 h-3" />
                      <span>{activity.timestamp}</span>
                    </div>
                    {activity.actionable && (
                      <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    )}
                  </div>
                </div>
                
                <p className="text-slate-300 mb-4 leading-relaxed text-sm">
                  {activity.description}
                </p>
                
                {activity.connections && activity.connections.length > 0 && (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <Globe className="w-4 h-4 text-cyan-400" />
                      <span className="text-sm text-cyan-400">
                        Connected: {activity.connections.slice(0, 3).join(', ')}
                        {activity.connections.length > 3 && ` +${activity.connections.length - 3} more`}
                      </span>
                    </div>
                    {activity.actionable && (
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
                )}
              </div>
            );
          })
        )}

        <div className="text-center pt-4 border-t border-slate-600/30">
          <Button 
            variant="outline" 
            onClick={generateIntelligentActivities}
            disabled={isGenerating}
            className="border-purple-500/30 text-purple-300 hover:bg-purple-500/10"
          >
            {isGenerating ? (
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
      </CardContent>
    </Card>
  );
};
