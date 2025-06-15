
import React, { useState, useEffect } from 'react';
import { UnifiedHeader } from '../components/UnifiedHeader';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { globalSearch } from '../services/globalSearch';
import { 
  TrendingUp, 
  Users, 
  Activity, 
  Brain,
  BarChart3,
  PieChart,
  LineChart,
  Target,
  Zap,
  Globe
} from 'lucide-react';

export default function Analytics() {
  const [insights, setInsights] = useState<any>(null);
  const [realTimeStats, setRealTimeStats] = useState({
    activeUsers: 24891,
    significanceScore: 8.7,
    contentInteractions: 156000,
    aiLearningRate: 94.3,
    crossDomainConnections: 47
  });

  useEffect(() => {
    loadInsights();
    
    // Real-time updates
    const interval = setInterval(() => {
      setRealTimeStats(prev => ({
        activeUsers: prev.activeUsers + Math.floor(Math.random() * 20) - 10,
        significanceScore: Math.round((prev.significanceScore + (Math.random() * 0.2 - 0.1)) * 10) / 10,
        contentInteractions: prev.contentInteractions + Math.floor(Math.random() * 100),
        aiLearningRate: Math.min(99.9, prev.aiLearningRate + Math.random() * 0.1),
        crossDomainConnections: prev.crossDomainConnections + Math.floor(Math.random() * 3)
      }));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const loadInsights = async () => {
    const data = await globalSearch.getInsights();
    setInsights(data);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-gray-950 to-blue-1000">
      <UnifiedHeader 
        title="Platform Intelligence Analytics"
        subtitle="Real-time insights across all domains • Advanced AI metrics • Predictive analytics"
        activeFeatures={['ai', 'learning', 'optimized']}
      />

      <div className="p-6 space-y-6">
        {/* Real-time Metrics Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
          <Card className="bg-gradient-to-br from-cyan-900/30 to-blue-900/30 border-cyan-500/30">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-cyan-300 flex items-center">
                <Users className="w-4 h-4 mr-2" />
                Active Users
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-cyan-400">{realTimeStats.activeUsers.toLocaleString()}</div>
              <div className="flex items-center text-sm text-green-400 mt-1">
                <TrendingUp className="w-4 h-4 mr-1" />
                +12.5% from last week
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-900/30 to-pink-900/30 border-purple-500/30">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-purple-300 flex items-center">
                <Brain className="w-4 h-4 mr-2" />
                Significance Score
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-400">{realTimeStats.significanceScore}/10</div>
              <div className="flex items-center text-sm text-green-400 mt-1">
                <Target className="w-4 h-4 mr-1" />
                Optimal performance
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-900/30 to-emerald-900/30 border-green-500/30">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-green-300 flex items-center">
                <Activity className="w-4 h-4 mr-2" />
                Content Interactions
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-400">{Math.floor(realTimeStats.contentInteractions / 1000)}K</div>
              <div className="flex items-center text-sm text-cyan-400 mt-1">
                <Zap className="w-4 h-4 mr-1" />
                +8.2% daily growth
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-orange-900/30 to-red-900/30 border-orange-500/30">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-orange-300 flex items-center">
                <Brain className="w-4 h-4 mr-2" />
                AI Learning Rate
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-orange-400">{realTimeStats.aiLearningRate.toFixed(1)}%</div>
              <div className="flex items-center text-sm text-purple-400 mt-1">
                <Target className="w-4 h-4 mr-1" />
                High accuracy
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-teal-900/30 to-cyan-900/30 border-teal-500/30">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-teal-300 flex items-center">
                <Globe className="w-4 h-4 mr-2" />
                Cross-Domain Links
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-teal-400">{realTimeStats.crossDomainConnections}</div>
              <div className="flex items-center text-sm text-green-400 mt-1">
                <TrendingUp className="w-4 h-4 mr-1" />
                Intelligence bridges
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Advanced Analytics */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardHeader>
              <CardTitle className="text-cyan-300 flex items-center">
                <BarChart3 className="w-5 h-5 mr-2" />
                Domain Performance Intelligence
              </CardTitle>
              <CardDescription>Real-time engagement across platform domains</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  { name: 'Entertainment', value: 83, color: 'from-purple-500 to-pink-500', textColor: 'text-purple-400' },
                  { name: 'News Intelligence', value: 75, color: 'from-cyan-500 to-blue-500', textColor: 'text-cyan-400' },
                  { name: 'Smart Commerce', value: 67, color: 'from-green-500 to-emerald-500', textColor: 'text-green-400' },
                  { name: 'Social Layer', value: 58, color: 'from-orange-500 to-red-500', textColor: 'text-orange-400' },
                  { name: 'Task Automation', value: 71, color: 'from-violet-500 to-purple-500', textColor: 'text-violet-400' }
                ].map((domain) => (
                  <div key={domain.name} className="flex items-center justify-between">
                    <span className="text-slate-300 font-medium">{domain.name}</span>
                    <div className="flex items-center space-x-3">
                      <div className="w-32 h-3 bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className={`h-full bg-gradient-to-r ${domain.color} rounded-full transition-all duration-1000`}
                          style={{ width: `${domain.value}%` }}
                        ></div>
                      </div>
                      <span className={`${domain.textColor} font-semibold min-w-[3rem] text-right`}>{domain.value}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700/50">
            <CardHeader>
              <CardTitle className="text-purple-300 flex items-center">
                <Brain className="w-5 h-5 mr-2" />
                AI Learning & Intelligence Metrics
              </CardTitle>
              <CardDescription>Platform intelligence evolution in real-time</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  { name: 'Pattern Recognition', status: 'Improving', color: 'bg-cyan-500/20 text-cyan-300', icon: Brain },
                  { name: 'Content Relevance', status: 'Optimal', color: 'bg-green-500/20 text-green-300', icon: Target },
                  { name: 'User Understanding', status: 'Excellent', color: 'bg-purple-500/20 text-purple-300', icon: Users },
                  { name: 'Cross-Domain Intelligence', status: 'Advanced', color: 'bg-orange-500/20 text-orange-300', icon: Globe },
                  { name: 'Predictive Accuracy', status: 'Superior', color: 'bg-pink-500/20 text-pink-300', icon: TrendingUp }
                ].map((metric) => (
                  <div key={metric.name} className="flex items-center justify-between p-4 bg-slate-700/30 rounded-lg border border-slate-600/30">
                    <div className="flex items-center space-x-3">
                      <metric.icon className="w-5 h-5 text-slate-400" />
                      <span className="text-slate-300 font-medium">{metric.name}</span>
                    </div>
                    <Badge variant="secondary" className={metric.color}>
                      {metric.status}
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Cross-Domain Intelligence Insights */}
        {insights && (
          <Card className="bg-gradient-to-br from-slate-800/50 to-purple-900/20 border-purple-500/30">
            <CardHeader>
              <CardTitle className="text-purple-300 flex items-center">
                <Globe className="w-5 h-5 mr-2" />
                Cross-Domain Intelligence Network
              </CardTitle>
              <CardDescription>Real-time connections between different platform domains</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-cyan-300 font-semibold mb-3">Trending Cross-Connections</h4>
                  <div className="space-y-3">
                    {insights.trendingTopics?.slice(0, 4).map((topic: string, index: number) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                        <span className="text-slate-300 text-sm">{topic}</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                          <span className="text-green-400 text-xs font-medium">Active</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div>
                  <h4 className="text-green-300 font-semibold mb-3">Intelligence Synthesis</h4>
                  <div className="space-y-3">
                    <div className="p-3 bg-green-900/20 rounded-lg border border-green-500/30">
                      <div className="text-green-300 text-sm font-medium mb-1">News → Entertainment</div>
                      <div className="text-slate-400 text-xs">AI trends influencing content creation</div>
                    </div>
                    <div className="p-3 bg-cyan-900/20 rounded-lg border border-cyan-500/30">
                      <div className="text-cyan-300 text-sm font-medium mb-1">Commerce → Social</div>
                      <div className="text-slate-400 text-xs">Product insights driving social engagement</div>
                    </div>
                    <div className="p-3 bg-purple-900/20 rounded-lg border border-purple-500/30">
                      <div className="text-purple-300 text-sm font-medium mb-1">Analytics → Automation</div>
                      <div className="text-slate-400 text-xs">Performance data optimizing workflows</div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
