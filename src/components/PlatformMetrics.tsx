
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { 
  Brain, 
  Globe, 
  Users, 
  Zap, 
  TrendingUp, 
  Target, 
  Activity,
  BarChart3,
  Sparkles
} from 'lucide-react';

interface MetricData {
  label: string;
  value: number;
  change: number;
  unit: string;
  icon: any;
  color: string;
  description: string;
}

export const PlatformMetrics: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricData[]>([
    {
      label: 'Global Users',
      value: 24891,
      change: 12.5,
      unit: '',
      icon: Users,
      color: 'from-cyan-500 to-blue-500',
      description: 'Active users across all domains'
    },
    {
      label: 'Intelligence Score',
      value: 8.9,
      change: 0.3,
      unit: '/10',
      icon: Brain,
      color: 'from-purple-500 to-pink-500',
      description: 'AI learning and adaptation rate'
    },
    {
      label: 'Cross-Domain Links',
      value: 47,
      change: 8.2,
      unit: '',
      icon: Globe,
      color: 'from-green-500 to-emerald-500',
      description: 'Active intelligence connections'
    },
    {
      label: 'Optimization Rate',
      value: 94.3,
      change: 2.1,
      unit: '%',
      icon: Target,
      color: 'from-orange-500 to-red-500',
      description: 'Platform performance optimization'
    },
    {
      label: 'Learning Velocity',
      value: 97.2,
      change: 1.8,
      unit: '%',
      icon: TrendingUp,
      color: 'from-violet-500 to-purple-500',
      description: 'AI knowledge acquisition speed'
    },
    {
      label: 'Neural Efficiency',
      value: 156,
      change: 15.7,
      unit: 'ops/s',
      icon: Zap,
      color: 'from-yellow-500 to-orange-500',
      description: 'Neural network processing rate'
    }
  ]);

  const [systemHealth, setSystemHealth] = useState({
    status: 'optimal',
    uptime: 99.97,
    responseTime: 12,
    throughput: 2847
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => prev.map(metric => ({
        ...metric,
        value: metric.label === 'Intelligence Score' || metric.label === 'Optimization Rate' || metric.label === 'Learning Velocity'
          ? Math.min(metric.label === 'Intelligence Score' ? 10 : 100, 
              metric.value + (Math.random() * 0.2 - 0.1))
          : metric.value + Math.floor(Math.random() * 10 - 5)
      })));

      setSystemHealth(prev => ({
        ...prev,
        responseTime: Math.max(8, prev.responseTime + Math.floor(Math.random() * 6 - 3)),
        throughput: prev.throughput + Math.floor(Math.random() * 100 - 50)
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const formatValue = (value: number, unit: string) => {
    if (unit === '' && value > 1000) {
      return `${(value / 1000).toFixed(1)}K`;
    }
    if (unit === '/10' || unit === '%') {
      return value.toFixed(1);
    }
    if (unit === 'ops/s') {
      return Math.floor(value).toString();
    }
    return Math.floor(value).toLocaleString();
  };

  return (
    <div className="space-y-6">
      {/* Real-time System Health */}
      <Card className="bg-gradient-to-r from-green-900/20 to-emerald-900/20 border-green-500/30">
        <CardHeader>
          <CardTitle className="text-green-300 flex items-center">
            <Activity className="w-5 h-5 mr-2" />
            Platform Intelligence Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400">OPTIMAL</div>
              <div className="text-sm text-green-300">System Status</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-cyan-400">{systemHealth.uptime}%</div>
              <div className="text-sm text-cyan-300">Uptime</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-400">{systemHealth.responseTime}ms</div>
              <div className="text-sm text-purple-300">Response Time</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-400">{systemHealth.throughput.toLocaleString()}</div>
              <div className="text-sm text-orange-300">Req/min</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Enhanced Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metrics.map((metric, index) => {
          const IconComponent = metric.icon;
          return (
            <Card 
              key={metric.label} 
              className="bg-slate-800/50 border-slate-700/50 hover:border-cyan-500/30 transition-all duration-300 group"
            >
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`w-12 h-12 bg-gradient-to-r ${metric.color} bg-opacity-20 rounded-xl flex items-center justify-center border border-white/20`}>
                      <IconComponent className="w-6 h-6 text-white" />
                    </div>
                    <div>
                      <CardTitle className="text-white text-lg group-hover:text-cyan-300 transition-colors">
                        {metric.label}
                      </CardTitle>
                      <p className="text-slate-400 text-sm">{metric.description}</p>
                    </div>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-3xl font-bold text-white">
                      {formatValue(metric.value, metric.unit)}{metric.unit}
                    </div>
                    <div className="flex items-center space-x-2 mt-2">
                      <TrendingUp className="w-4 h-4 text-green-400" />
                      <span className="text-green-400 text-sm font-medium">
                        +{metric.change.toFixed(1)}% this week
                      </span>
                    </div>
                  </div>
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Intelligence Synthesis Overview */}
      <Card className="bg-gradient-to-br from-slate-800/50 to-purple-900/20 border-purple-500/30">
        <CardHeader>
          <CardTitle className="text-purple-300 flex items-center">
            <Sparkles className="w-5 h-5 mr-2" />
            AI Intelligence Synthesis Network
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-full flex items-center justify-center border border-cyan-500/30 mx-auto mb-3">
                <Brain className="w-8 h-8 text-cyan-400" />
              </div>
              <div className="text-2xl font-bold text-cyan-400">15,847</div>
              <div className="text-sm text-slate-400">AI Insights Generated</div>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-full flex items-center justify-center border border-green-500/30 mx-auto mb-3">
                <Globe className="w-8 h-8 text-green-400" />
              </div>
              <div className="text-2xl font-bold text-green-400">432</div>
              <div className="text-sm text-slate-400">Cross-Domain Connections</div>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-full flex items-center justify-center border border-purple-500/30 mx-auto mb-3">
                <BarChart3 className="w-8 h-8 text-purple-400" />
              </div>
              <div className="text-2xl font-bold text-purple-400">96.8%</div>
              <div className="text-sm text-slate-400">Intelligence Accuracy</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
