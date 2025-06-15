
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
  Sparkles,
  Database,
  Network,
  Cpu
} from 'lucide-react';

interface SystemMetric {
  label: string;
  value: number;
  change: number;
  unit: string;
  icon: any;
  color: string;
  description: string;
  status: 'excellent' | 'good' | 'warning' | 'critical';
}

export const SystemMetricsDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetric[]>([
    {
      label: 'Global Users',
      value: 24891,
      change: 12.5,
      unit: '',
      icon: Users,
      color: 'from-cyan-500 to-blue-500',
      description: 'Active users across all domains',
      status: 'excellent'
    },
    {
      label: 'Intelligence Score',
      value: 8.9,
      change: 0.3,
      unit: '/10',
      icon: Brain,
      color: 'from-purple-500 to-pink-500',
      description: 'AI learning and adaptation rate',
      status: 'excellent'
    },
    {
      label: 'Cross-Domain Links',
      value: 127,
      change: 8.2,
      unit: '',
      icon: Globe,
      color: 'from-green-500 to-emerald-500',
      description: 'Active intelligence connections',
      status: 'good'
    },
    {
      label: 'System Performance',
      value: 96.7,
      change: 2.1,
      unit: '%',
      icon: Cpu,
      color: 'from-orange-500 to-red-500',
      description: 'Platform optimization rate',
      status: 'excellent'
    },
    {
      label: 'Learning Velocity',
      value: 97.2,
      change: 1.8,
      unit: '%',
      icon: TrendingUp,
      color: 'from-violet-500 to-purple-500',
      description: 'AI knowledge acquisition speed',
      status: 'excellent'
    },
    {
      label: 'Neural Efficiency',
      value: 156,
      change: 15.7,
      unit: 'ops/s',
      icon: Zap,
      color: 'from-yellow-500 to-orange-500',
      description: 'Neural network processing rate',
      status: 'good'
    }
  ]);

  const [systemHealth, setSystemHealth] = useState({
    status: 'optimal',
    uptime: 99.97,
    responseTime: 12,
    throughput: 2847,
    dataProcessed: 15.7,
    accuracy: 98.9
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => prev.map(metric => {
        let newValue = metric.value;
        
        if (metric.unit === '/10' || metric.unit === '%') {
          const maxValue = metric.unit === '/10' ? 10 : 100;
          newValue = Math.min(maxValue, Math.max(0, metric.value + (Math.random() * 0.4 - 0.2)));
        } else {
          newValue = metric.value + Math.floor(Math.random() * 20 - 10);
        }
        
        return {
          ...metric,
          value: newValue,
          change: metric.change + (Math.random() * 0.4 - 0.2)
        };
      }));

      setSystemHealth(prev => ({
        ...prev,
        responseTime: Math.max(8, prev.responseTime + Math.floor(Math.random() * 6 - 3)),
        throughput: prev.throughput + Math.floor(Math.random() * 100 - 50),
        dataProcessed: prev.dataProcessed + (Math.random() * 0.2),
        accuracy: Math.min(99.9, Math.max(98, prev.accuracy + (Math.random() * 0.2 - 0.1)))
      }));
    }, 4000);

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

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'text-green-400';
      case 'good': return 'text-cyan-400';
      case 'warning': return 'text-yellow-400';
      case 'critical': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      {/* System Health Overview */}
      <Card className="bg-gradient-to-r from-green-900/20 to-emerald-900/20 border-green-500/30">
        <CardHeader>
          <CardTitle className="text-green-300 flex items-center">
            <Activity className="w-5 h-5 mr-2" />
            Platform Intelligence Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            <div className="text-center">
              <div className="text-xl font-bold text-green-400">OPTIMAL</div>
              <div className="text-xs text-green-300">System Status</div>
            </div>
            <div className="text-center">
              <div className="text-xl font-bold text-cyan-400">{systemHealth.uptime}%</div>
              <div className="text-xs text-cyan-300">Uptime</div>
            </div>
            <div className="text-center">
              <div className="text-xl font-bold text-purple-400">{systemHealth.responseTime}ms</div>
              <div className="text-xs text-purple-300">Response</div>
            </div>
            <div className="text-center">
              <div className="text-xl font-bold text-orange-400">{systemHealth.throughput.toLocaleString()}</div>
              <div className="text-xs text-orange-300">Req/min</div>
            </div>
            <div className="text-center">
              <div className="text-xl font-bold text-blue-400">{systemHealth.dataProcessed.toFixed(1)}TB</div>
              <div className="text-xs text-blue-300">Processed</div>
            </div>
            <div className="text-center">
              <div className="text-xl font-bold text-pink-400">{systemHealth.accuracy.toFixed(1)}%</div>
              <div className="text-xs text-pink-300">Accuracy</div>
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
                    <div className="flex-1">
                      <CardTitle className="text-white text-base group-hover:text-cyan-300 transition-colors">
                        {metric.label}
                      </CardTitle>
                      <p className="text-slate-400 text-xs mt-1">{metric.description}</p>
                    </div>
                  </div>
                  <div className={`w-2 h-2 rounded-full ${getStatusColor(metric.status)} animate-pulse`}></div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div>
                    <div className="text-2xl font-bold text-white">
                      {formatValue(metric.value, metric.unit)}{metric.unit}
                    </div>
                    <div className="flex items-center space-x-2 mt-2">
                      <TrendingUp className="w-3 h-3 text-green-400" />
                      <span className="text-green-400 text-xs font-medium">
                        +{metric.change.toFixed(1)}% this week
                      </span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Intelligence Network Overview */}
      <Card className="bg-gradient-to-br from-slate-800/50 to-purple-900/20 border-purple-500/30">
        <CardHeader>
          <CardTitle className="text-purple-300 flex items-center">
            <Sparkles className="w-5 h-5 mr-2" />
            AI Intelligence Network
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-full flex items-center justify-center border border-cyan-500/30 mx-auto mb-3">
                <Brain className="w-8 h-8 text-cyan-400" />
              </div>
              <div className="text-xl font-bold text-cyan-400">47,891</div>
              <div className="text-xs text-slate-400">AI Insights Generated</div>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-full flex items-center justify-center border border-green-500/30 mx-auto mb-3">
                <Network className="w-8 h-8 text-green-400" />
              </div>
              <div className="text-xl font-bold text-green-400">1,247</div>
              <div className="text-xs text-slate-400">Cross-Domain Connections</div>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-full flex items-center justify-center border border-purple-500/30 mx-auto mb-3">
                <Database className="w-8 h-8 text-purple-400" />
              </div>
              <div className="text-xl font-bold text-purple-400">15.7TB</div>
              <div className="text-xs text-slate-400">Data Processed</div>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-orange-500/20 to-red-500/20 rounded-full flex items-center justify-center border border-orange-500/30 mx-auto mb-3">
                <BarChart3 className="w-8 h-8 text-orange-400" />
              </div>
              <div className="text-xl font-bold text-orange-400">98.9%</div>
              <div className="text-xs text-slate-400">Intelligence Accuracy</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
