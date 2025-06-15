
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { 
  Brain, Globe, Users, Zap, Activity, Database, Network, Cpu
} from 'lucide-react';
import { usePlatformState } from '../hooks/usePlatformState';
import { platformTheme, formatMetricValue } from '../lib/platformTheme';

export const IntelligenceDashboard: React.FC = () => {
  const { metrics, aiStatus } = usePlatformState();

  const intelligenceMetrics = [
    {
      label: 'Intelligence Score',
      value: metrics.significanceScore,
      unit: '/10',
      icon: Brain,
      color: platformTheme.colors.primary.purple,
      status: 'excellent' as const
    },
    {
      label: 'Cross-Domain Links',
      value: metrics.crossDomainConnections,
      unit: '',
      icon: Globe,
      color: platformTheme.colors.primary.cyan,
      status: 'excellent' as const
    },
    {
      label: 'System Health',
      value: metrics.systemHealth,
      unit: '%',
      icon: Cpu,
      color: platformTheme.colors.primary.green,
      status: 'excellent' as const
    },
    {
      label: 'Learning Velocity',
      value: metrics.learningVelocity,
      unit: '%',
      icon: Zap,
      color: platformTheme.colors.primary.orange,
      status: 'excellent' as const
    }
  ];

  const getStatusColor = (status: string) => {
    return platformTheme.colors.status[status as keyof typeof platformTheme.colors.status];
  };

  return (
    <div className="space-y-6">
      {/* AI Status Overview */}
      <Card className="bg-gradient-to-r from-green-900/20 to-emerald-900/20 border-green-500/30">
        <CardHeader>
          <CardTitle className="text-green-300 flex items-center">
            <Activity className="w-5 h-5 mr-2" />
            AI Intelligence System Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-xl font-bold text-green-400">OPTIMAL</div>
              <div className="text-xs text-green-300">System Status</div>
            </div>
            <div className="text-center">
              <div className="text-xl font-bold text-cyan-400">{metrics.intelligenceSynergy.toFixed(1)}%</div>
              <div className="text-xs text-cyan-300">AI Synergy</div>
            </div>
            <div className="text-center">
              <div className="text-xl font-bold text-purple-400">{metrics.responseTime}ms</div>
              <div className="text-xs text-purple-300">Response Time</div>
            </div>
            <div className="text-center">
              <div className="text-xl font-bold text-orange-400">{aiStatus.toUpperCase()}</div>
              <div className="text-xs text-orange-300">AI Status</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Intelligence Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {intelligenceMetrics.map((metric) => {
          const IconComponent = metric.icon;
          return (
            <Card 
              key={metric.label} 
              className="bg-slate-800/50 border-slate-700/50 hover:border-cyan-500/30 transition-all duration-300"
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className={`w-10 h-10 bg-gradient-to-r ${metric.color} bg-opacity-20 rounded-lg flex items-center justify-center border border-white/20`}>
                    <IconComponent className="w-5 h-5 text-white" />
                  </div>
                  <div className={`w-2 h-2 rounded-full ${getStatusColor(metric.status)} animate-pulse`}></div>
                </div>
                <div className="text-2xl font-bold text-white mb-1">
                  {formatMetricValue(metric.value, metric.unit)}{metric.unit}
                </div>
                <div className="text-sm text-slate-400">{metric.label}</div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Global Network Stats */}
      <Card className="bg-gradient-to-br from-slate-800/50 to-purple-900/20 border-purple-500/30">
        <CardHeader>
          <CardTitle className="text-purple-300 flex items-center">
            <Network className="w-5 h-5 mr-2" />
            Global Intelligence Network
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-full flex items-center justify-center border border-cyan-500/30 mx-auto mb-3">
                <Users className="w-8 h-8 text-cyan-400" />
              </div>
              <div className="text-xl font-bold text-cyan-400">{metrics.globalUsers.toLocaleString()}</div>
              <div className="text-xs text-slate-400">Active Global Users</div>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-green-500/20 to-emerald-500/20 rounded-full flex items-center justify-center border border-green-500/30 mx-auto mb-3">
                <Database className="w-8 h-8 text-green-400" />
              </div>
              <div className="text-xl font-bold text-green-400">{metrics.aiOptimizations}</div>
              <div className="text-xs text-slate-400">AI Optimizations</div>
            </div>
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-full flex items-center justify-center border border-purple-500/30 mx-auto mb-3">
                <Brain className="w-8 h-8 text-purple-400" />
              </div>
              <div className="text-xl font-bold text-purple-400">{metrics.crossDomainConnections}</div>
              <div className="text-xs text-slate-400">Cross-Domain Links</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
