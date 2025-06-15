
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { 
  Brain, 
  Globe, 
  Users, 
  Zap, 
  Activity, 
  Database, 
  Network, 
  Cpu,
  TrendingUp,
  Target
} from 'lucide-react';
import { useIntelligenceEngine } from '../hooks/useIntelligenceEngine';
import { PlatformPerformanceMonitor } from './PlatformPerformanceMonitor';
import { PLATFORM_CONFIG, getSystemStatus, formatMetric } from '../lib/platformConfig';

export const IntelligenceDashboardOptimized: React.FC = () => {
  const { metrics, realTimeData } = useIntelligenceEngine();
  const systemStatus = getSystemStatus(metrics);

  const coreMetrics = [
    {
      label: 'Intelligence Score',
      value: metrics.significanceScore,
      unit: '/10',
      icon: Brain,
      gradient: PLATFORM_CONFIG.ui.colors.gradients.intelligence,
      trend: '+0.3',
      description: 'AI learning and adaptation rate'
    },
    {
      label: 'Cross-Domain Links',
      value: metrics.crossDomainConnections,
      unit: '',
      icon: Globe,
      gradient: PLATFORM_CONFIG.ui.colors.gradients.system,
      trend: '+12',
      description: 'Active intelligence connections'
    },
    {
      label: 'System Health',
      value: metrics.systemHealth,
      unit: '%',
      icon: Cpu,
      gradient: PLATFORM_CONFIG.ui.colors.gradients.success,
      trend: '+2.1',
      description: 'Platform optimization rate'
    },
    {
      label: 'Learning Velocity',
      value: metrics.learningRate,
      unit: '%',
      icon: Zap,
      gradient: PLATFORM_CONFIG.ui.colors.gradients.warning,
      trend: '+1.8',
      description: 'AI knowledge acquisition speed'
    }
  ];

  const networkStats = [
    {
      label: 'Global Users',
      value: realTimeData.globalUsers,
      icon: Users,
      color: 'cyan'
    },
    {
      label: 'Data Processed',
      value: realTimeData.dataProcessed,
      unit: 'TB',
      icon: Database,
      color: 'green'
    },
    {
      label: 'AI Optimizations',
      value: realTimeData.aiOptimizations,
      icon: Target,
      color: 'purple'
    },
    {
      label: 'Active Connections',
      value: realTimeData.activeConnections,
      icon: Network,
      color: 'orange'
    }
  ];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* System Status Overview */}
      <Card className="bg-gradient-to-r from-green-900/20 to-emerald-900/20 border-green-500/30 hover:border-green-400/50 transition-all duration-300">
        <CardHeader>
          <CardTitle className="text-green-300 flex items-center justify-between">
            <div className="flex items-center">
              <Activity className="w-5 h-5 mr-2 animate-pulse" />
              AI Intelligence System Status
            </div>
            <div className="flex items-center space-x-2">
              <div 
                className="w-3 h-3 rounded-full animate-pulse"
                style={{ backgroundColor: systemStatus.color }}
              ></div>
              <span className="text-sm font-medium uppercase" style={{ color: systemStatus.color }}>
                {systemStatus.status}
              </span>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-3 bg-green-500/10 rounded-lg border border-green-500/20">
              <div className="text-xl font-bold text-green-400">{metrics.accuracy.toFixed(1)}%</div>
              <div className="text-xs text-green-300">AI Accuracy</div>
            </div>
            <div className="text-center p-3 bg-cyan-500/10 rounded-lg border border-cyan-500/20">
              <div className="text-xl font-bold text-cyan-400">{realTimeData.responseTime}ms</div>
              <div className="text-xs text-cyan-300">Response Time</div>
            </div>
            <div className="text-center p-3 bg-purple-500/10 rounded-lg border border-purple-500/20">
              <div className="text-xl font-bold text-purple-400">{formatMetric(realTimeData.throughput, 'req/min')}</div>
              <div className="text-xs text-purple-300">Throughput</div>
            </div>
            <div className="text-center p-3 bg-orange-500/10 rounded-lg border border-orange-500/20">
              <div className="text-xl font-bold text-orange-400">{metrics.processingSpeed.toFixed(1)}%</div>
              <div className="text-xs text-orange-300">Processing Speed</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Monitor */}
      <PlatformPerformanceMonitor />

      {/* Core Intelligence Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {coreMetrics.map((metric) => {
          const IconComponent = metric.icon;
          return (
            <Card 
              key={metric.label} 
              className="bg-slate-800/50 border-slate-700/50 hover:border-cyan-500/30 transition-all duration-300 group hover:scale-105"
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className={`w-12 h-12 bg-gradient-to-r ${metric.gradient} bg-opacity-20 rounded-xl flex items-center justify-center border border-white/20 group-hover:scale-110 transition-transform duration-300`}>
                    <IconComponent className="w-6 h-6 text-white" />
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-green-400 flex items-center">
                      <TrendingUp className="w-3 h-3 mr-1" />
                      {metric.trend}
                    </div>
                  </div>
                </div>
                <div className="text-2xl font-bold text-white mb-1 group-hover:text-cyan-300 transition-colors">
                  {formatMetric(metric.value, metric.unit)}{metric.unit}
                </div>
                <div className="text-sm text-slate-400 mb-2">{metric.label}</div>
                <div className="text-xs text-slate-500">{metric.description}</div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Global Intelligence Network */}
      <Card className="bg-gradient-to-br from-slate-800/50 to-purple-900/20 border-purple-500/30 hover:border-purple-400/50 transition-all duration-300">
        <CardHeader>
          <CardTitle className="text-purple-300 flex items-center justify-between">
            <div className="flex items-center">
              <Network className="w-5 h-5 mr-2" />
              Global Intelligence Network
            </div>
            <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/30">
              Live Data
            </Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {networkStats.map((stat) => {
              const IconComponent = stat.icon;
              const colorClass = `text-${stat.color}-400`;
              const bgClass = `from-${stat.color}-500/20 to-${stat.color}-600/20`;
              const borderClass = `border-${stat.color}-500/30`;
              
              return (
                <div key={stat.label} className="text-center group">
                  <div className={`w-16 h-16 bg-gradient-to-r ${bgClass} rounded-full flex items-center justify-center border ${borderClass} mx-auto mb-3 group-hover:scale-110 transition-transform duration-300`}>
                    <IconComponent className={`w-8 h-8 ${colorClass}`} />
                  </div>
                  <div className={`text-xl font-bold ${colorClass} group-hover:scale-105 transition-transform duration-300`}>
                    {formatMetric(stat.value, stat.unit || '')}{stat.unit || ''}
                  </div>
                  <div className="text-xs text-slate-400 mt-1">{stat.label}</div>
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
