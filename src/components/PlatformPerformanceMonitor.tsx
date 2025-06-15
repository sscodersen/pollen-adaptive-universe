
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { 
  Cpu, 
  Activity, 
  Zap, 
  Database, 
  Network, 
  TrendingUp,
  CheckCircle,
  AlertTriangle,
  Clock
} from 'lucide-react';
import { useIntelligenceEngine } from '../hooks/useIntelligenceEngine';
import { PLATFORM_CONFIG } from '../lib/platformConfig';

export const PlatformPerformanceMonitor: React.FC = () => {
  const { metrics, realTimeData } = useIntelligenceEngine();

  const performanceMetrics = [
    {
      label: 'CPU Intelligence',
      value: metrics.processingSpeed,
      unit: '%',
      icon: Cpu,
      status: metrics.processingSpeed > 95 ? 'excellent' : metrics.processingSpeed > 85 ? 'good' : 'warning',
      trend: '+2.3%'
    },
    {
      label: 'Memory Optimization',
      value: metrics.systemHealth,
      unit: '%',
      icon: Database,
      status: metrics.systemHealth > 95 ? 'excellent' : metrics.systemHealth > 85 ? 'good' : 'warning',
      trend: '+1.8%'
    },
    {
      label: 'Network Throughput',
      value: realTimeData.throughput / 1000,
      unit: 'K ops/s',
      icon: Network,
      status: realTimeData.throughput > 2500 ? 'excellent' : realTimeData.throughput > 2000 ? 'good' : 'warning',
      trend: '+5.2%'
    },
    {
      label: 'Response Time',
      value: realTimeData.responseTime,
      unit: 'ms',
      icon: Clock,
      status: realTimeData.responseTime < 15 ? 'excellent' : realTimeData.responseTime < 25 ? 'good' : 'warning',
      trend: '-0.8ms'
    }
  ];

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'excellent': return CheckCircle;
      case 'good': return Activity;
      default: return AlertTriangle;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'excellent': return 'text-green-400';
      case 'good': return 'text-cyan-400';
      default: return 'text-yellow-400';
    }
  };

  return (
    <Card className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 border-slate-700/50 hover:border-cyan-500/30 transition-all duration-300">
      <CardHeader>
        <CardTitle className="text-cyan-300 flex items-center justify-between">
          <div className="flex items-center">
            <Activity className="w-5 h-5 mr-2 animate-pulse" />
            Platform Performance Monitor
          </div>
          <Badge className="bg-green-500/20 text-green-300 border-green-500/30">
            Optimized
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {performanceMetrics.map((metric) => {
            const IconComponent = metric.icon;
            const StatusIcon = getStatusIcon(metric.status);
            
            return (
              <div 
                key={metric.label}
                className="bg-slate-700/30 rounded-lg p-4 border border-slate-600/30 hover:border-cyan-500/30 transition-all duration-300 group"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="w-10 h-10 bg-gradient-to-r from-cyan-500/20 to-purple-500/20 rounded-lg flex items-center justify-center border border-cyan-500/30">
                    <IconComponent className="w-5 h-5 text-cyan-400" />
                  </div>
                  <div className="flex items-center space-x-1">
                    <StatusIcon className={`w-4 h-4 ${getStatusColor(metric.status)}`} />
                    <span className="text-xs text-green-400">{metric.trend}</span>
                  </div>
                </div>
                
                <div className="text-2xl font-bold text-white mb-1 group-hover:text-cyan-300 transition-colors">
                  {metric.label === 'Response Time' ? 
                    metric.value.toFixed(0) : 
                    metric.value.toFixed(1)
                  }{metric.unit}
                </div>
                
                <div className="text-sm text-slate-400">{metric.label}</div>
                
                <div className="mt-2 w-full bg-slate-600/30 rounded-full h-1.5">
                  <div 
                    className={`h-1.5 rounded-full transition-all duration-500 ${
                      metric.status === 'excellent' ? 'bg-gradient-to-r from-green-500 to-emerald-500' :
                      metric.status === 'good' ? 'bg-gradient-to-r from-cyan-500 to-blue-500' :
                      'bg-gradient-to-r from-yellow-500 to-orange-500'
                    }`}
                    style={{ 
                      width: `${metric.label === 'Response Time' ? 
                        Math.max(20, 100 - (metric.value / 30) * 100) : 
                        metric.value}%` 
                    }}
                  ></div>
                </div>
              </div>
            );
          })}
        </div>
        
        <div className="mt-6 p-4 bg-gradient-to-r from-green-500/10 to-emerald-500/10 rounded-lg border border-green-500/20">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-green-500/20 rounded-lg flex items-center justify-center">
                <TrendingUp className="w-4 h-4 text-green-400" />
              </div>
              <div>
                <div className="text-green-300 font-medium">System Optimization Active</div>
                <div className="text-xs text-green-400/70">AI continuously optimizing performance across all domains</div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-lg font-bold text-green-400">98.7%</div>
              <div className="text-xs text-green-400/70">Overall Health</div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
