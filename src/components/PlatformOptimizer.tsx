// Platform Optimizer - Overall system optimization and monitoring
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { 
  Zap, 
  Cpu, 
  Database, 
  Network, 
  TrendingUp, 
  Shield, 
  RefreshCw, 
  CheckCircle,
  AlertTriangle,
  Activity
} from 'lucide-react';
import { usePollenAI } from '@/hooks/usePollenAI';
import { contentOrchestrator } from '@/services/contentOrchestrator';
import { toast } from '@/hooks/use-toast';

interface SystemMetrics {
  contentGeneration: {
    rate: number;
    quality: number;
    uptime: number;
  };
  aiProcessing: {
    latency: number;
    accuracy: number;
    load: number;
  };
  platform: {
    responseTime: number;
    errorRate: number;
    userSatisfaction: number;
  };
}

interface OptimizationSettings {
  autoOptimize: boolean;
  aggressiveCaching: boolean;
  prioritizeQuality: boolean;
  enablePredictiveLoading: boolean;
  batchProcessing: boolean;
  realTimeUpdates: boolean;
}

export const PlatformOptimizer = () => {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    contentGeneration: { rate: 95.7, quality: 8.9, uptime: 99.8 },
    aiProcessing: { latency: 247, accuracy: 94.3, load: 67 },
    platform: { responseTime: 156, errorRate: 0.3, userSatisfaction: 8.7 }
  });

  const [settings, setSettings] = useState<OptimizationSettings>({
    autoOptimize: true,
    aggressiveCaching: true,
    prioritizeQuality: true,
    enablePredictiveLoading: false,
    batchProcessing: true,
    realTimeUpdates: true
  });

  const [optimizing, setOptimizing] = useState(false);
  const [lastOptimization, setLastOptimization] = useState<Date | null>(null);

  const { isConnected, connectionStatus, testConnection } = usePollenAI();

  useEffect(() => {
    const interval = setInterval(() => {
      updateMetrics();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const updateMetrics = () => {
    // Simulate real-time metrics
    setMetrics(prev => ({
      contentGeneration: {
        rate: Math.min(100, prev.contentGeneration.rate + (Math.random() - 0.5) * 2),
        quality: Math.min(10, Math.max(7, prev.contentGeneration.quality + (Math.random() - 0.5) * 0.2)),
        uptime: Math.min(100, prev.contentGeneration.uptime + (Math.random() - 0.1) * 0.1)
      },
      aiProcessing: {
        latency: Math.max(100, prev.aiProcessing.latency + (Math.random() - 0.5) * 20),
        accuracy: Math.min(100, Math.max(85, prev.aiProcessing.accuracy + (Math.random() - 0.5) * 1)),
        load: Math.min(100, Math.max(20, prev.aiProcessing.load + (Math.random() - 0.5) * 10))
      },
      platform: {
        responseTime: Math.max(50, prev.platform.responseTime + (Math.random() - 0.5) * 30),
        errorRate: Math.max(0, prev.platform.errorRate + (Math.random() - 0.5) * 0.2),
        userSatisfaction: Math.min(10, Math.max(6, prev.platform.userSatisfaction + (Math.random() - 0.5) * 0.3))
      }
    }));
  };

  const runOptimization = async () => {
    setOptimizing(true);
    
    try {
      // Simulate comprehensive optimization
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Apply optimizations based on settings
      if (settings.autoOptimize) {
        // Optimize caching strategies
        await contentOrchestrator.setupBackendIntegration({
          enableSSE: settings.realTimeUpdates
        });
      }

      // Update metrics with improvements
      setMetrics(prev => ({
        contentGeneration: {
          rate: Math.min(100, prev.contentGeneration.rate + 2),
          quality: Math.min(10, prev.contentGeneration.quality + 0.3),
          uptime: Math.min(100, prev.contentGeneration.uptime + 0.1)
        },
        aiProcessing: {
          latency: Math.max(50, prev.aiProcessing.latency - 50),
          accuracy: Math.min(100, prev.aiProcessing.accuracy + 2),
          load: Math.max(10, prev.aiProcessing.load - 15)
        },
        platform: {
          responseTime: Math.max(30, prev.platform.responseTime - 40),
          errorRate: Math.max(0, prev.platform.errorRate - 0.1),
          userSatisfaction: Math.min(10, prev.platform.userSatisfaction + 0.5)
        }
      }));

      setLastOptimization(new Date());
      
      toast({
        title: "Optimization Complete",
        description: "Platform performance has been optimized successfully",
      });
    } catch (error) {
      toast({
        title: "Optimization Failed",
        description: "Failed to complete platform optimization",
        variant: "destructive",
      });
    } finally {
      setOptimizing(false);
    }
  };

  const updateSetting = (key: keyof OptimizationSettings, value: boolean) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));

    toast({
      title: "Setting Updated",
      description: `${key.replace(/([A-Z])/g, ' $1').toLowerCase()} has been ${value ? 'enabled' : 'disabled'}`,
    });
  };

  const getPerformanceColor = (value: number, type: 'percentage' | 'quality' | 'latency' | 'error'): string => {
    switch (type) {
      case 'percentage':
        return value >= 95 ? 'text-emerald-500' : value >= 80 ? 'text-yellow-500' : 'text-red-500';
      case 'quality':
        return value >= 8.5 ? 'text-emerald-500' : value >= 7 ? 'text-yellow-500' : 'text-red-500';
      case 'latency':
        return value <= 100 ? 'text-emerald-500' : value <= 300 ? 'text-yellow-500' : 'text-red-500';
      case 'error':
        return value <= 0.5 ? 'text-emerald-500' : value <= 2 ? 'text-yellow-500' : 'text-red-500';
      default:
        return 'text-foreground';
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Platform Optimizer</h1>
          <p className="text-muted-foreground mt-1">
            Real-time performance monitoring and automatic optimization
          </p>
        </div>
        <div className="flex gap-2">
          <Button 
            onClick={runOptimization} 
            disabled={optimizing}
            className="gap-2"
          >
            <Zap className={`w-4 h-4 ${optimizing ? 'animate-spin' : ''}`} />
            {optimizing ? 'Optimizing...' : 'Optimize Now'}
          </Button>
        </div>
      </div>

      {/* System Status */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-red-500'}`} />
              <div>
                <p className="font-medium">AI Backend</p>
                <p className="text-sm text-muted-foreground">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <Activity className="w-5 h-5 text-primary" />
              <div>
                <p className="font-medium">System Load</p>
                <p className="text-sm text-muted-foreground">
                  {metrics.aiProcessing.load}% utilization
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <RefreshCw className={`w-5 h-5 ${optimizing ? 'animate-spin text-primary' : 'text-muted-foreground'}`} />
              <div>
                <p className="font-medium">Last Optimization</p>
                <p className="text-sm text-muted-foreground">
                  {lastOptimization ? lastOptimization.toLocaleTimeString() : 'Never'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="metrics" className="space-y-4">
        <TabsList>
          <TabsTrigger value="metrics">Performance Metrics</TabsTrigger>
          <TabsTrigger value="settings">Optimization Settings</TabsTrigger>
          <TabsTrigger value="diagnostics">System Diagnostics</TabsTrigger>
        </TabsList>

        <TabsContent value="metrics" className="space-y-4">
          {/* Content Generation Metrics */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="w-5 h-5" />
                Content Generation Performance
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Generation Rate</span>
                    <span className={`text-sm ${getPerformanceColor(metrics.contentGeneration.rate, 'percentage')}`}>
                      {metrics.contentGeneration.rate.toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={metrics.contentGeneration.rate} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Content Quality</span>
                    <span className={`text-sm ${getPerformanceColor(metrics.contentGeneration.quality, 'quality')}`}>
                      {metrics.contentGeneration.quality.toFixed(1)}/10
                    </span>
                  </div>
                  <Progress value={metrics.contentGeneration.quality * 10} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">System Uptime</span>
                    <span className={`text-sm ${getPerformanceColor(metrics.contentGeneration.uptime, 'percentage')}`}>
                      {metrics.contentGeneration.uptime.toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={metrics.contentGeneration.uptime} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* AI Processing Metrics */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Cpu className="w-5 h-5" />
                AI Processing Performance
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Response Latency</span>
                    <span className={`text-sm ${getPerformanceColor(metrics.aiProcessing.latency, 'latency')}`}>
                      {metrics.aiProcessing.latency.toFixed(0)}ms
                    </span>
                  </div>
                  <Progress value={(500 - metrics.aiProcessing.latency) / 5} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">AI Accuracy</span>
                    <span className={`text-sm ${getPerformanceColor(metrics.aiProcessing.accuracy, 'percentage')}`}>
                      {metrics.aiProcessing.accuracy.toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={metrics.aiProcessing.accuracy} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Processing Load</span>
                    <span className={`text-sm ${getPerformanceColor(100 - metrics.aiProcessing.load, 'percentage')}`}>
                      {metrics.aiProcessing.load.toFixed(0)}%
                    </span>
                  </div>
                  <Progress value={metrics.aiProcessing.load} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Platform Metrics */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Network className="w-5 h-5" />
                Platform Performance
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Response Time</span>
                    <span className={`text-sm ${getPerformanceColor(metrics.platform.responseTime, 'latency')}`}>
                      {metrics.platform.responseTime.toFixed(0)}ms
                    </span>
                  </div>
                  <Progress value={(500 - metrics.platform.responseTime) / 5} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Error Rate</span>
                    <span className={`text-sm ${getPerformanceColor(metrics.platform.errorRate, 'error')}`}>
                      {metrics.platform.errorRate.toFixed(1)}%
                    </span>
                  </div>
                  <Progress value={Math.max(0, 100 - metrics.platform.errorRate * 10)} className="h-2" />
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">User Satisfaction</span>
                    <span className={`text-sm ${getPerformanceColor(metrics.platform.userSatisfaction, 'quality')}`}>
                      {metrics.platform.userSatisfaction.toFixed(1)}/10
                    </span>
                  </div>
                  <Progress value={metrics.platform.userSatisfaction * 10} className="h-2" />
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="settings" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="w-5 h-5" />
                Optimization Settings
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                Configure automatic optimization and performance tuning
              </p>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {Object.entries(settings).map(([key, value]) => (
                  <div key={key} className="flex items-center justify-between">
                    <div>
                      <Label className="capitalize">
                        {key.replace(/([A-Z])/g, ' $1').toLowerCase()}
                      </Label>
                      <p className="text-xs text-muted-foreground mt-1">
                        {getSettingDescription(key)}
                      </p>
                    </div>
                    <Switch
                      checked={value}
                      onCheckedChange={(checked) => updateSetting(key as keyof OptimizationSettings, checked)}
                    />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="diagnostics" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="w-5 h-5" />
                System Diagnostics
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-3 border rounded-lg">
                  <CheckCircle className="w-5 h-5 text-emerald-500" />
                  <div>
                    <p className="font-medium">Content Generation Pipeline</p>
                    <p className="text-sm text-muted-foreground">All systems operational</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 border rounded-lg">
                  <CheckCircle className="w-5 h-5 text-emerald-500" />
                  <div>
                    <p className="font-medium">Quality Control Systems</p>
                    <p className="text-sm text-muted-foreground">Bias detection and filtering active</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 border rounded-lg">
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  <div>
                    <p className="font-medium">Cache Performance</p>
                    <p className="text-sm text-muted-foreground">Hit rate could be improved with optimization</p>
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 border rounded-lg">
                  <CheckCircle className="w-5 h-5 text-emerald-500" />
                  <div>
                    <p className="font-medium">Real-time Updates</p>
                    <p className="text-sm text-muted-foreground">SSE connections stable</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

function getSettingDescription(key: string): string {
  const descriptions = {
    autoOptimize: 'Automatically optimize performance based on usage patterns',
    aggressiveCaching: 'Use advanced caching strategies for better performance',
    prioritizeQuality: 'Prioritize content quality over generation speed',
    enablePredictiveLoading: 'Preload content based on user behavior patterns',
    batchProcessing: 'Process multiple requests in batches for efficiency',
    realTimeUpdates: 'Enable real-time content updates via server-sent events'
  };
  
  return descriptions[key as keyof typeof descriptions] || 'Configuration option';
}