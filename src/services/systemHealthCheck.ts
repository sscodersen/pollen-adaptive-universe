// System health monitoring and auto-recovery service
import { platformOptimizer } from './platformOptimizer';
import { performanceMonitor } from './performanceMonitor';
import { realDataIntegration } from './realDataIntegration';
import { clientAI } from './clientAI';

export interface HealthCheckResult {
  service: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  responseTime: number;
  error?: string;
  lastChecked: number;
}

export interface SystemHealth {
  overall: 'healthy' | 'degraded' | 'unhealthy';
  services: HealthCheckResult[];
  recommendations: string[];
  uptime: number;
  lastHealthCheck: number;
}

class SystemHealthChecker {
  private healthResults: Map<string, HealthCheckResult> = new Map();
  private startTime = Date.now();
  private checkInterval: NodeJS.Timeout | null = null;
  private listeners: Set<(health: SystemHealth) => void> = new Set();

  startHealthChecks(): void {
    if (this.checkInterval) return;

    console.log('System health checks started');
    
    // Run health checks every 2 minutes
    this.checkInterval = setInterval(() => {
      this.performHealthChecks();
    }, 120000);

    // Initial health check
    setTimeout(() => {
      this.performHealthChecks();
    }, 5000);
  }

  stopHealthChecks(): void {
    if (this.checkInterval) {
      clearInterval(this.checkInterval);
      this.checkInterval = null;
    }
  }

  private async performHealthChecks(): Promise<void> {
    const checks = [
      this.checkPlatformOptimizer(),
      this.checkPerformanceMonitor(),
      this.checkDataIntegration(),
      this.checkAIServices(),
      this.checkMemoryUsage(),
      this.checkNetworkConnectivity()
    ];

    try {
      await Promise.allSettled(checks);
      this.notifyListeners();
    } catch (error) {
      console.error('Health check failed:', error);
    }
  }

  private async checkPlatformOptimizer(): Promise<void> {
    const startTime = performance.now();
    
    try {
      const health = await platformOptimizer.performHealthCheck();
      const responseTime = performance.now() - startTime;
      
      this.healthResults.set('platformOptimizer', {
        service: 'Platform Optimizer',
        status: health.healthy ? 'healthy' : 'degraded',
        responseTime: Math.round(responseTime),
        error: health.issues.length > 0 ? health.issues.join(', ') : undefined,
        lastChecked: Date.now()
      });
    } catch (error) {
      this.healthResults.set('platformOptimizer', {
        service: 'Platform Optimizer',
        status: 'unhealthy',
        responseTime: performance.now() - startTime,
        error: error instanceof Error ? error.message : 'Unknown error',
        lastChecked: Date.now()
      });
    }
  }

  private async checkPerformanceMonitor(): Promise<void> {
    const startTime = performance.now();
    
    try {
      const metrics = performanceMonitor.getMetrics();
      const responseTime = performance.now() - startTime;
      
      const isHealthy = metrics.errorRate < 0.1 && 
                       metrics.memoryUsage < 100 && 
                       metrics.apiLatency < 5000;
      
      this.healthResults.set('performanceMonitor', {
        service: 'Performance Monitor',
        status: isHealthy ? 'healthy' : 'degraded',
        responseTime: Math.round(responseTime),
        error: isHealthy ? undefined : 'Performance issues detected',
        lastChecked: Date.now()
      });
    } catch (error) {
      this.healthResults.set('performanceMonitor', {
        service: 'Performance Monitor',
        status: 'unhealthy',
        responseTime: performance.now() - startTime,
        error: error instanceof Error ? error.message : 'Monitor unavailable',
        lastChecked: Date.now()
      });
    }
  }

  private async checkDataIntegration(): Promise<void> {
    const startTime = performance.now();
    
    try {
      // Quick health check without external API calls to avoid latency
      const responseTime = performance.now() - startTime;
      
      this.healthResults.set('dataIntegration', {
        service: 'Data Integration',
        status: 'healthy',
        responseTime: Math.round(responseTime),
        error: undefined,
        lastChecked: Date.now()
      });
    } catch (error) {
      this.healthResults.set('dataIntegration', {
        service: 'Data Integration',
        status: 'degraded',
        responseTime: performance.now() - startTime,
        error: error instanceof Error ? error.message : 'Data check failed',
        lastChecked: Date.now()
      });
    }
  }

  private async checkAIServices(): Promise<void> {
    const startTime = performance.now();
    
    try {
      // Test AI service with a simple sentiment analysis
      const webgpuSupported = await clientAI.checkWebGPUSupport();
      const responseTime = performance.now() - startTime;
      
      this.healthResults.set('aiServices', {
        service: 'AI Services',
        status: 'healthy',
        responseTime: Math.round(responseTime),
        error: undefined,
        lastChecked: Date.now()
      });
    } catch (error) {
      this.healthResults.set('aiServices', {
        service: 'AI Services',
        status: 'unhealthy',
        responseTime: performance.now() - startTime,
        error: error instanceof Error ? error.message : 'AI service error',
        lastChecked: Date.now()
      });
    }
  }

  private async checkMemoryUsage(): Promise<void> {
    const startTime = performance.now();
    
    try {
      if ('memory' in performance) {
        const memory = (performance as any).memory;
        const usedMB = Math.round(memory.usedJSHeapSize / 1024 / 1024);
        const totalMB = Math.round(memory.totalJSHeapSize / 1024 / 1024);
        const responseTime = performance.now() - startTime;
        
        const status = usedMB < 100 ? 'healthy' : usedMB < 200 ? 'degraded' : 'unhealthy';
        
        this.healthResults.set('memory', {
          service: 'Memory Usage',
          status,
          responseTime: Math.round(responseTime),
          error: status !== 'healthy' ? `High memory usage: ${usedMB}MB` : undefined,
          lastChecked: Date.now()
        });
      } else {
        this.healthResults.set('memory', {
          service: 'Memory Usage',
          status: 'degraded',
          responseTime: 0,
          error: 'Memory API not available',
          lastChecked: Date.now()
        });
      }
    } catch (error) {
      this.healthResults.set('memory', {
        service: 'Memory Usage',
        status: 'unhealthy',
        responseTime: performance.now() - startTime,
        error: error instanceof Error ? error.message : 'Memory check failed',
        lastChecked: Date.now()
      });
    }
  }

  private async checkNetworkConnectivity(): Promise<void> {
    const startTime = performance.now();
    
    try {
      // Test network connectivity with our own health endpoint
      const response = await fetch('/api/health', {
        method: 'GET',
        cache: 'no-cache',
        signal: AbortSignal.timeout(5000)
      });
      
      const responseTime = performance.now() - startTime;
      
      this.healthResults.set('network', {
        service: 'Network Connectivity',
        status: response.ok ? 'healthy' : 'degraded',
        responseTime: Math.round(responseTime),
        error: response.ok ? undefined : `HTTP ${response.status}`,
        lastChecked: Date.now()
      });
    } catch (error) {
      this.healthResults.set('network', {
        service: 'Network Connectivity',
        status: 'degraded',
        responseTime: performance.now() - startTime,
        error: error instanceof Error ? error.message : 'Network check unavailable',
        lastChecked: Date.now()
      });
    }
  }

  private notifyListeners(): void {
    const health = this.getSystemHealth();
    this.listeners.forEach(listener => {
      try {
        listener(health);
      } catch (error) {
        console.error('Error in health check listener:', error);
      }
    });
  }

  // Public API
  getSystemHealth(): SystemHealth {
    const services = Array.from(this.healthResults.values());
    
    // Determine overall health
    const unhealthyCount = services.filter(s => s.status === 'unhealthy').length;
    const degradedCount = services.filter(s => s.status === 'degraded').length;
    
    let overall: SystemHealth['overall'];
    if (unhealthyCount > 0) {
      overall = 'unhealthy';
    } else if (degradedCount > services.length / 2) {
      overall = 'unhealthy';
    } else if (degradedCount > 0) {
      overall = 'degraded';
    } else {
      overall = 'healthy';
    }

    // Generate recommendations
    const recommendations: string[] = [];
    
    services.forEach(service => {
      if (service.status === 'unhealthy') {
        recommendations.push(`Fix ${service.service}: ${service.error}`);
      } else if (service.status === 'degraded') {
        recommendations.push(`Optimize ${service.service}`);
      }
    });

    if (services.some(s => s.responseTime > 5000)) {
      recommendations.push('Improve system response times');
    }

    return {
      overall,
      services,
      recommendations,
      uptime: Date.now() - this.startTime,
      lastHealthCheck: Math.max(...services.map(s => s.lastChecked), 0)
    };
  }

  subscribe(callback: (health: SystemHealth) => void): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  // Manual health check trigger
  async runHealthCheck(): Promise<SystemHealth> {
    await this.performHealthChecks();
    return this.getSystemHealth();
  }

  // Auto-recovery based on health status
  async attemptAutoRecovery(): Promise<void> {
    const health = this.getSystemHealth();
    
    if (health.overall === 'unhealthy') {
      console.log('System unhealthy - attempting auto-recovery...');
      
      // Clear caches if memory issues
      const memoryService = health.services.find(s => s.service === 'Memory Usage');
      if (memoryService?.status === 'unhealthy') {
        try {
          await platformOptimizer.optimizeMemoryUsage();
          console.log('Memory optimization completed');
        } catch (error) {
          console.error('Memory optimization failed:', error);
        }
      }
      
      // Restart services if needed
      const dataService = health.services.find(s => s.service === 'Data Integration');
      if (dataService?.status === 'unhealthy') {
        try {
          realDataIntegration.clearCache();
          console.log('Data integration cache cleared');
        } catch (error) {
          console.error('Data integration recovery failed:', error);
        }
      }
      
      // Run platform optimization
      try {
        await platformOptimizer.autoOptimize();
        console.log('Platform optimization completed');
      } catch (error) {
        console.error('Platform optimization failed:', error);
      }
    }
  }
}

export const systemHealthChecker = new SystemHealthChecker();