// Performance monitoring and optimization service
export interface PerformanceMetrics {
  memoryUsage: number;
  renderTime: number;
  apiLatency: number;
  errorRate: number;
  cacheHitRatio: number;
  activeConnections: number;
}

export interface PerformanceAlert {
  type: 'memory' | 'performance' | 'error' | 'network';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: number;
  metric: string;
  value: number;
  threshold: number;
}

class PerformanceMonitor {
  private metrics: PerformanceMetrics = {
    memoryUsage: 0,
    renderTime: 0,
    apiLatency: 0,
    errorRate: 0,
    cacheHitRatio: 0,
    activeConnections: 0
  };

  private alerts: PerformanceAlert[] = [];
  private listeners: Set<(metrics: PerformanceMetrics) => void> = new Set();
  private alertListeners: Set<(alert: PerformanceAlert) => void> = new Set();
  private isMonitoring = false;
  private performanceEntries: PerformanceEntry[] = [];
  private errorCount = 0;
  private requestCount = 0;

  // Performance thresholds
  private thresholds = {
    memoryUsage: 100, // MB
    renderTime: 16, // ms (60 FPS)
    apiLatency: 2000, // ms
    errorRate: 0.05, // 5%
    cacheHitRatio: 0.6 // 60%
  };

  startMonitoring(): void {
    if (this.isMonitoring) return;
    
    this.isMonitoring = true;
    console.log('Performance monitoring started');

    // Monitor performance every 10 seconds
    setInterval(() => {
      this.updateMetrics();
    }, 10000);

    // Setup performance observer
    this.setupPerformanceObserver();

    // Setup error tracking
    this.setupErrorTracking();

    // Initial metrics update
    this.updateMetrics();
  }

  private setupPerformanceObserver(): void {
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          this.performanceEntries.push(...list.getEntries());
          
          // Keep only last 100 entries
          if (this.performanceEntries.length > 100) {
            this.performanceEntries = this.performanceEntries.slice(-100);
          }
        });

        observer.observe({ 
          entryTypes: ['navigation', 'resource', 'measure', 'paint'] 
        });
      } catch (error) {
        console.warn('PerformanceObserver not supported:', error);
      }
    }
  }

  private setupErrorTracking(): void {
    // Track unhandled errors
    window.addEventListener('error', (event) => {
      this.errorCount++;
      this.checkErrorRate();
    });

    // Track unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
      this.errorCount++;
      this.checkErrorRate();
      console.warn('Unhandled promise rejection:', event.reason);
    });
  }

  private updateMetrics(): void {
    // Memory usage
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      this.metrics.memoryUsage = Math.round(memory.usedJSHeapSize / 1024 / 1024);
    }

    // Render time (based on performance entries)
    const paintEntries = this.performanceEntries.filter(
      entry => entry.entryType === 'paint'
    );
    if (paintEntries.length > 0) {
      const lastPaint = paintEntries[paintEntries.length - 1];
      this.metrics.renderTime = Math.round(lastPaint.startTime);
    }

    // API latency (average of recent resource entries)
    const resourceEntries = this.performanceEntries
      .filter(entry => entry.entryType === 'resource')
      .slice(-10); // Last 10 requests

    if (resourceEntries.length > 0) {
      const avgLatency = resourceEntries.reduce((sum, entry) => 
        sum + entry.duration, 0) / resourceEntries.length;
      this.metrics.apiLatency = Math.round(avgLatency);
    }

    // Error rate
    this.metrics.errorRate = this.requestCount > 0 ? 
      this.errorCount / this.requestCount : 0;

    // Check thresholds and create alerts
    this.checkThresholds();

    // Notify listeners
    this.notifyListeners();
  }

  private checkThresholds(): void {
    const checks = [
      {
        metric: 'memoryUsage',
        value: this.metrics.memoryUsage,
        threshold: this.thresholds.memoryUsage,
        type: 'memory' as const,
        message: `High memory usage: ${this.metrics.memoryUsage}MB`
      },
      {
        metric: 'renderTime',
        value: this.metrics.renderTime,
        threshold: this.thresholds.renderTime,
        type: 'performance' as const,
        message: `Slow render time: ${this.metrics.renderTime}ms`
      },
      {
        metric: 'apiLatency',
        value: this.metrics.apiLatency,
        threshold: this.thresholds.apiLatency,
        type: 'network' as const,
        message: `High API latency: ${this.metrics.apiLatency}ms`
      },
      {
        metric: 'errorRate',
        value: this.metrics.errorRate,
        threshold: this.thresholds.errorRate,
        type: 'error' as const,
        message: `High error rate: ${(this.metrics.errorRate * 100).toFixed(1)}%`
      }
    ];

    checks.forEach(check => {
      if (check.value > check.threshold) {
        this.createAlert(
          check.type,
          this.getSeverity(check.value, check.threshold),
          check.message,
          check.metric,
          check.value,
          check.threshold
        );
      }
    });
  }

  private checkErrorRate(): void {
    this.requestCount++;
    
    // Update error rate
    this.metrics.errorRate = this.errorCount / this.requestCount;
    
    if (this.metrics.errorRate > this.thresholds.errorRate) {
      this.createAlert(
        'error',
        'high',
        `Error rate increased to ${(this.metrics.errorRate * 100).toFixed(1)}%`,
        'errorRate',
        this.metrics.errorRate,
        this.thresholds.errorRate
      );
    }
  }

  private getSeverity(value: number, threshold: number): PerformanceAlert['severity'] {
    const ratio = value / threshold;
    if (ratio > 3) return 'critical';
    if (ratio > 2) return 'high';
    if (ratio > 1.5) return 'medium';
    return 'low';
  }

  private createAlert(
    type: PerformanceAlert['type'],
    severity: PerformanceAlert['severity'],
    message: string,
    metric: string,
    value: number,
    threshold: number
  ): void {
    // Don't create duplicate alerts within 30 seconds
    const recentAlert = this.alerts.find(alert => 
      alert.metric === metric && 
      Date.now() - alert.timestamp < 30000
    );
    
    if (recentAlert) return;

    const alert: PerformanceAlert = {
      type,
      severity,
      message,
      timestamp: Date.now(),
      metric,
      value,
      threshold
    };

    this.alerts.push(alert);
    
    // Keep only last 50 alerts
    if (this.alerts.length > 50) {
      this.alerts = this.alerts.slice(-50);
    }

    // Notify alert listeners
    this.alertListeners.forEach(listener => {
      try {
        listener(alert);
      } catch (error) {
        console.error('Error in performance alert listener:', error);
      }
    });

    console.warn(`Performance Alert [${severity.toUpperCase()}]:`, message);
  }

  private notifyListeners(): void {
    this.listeners.forEach(listener => {
      try {
        listener({ ...this.metrics });
      } catch (error) {
        console.error('Error in performance listener:', error);
      }
    });
  }

  // Public API
  getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  getAlerts(): PerformanceAlert[] {
    return [...this.alerts];
  }

  subscribe(callback: (metrics: PerformanceMetrics) => void): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  subscribeToAlerts(callback: (alert: PerformanceAlert) => void): () => void {
    this.alertListeners.add(callback);
    return () => this.alertListeners.delete(callback);
  }

  // Mark API request for tracking
  markRequest(): void {
    this.requestCount++;
  }

  // Mark API error for tracking
  markError(): void {
    this.errorCount++;
    this.checkErrorRate();
  }

  // Update cache hit ratio
  updateCacheHitRatio(hits: number, total: number): void {
    this.metrics.cacheHitRatio = total > 0 ? hits / total : 0;
  }

  // Get performance summary
  getPerformanceSummary(): {
    score: number;
    status: 'excellent' | 'good' | 'fair' | 'poor';
    recommendations: string[];
  } {
    const scores = {
      memory: this.metrics.memoryUsage < 50 ? 100 : Math.max(0, 100 - (this.metrics.memoryUsage - 50) * 2),
      render: this.metrics.renderTime < 16 ? 100 : Math.max(0, 100 - (this.metrics.renderTime - 16) * 5),
      latency: this.metrics.apiLatency < 500 ? 100 : Math.max(0, 100 - (this.metrics.apiLatency - 500) / 20),
      errors: this.metrics.errorRate < 0.01 ? 100 : Math.max(0, 100 - this.metrics.errorRate * 1000),
      cache: this.metrics.cacheHitRatio * 100
    };

    const overallScore = Math.round(
      (scores.memory + scores.render + scores.latency + scores.errors + scores.cache) / 5
    );

    let status: 'excellent' | 'good' | 'fair' | 'poor';
    if (overallScore >= 90) status = 'excellent';
    else if (overallScore >= 75) status = 'good';
    else if (overallScore >= 60) status = 'fair';
    else status = 'poor';

    const recommendations: string[] = [];
    if (scores.memory < 80) recommendations.push('Optimize memory usage');
    if (scores.render < 80) recommendations.push('Improve render performance');
    if (scores.latency < 80) recommendations.push('Reduce API latency');
    if (scores.errors < 80) recommendations.push('Fix error handling');
    if (scores.cache < 80) recommendations.push('Improve caching strategy');

    return {
      score: overallScore,
      status,
      recommendations
    };
  }

  // Clear metrics and reset
  reset(): void {
    this.metrics = {
      memoryUsage: 0,
      renderTime: 0,
      apiLatency: 0,
      errorRate: 0,
      cacheHitRatio: 0,
      activeConnections: 0
    };
    this.alerts = [];
    this.performanceEntries = [];
    this.errorCount = 0;
    this.requestCount = 0;
  }
}

export const performanceMonitor = new PerformanceMonitor();
