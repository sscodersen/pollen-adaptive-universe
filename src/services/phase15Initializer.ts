/**
 * Phase 15 Initialization Service
 * Integrates all Phase 15 enhancements: logging, feedback, health checks, loading states
 */

import { loggingService } from './loggingService';
import { contentHealthCheck } from './contentHealthCheck';

export class Phase15Initializer {
  private static instance: Phase15Initializer;
  private initialized = false;

  static getInstance(): Phase15Initializer {
    if (!Phase15Initializer.instance) {
      Phase15Initializer.instance = new Phase15Initializer();
    }
    return Phase15Initializer.instance;
  }

  async initialize() {
    if (this.initialized) {
      console.log('📋 Phase 15 already initialized');
      return;
    }

    console.log('🚀 Initializing Phase 15 Enhancements...');

    // 1. Initialize logging service (auto-initialized on import)
    loggingService.log('info', 'performance', 'Phase 15 initialization started');

    // 2. Start content health monitoring
    contentHealthCheck.startMonitoring(60000); // Check every minute
    loggingService.log('info', 'performance', 'Content health monitoring started');

    // 3. Setup health check listeners
    contentHealthCheck.onHealthChange((result) => {
      if (result.overall === 'failed') {
        loggingService.log('error', 'content_generation', 
          'Content health check failed', { alerts: result.alerts });
      } else if (result.overall === 'degraded') {
        loggingService.log('warn', 'content_generation', 
          'Content health degraded', { alerts: result.alerts });
      }
    });

    // 4. Log platform capabilities
    this.logPlatformCapabilities();

    // 5. Setup performance monitoring
    this.setupPerformanceMonitoring();

    this.initialized = true;
    loggingService.log('info', 'performance', 'Phase 15 initialization completed');
    console.log('✅ Phase 15 Enhancements Initialized Successfully');
  }

  private logPlatformCapabilities() {
    loggingService.log('info', 'performance', 'Platform capabilities logged', {
      features: [
        'Enhanced Logging System',
        'User Feedback Integration',
        'Multi-Model AI Detection',
        'Advanced Crop Analysis',
        'Content Health Monitoring',
        'Universal Loading States',
        'Error Notifications',
        'Performance Tracking'
      ],
      version: '15.0.0',
      timestamp: new Date().toISOString()
    });
  }

  private setupPerformanceMonitoring() {
    // Monitor page load performance
    if (window.performance && window.performance.timing) {
      window.addEventListener('load', () => {
        setTimeout(() => {
          const perfData = window.performance.timing;
          const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
          const domReadyTime = perfData.domContentLoadedEventEnd - perfData.navigationStart;

          loggingService.logPerformance('page_load', pageLoadTime, {
            domReady: domReadyTime,
            resourcesLoaded: pageLoadTime - domReadyTime
          });
        }, 0);
      });
    }

    // Monitor long tasks
    if ('PerformanceObserver' in window) {
      try {
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.duration > 50) {
              loggingService.log('warn', 'performance', 
                `Long task detected: ${entry.duration.toFixed(2)}ms`, {
                  name: entry.name,
                  entryType: entry.entryType
                });
            }
          }
        });
        observer.observe({ entryTypes: ['measure', 'navigation'] });
      } catch (e) {
        console.log('Performance Observer not supported for long tasks');
      }
    }
  }

  getStatus() {
    return {
      initialized: this.initialized,
      logging: {
        active: true,
        analytics: loggingService.getAnalytics()
      },
      healthMonitoring: {
        active: true,
        sections: contentHealthCheck.getAllHealth()
      }
    };
  }
}

export const phase15Initializer = Phase15Initializer.getInstance();
