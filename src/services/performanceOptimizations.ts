/**
 * Performance Optimizations - Phase 15
 * Caching strategies, CDN preparation, and edge computing enhancements
 */

import { loggingService } from './loggingService';

export class PerformanceOptimizer {
  private cache: Map<string, { data: any; timestamp: number; ttl: number }> = new Map();
  private static instance: PerformanceOptimizer;

  static getInstance(): PerformanceOptimizer {
    if (!PerformanceOptimizer.instance) {
      PerformanceOptimizer.instance = new PerformanceOptimizer();
    }
    return PerformanceOptimizer.instance;
  }

  // Advanced Caching Strategy
  setCache(key: string, data: any, ttlMinutes: number = 15): void {
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl: ttlMinutes * 60 * 1000
    });

    loggingService.log('info', 'performance', `Cached data for key: ${key}`, {
      size: JSON.stringify(data).length,
      ttl: ttlMinutes
    });
  }

  getCache<T>(key: string): T | null {
    const cached = this.cache.get(key);
    
    if (!cached) {
      loggingService.log('debug', 'performance', `Cache miss: ${key}`);
      return null;
    }

    if (Date.now() - cached.timestamp > cached.ttl) {
      this.cache.delete(key);
      loggingService.log('debug', 'performance', `Cache expired: ${key}`);
      return null;
    }

    loggingService.log('debug', 'performance', `Cache hit: ${key}`);
    return cached.data as T;
  }

  clearCache(pattern?: string): void {
    if (pattern) {
      const keysToDelete: string[] = [];
      this.cache.forEach((_, key) => {
        if (key.includes(pattern)) {
          keysToDelete.push(key);
        }
      });
      keysToDelete.forEach(key => this.cache.delete(key));
      loggingService.log('info', 'performance', `Cleared cache pattern: ${pattern}`, {
        count: keysToDelete.length
      });
    } else {
      const count = this.cache.size;
      this.cache.clear();
      loggingService.log('info', 'performance', `Cleared all cache`, { count });
    }
  }

  // CDN Resource Hints
  addResourceHints(): void {
    const head = document.head;

    // DNS Prefetch for external resources
    const dnsPrefetch = [
      'https://fonts.googleapis.com',
      'https://cdn.jsdelivr.net'
    ];

    dnsPrefetch.forEach(url => {
      const link = document.createElement('link');
      link.rel = 'dns-prefetch';
      link.href = url;
      head.appendChild(link);
    });

    // Preconnect for critical resources
    const preconnect = [
      'https://fonts.gstatic.com'
    ];

    preconnect.forEach(url => {
      const link = document.createElement('link');
      link.rel = 'preconnect';
      link.href = url;
      link.crossOrigin = 'anonymous';
      head.appendChild(link);
    });

    loggingService.log('info', 'performance', 'Resource hints added for CDN optimization');
  }

  // Image Lazy Loading Enhancement
  optimizeImages(): void {
    const images = document.querySelectorAll('img[data-src]');
    
    const imageObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target as HTMLImageElement;
          const src = img.getAttribute('data-src');
          if (src) {
            img.src = src;
            img.removeAttribute('data-src');
            imageObserver.unobserve(img);
          }
        }
      });
    }, {
      rootMargin: '50px'
    });

    images.forEach(img => imageObserver.observe(img));
  }

  // Request Batching
  private requestBatches: Map<string, {
    requests: Array<{ resolve: Function; reject: Function; params: any }>;
    timeout: NodeJS.Timeout;
  }> = new Map();

  batchRequest<T>(
    endpoint: string,
    params: any,
    batchFn: (allParams: any[]) => Promise<T[]>,
    delayMs: number = 50
  ): Promise<T> {
    return new Promise((resolve, reject) => {
      let batch = this.requestBatches.get(endpoint);

      if (!batch) {
        batch = {
          requests: [],
          timeout: setTimeout(async () => {
            const currentBatch = this.requestBatches.get(endpoint);
            if (!currentBatch) return;

            this.requestBatches.delete(endpoint);
            const allParams = currentBatch.requests.map(r => r.params);

            try {
              const results = await batchFn(allParams);
              currentBatch.requests.forEach((req, index) => {
                req.resolve(results[index]);
              });

              loggingService.log('info', 'performance', `Batch request completed for ${endpoint}`, {
                count: currentBatch.requests.length
              });
            } catch (error) {
              currentBatch.requests.forEach(req => req.reject(error));
            }
          }, delayMs)
        };
        this.requestBatches.set(endpoint, batch);
      }

      batch.requests.push({ resolve, reject, params });
    });
  }

  // Memory Management
  cleanupMemory(): void {
    // Clear expired cache
    const now = Date.now();
    const toDelete: string[] = [];

    this.cache.forEach((value, key) => {
      if (now - value.timestamp > value.ttl) {
        toDelete.push(key);
      }
    });

    toDelete.forEach(key => this.cache.delete(key));

    if (toDelete.length > 0) {
      loggingService.log('info', 'performance', `Memory cleanup completed`, {
        itemsRemoved: toDelete.length
      });
    }

    // Force garbage collection hint (if available)
    if (global.gc) {
      global.gc();
    }
  }

  // Performance Metrics
  measureOperation<T>(
    operationName: string,
    operation: () => Promise<T>
  ): Promise<T> {
    const start = performance.now();

    return operation().then(
      result => {
        const duration = performance.now() - start;
        loggingService.logPerformance(operationName, duration, { success: true });
        return result;
      },
      error => {
        const duration = performance.now() - start;
        loggingService.logPerformance(operationName, duration, { success: false, error: error.message });
        throw error;
      }
    );
  }

  // Initialize all optimizations
  initialize(): void {
    console.log('⚡ Performance optimizations initializing...');
    
    this.addResourceHints();
    this.optimizeImages();
    
    // Schedule periodic cleanup
    setInterval(() => this.cleanupMemory(), 5 * 60 * 1000); // Every 5 minutes

    loggingService.log('info', 'performance', 'Performance optimizations initialized');
    console.log('✅ Performance optimizations active');
  }

  getStats() {
    return {
      cacheSize: this.cache.size,
      cacheSizeBytes: Array.from(this.cache.values()).reduce(
        (sum, item) => sum + JSON.stringify(item.data).length,
        0
      ),
      activeBatches: this.requestBatches.size
    };
  }
}

export const performanceOptimizer = PerformanceOptimizer.getInstance();
