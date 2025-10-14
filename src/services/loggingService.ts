/**
 * Comprehensive Logging and Monitoring Service
 * Phase 15: Enhanced logging with user interactions, errors, and performance metrics
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error' | 'critical';
export type LogCategory = 'user_interaction' | 'api_call' | 'performance' | 'error' | 'validation' | 'ai_operation' | 'content_generation';

export interface LogEntry {
  id: string;
  timestamp: string;
  level: LogLevel;
  category: LogCategory;
  message: string;
  metadata?: Record<string, any>;
  stackTrace?: string;
  userId?: string;
  sessionId?: string;
}

export interface PerformanceMetric {
  id: string;
  operation: string;
  duration: number;
  timestamp: string;
  success: boolean;
  metadata?: Record<string, any>;
}

export interface ErrorLog {
  id: string;
  timestamp: string;
  type: string;
  message: string;
  statusCode?: number;
  endpoint?: string;
  stackTrace?: string;
  userAction?: string;
}

class LoggingService {
  private logs: LogEntry[] = [];
  private performanceMetrics: PerformanceMetric[] = [];
  private errorLogs: ErrorLog[] = [];
  private maxLogs = 5000;
  private sessionId: string;

  constructor() {
    this.sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    this.initializeLogging();
  }

  private initializeLogging() {
    console.log('📊 Enhanced Logging Service initialized');
    console.log(`📝 Session ID: ${this.sessionId}`);
    
    // Intercept console methods for comprehensive logging
    this.interceptConsole();
    
    // Monitor network requests
    this.monitorNetworkRequests();
    
    // Track unhandled errors
    this.setupErrorTracking();
  }

  private interceptConsole() {
    const originalLog = console.log;
    const originalError = console.error;
    const originalWarn = console.warn;

    console.log = (...args: any[]) => {
      originalLog.apply(console, args);
    };

    console.error = (...args: any[]) => {
      this.log('error', 'error', args.join(' '), { stackTrace: new Error().stack });
      originalError.apply(console, args);
    };

    console.warn = (...args: any[]) => {
      this.log('warn', 'error', args.join(' '));
      originalWarn.apply(console, args);
    };
  }

  private monitorNetworkRequests() {
    const originalFetch = window.fetch;
    
    window.fetch = async (...args: any[]) => {
      const startTime = performance.now();
      const url = typeof args[0] === 'string' ? args[0] : args[0].url;
      
      try {
        const response = await originalFetch.apply(window, args);
        const duration = performance.now() - startTime;
        
        this.logAPICall(url, response.status, duration, response.ok);
        
        // Track 422 and validation errors
        if (response.status === 422) {
          const clonedResponse = response.clone();
          const errorData = await clonedResponse.json().catch(() => ({}));
          this.log422Error(url, errorData);
        }
        
        return response;
      } catch (error: any) {
        const duration = performance.now() - startTime;
        this.logAPICall(url, 0, duration, false, error.message);
        throw error;
      }
    };
  }

  private setupErrorTracking() {
    window.addEventListener('error', (event) => {
      this.logError({
        type: 'runtime_error',
        message: event.message,
        stackTrace: event.error?.stack,
        metadata: {
          filename: event.filename,
          lineno: event.lineno,
          colno: event.colno
        }
      });
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.logError({
        type: 'unhandled_promise_rejection',
        message: event.reason?.message || String(event.reason),
        stackTrace: event.reason?.stack,
        metadata: { reason: event.reason }
      });
    });
  }

  log(level: LogLevel, category: LogCategory, message: string, metadata?: Record<string, any>) {
    const entry: LogEntry = {
      id: `log-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      level,
      category,
      message,
      metadata,
      sessionId: this.sessionId
    };

    this.logs.push(entry);
    this.trimLogs();

    // Console output with color coding
    const emoji = this.getLevelEmoji(level);
    const categoryTag = `[${category}]`;
    console.debug(`${emoji} ${categoryTag} ${message}`, metadata || '');
  }

  logUserInteraction(action: string, element: string, metadata?: Record<string, any>) {
    this.log('info', 'user_interaction', `User ${action} on ${element}`, metadata);
  }

  logAPICall(endpoint: string, statusCode: number, duration: number, success: boolean, error?: string) {
    this.log('info', 'api_call', `API ${endpoint} - ${statusCode} (${duration.toFixed(2)}ms)`, {
      endpoint,
      statusCode,
      duration,
      success,
      error
    });

    // Add to performance metrics
    this.performanceMetrics.push({
      id: `perf-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      operation: endpoint,
      duration,
      timestamp: new Date().toISOString(),
      success,
      metadata: { statusCode, error }
    });
  }

  log422Error(endpoint: string, errorData: any) {
    this.log('error', 'validation', `422 Validation Error on ${endpoint}`, errorData);
    
    this.errorLogs.push({
      id: `error-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      type: '422_validation_error',
      message: 'Validation failed',
      statusCode: 422,
      endpoint,
      stackTrace: JSON.stringify(errorData, null, 2)
    });
  }

  logError(error: { type: string; message: string; stackTrace?: string; metadata?: Record<string, any> }) {
    this.log('error', 'error', `${error.type}: ${error.message}`, {
      ...error.metadata,
      stackTrace: error.stackTrace
    });

    this.errorLogs.push({
      id: `error-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date().toISOString(),
      type: error.type,
      message: error.message,
      stackTrace: error.stackTrace
    });
  }

  logPerformance(operation: string, duration: number, metadata?: Record<string, any>) {
    this.log('info', 'performance', `${operation} completed in ${duration.toFixed(2)}ms`, metadata);
    
    this.performanceMetrics.push({
      id: `perf-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      operation,
      duration,
      timestamp: new Date().toISOString(),
      success: true,
      metadata
    });
  }

  logAIOperation(operation: string, model: string, success: boolean, metadata?: Record<string, any>) {
    this.log(success ? 'info' : 'error', 'ai_operation', 
      `AI ${operation} using ${model} - ${success ? 'Success' : 'Failed'}`, metadata);
  }

  logContentGeneration(contentType: string, count: number, strategy: string, metadata?: Record<string, any>) {
    this.log('info', 'content_generation', 
      `Generated ${count} ${contentType} items using ${strategy}`, metadata);
  }

  private getLevelEmoji(level: LogLevel): string {
    const emojis = {
      debug: '🔍',
      info: '📝',
      warn: '⚠️',
      error: '❌',
      critical: '🚨'
    };
    return emojis[level];
  }

  private trimLogs() {
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(-this.maxLogs);
    }
    if (this.performanceMetrics.length > this.maxLogs) {
      this.performanceMetrics = this.performanceMetrics.slice(-this.maxLogs);
    }
    if (this.errorLogs.length > 1000) {
      this.errorLogs = this.errorLogs.slice(-1000);
    }
  }

  getLogs(filters?: { level?: LogLevel; category?: LogCategory; limit?: number }): LogEntry[] {
    let filtered = [...this.logs];
    
    if (filters?.level) {
      filtered = filtered.filter(log => log.level === filters.level);
    }
    if (filters?.category) {
      filtered = filtered.filter(log => log.category === filters.category);
    }
    
    const limit = filters?.limit || 100;
    return filtered.slice(-limit);
  }

  getPerformanceMetrics(operation?: string): PerformanceMetric[] {
    if (operation) {
      return this.performanceMetrics.filter(m => m.operation.includes(operation));
    }
    return this.performanceMetrics.slice(-100);
  }

  getErrorLogs(type?: string): ErrorLog[] {
    if (type) {
      return this.errorLogs.filter(e => e.type === type);
    }
    return this.errorLogs.slice(-100);
  }

  getAnalytics() {
    const errorRate = this.errorLogs.length / Math.max(this.logs.length, 1);
    const avgDuration = this.performanceMetrics.reduce((sum, m) => sum + m.duration, 0) / 
                        Math.max(this.performanceMetrics.length, 1);
    
    const errorsByType = this.errorLogs.reduce((acc, err) => {
      acc[err.type] = (acc[err.type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      totalLogs: this.logs.length,
      totalErrors: this.errorLogs.length,
      errorRate: (errorRate * 100).toFixed(2) + '%',
      avgResponseTime: avgDuration.toFixed(2) + 'ms',
      errorsByType,
      sessionId: this.sessionId,
      timestamp: new Date().toISOString()
    };
  }

  exportLogs() {
    return {
      logs: this.logs,
      performanceMetrics: this.performanceMetrics,
      errorLogs: this.errorLogs,
      analytics: this.getAnalytics()
    };
  }

  clearLogs() {
    this.logs = [];
    this.performanceMetrics = [];
    this.errorLogs = [];
    console.log('🧹 Logs cleared');
  }
}

export const loggingService = new LoggingService();
