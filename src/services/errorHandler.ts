/**
 * Universal Error Handler
 * Centralizes error handling across all AI-powered features
 */

export interface ErrorInfo {
  code: string;
  message: string;
  userMessage: string;
  retryable: boolean;
  timestamp: number;
  context?: Record<string, any>;
}

class ErrorHandler {
  private errorLog: ErrorInfo[] = [];
  private maxLogSize = 100;

  /**
   * Handle and classify errors
   */
  handleError(error: any, context?: {
    feature?: string;
    operation?: string;
    data?: Record<string, any>;
  }): ErrorInfo {
    const errorInfo: ErrorInfo = {
      code: this.getErrorCode(error),
      message: error.message || 'Unknown error occurred',
      userMessage: this.getUserFriendlyMessage(error),
      retryable: this.isRetryable(error),
      timestamp: Date.now(),
      context
    };

    this.logError(errorInfo);
    return errorInfo;
  }

  /**
   * Get error code from error object
   */
  private getErrorCode(error: any): string {
    if (error.response?.status) {
      return `HTTP_${error.response.status}`;
    }
    if (error.code) {
      return error.code;
    }
    if (error.name) {
      return error.name;
    }
    return 'UNKNOWN_ERROR';
  }

  /**
   * Convert technical errors to user-friendly messages
   */
  private getUserFriendlyMessage(error: any): string {
    const status = error.response?.status;

    if (status === 404) {
      return 'The requested resource could not be found. Please try again.';
    }
    if (status === 500 || status === 502 || status === 503) {
      return 'Our AI service is temporarily unavailable. Using fallback mode.';
    }
    if (status === 429) {
      return 'Too many requests. Please wait a moment and try again.';
    }
    if (error.code === 'ECONNREFUSED' || error.code === 'ERR_NETWORK') {
      return 'Unable to connect to AI service. Using offline mode.';
    }
    if (error.code === 'TIMEOUT' || error.code === 'ETIMEDOUT') {
      return 'Request timed out. Please try again.';
    }
    if (error.message?.includes('timeout')) {
      return 'The operation took too long. Please try again.';
    }

    return 'Something went wrong. Please try again or use fallback content.';
  }

  /**
   * Determine if error is retryable
   */
  private isRetryable(error: any): boolean {
    const status = error.response?.status;
    const retryableStatuses = [408, 429, 500, 502, 503, 504];
    const retryableCodes = ['ECONNREFUSED', 'ETIMEDOUT', 'TIMEOUT', 'ERR_NETWORK'];

    if (retryableStatuses.includes(status)) {
      return true;
    }
    if (retryableCodes.includes(error.code)) {
      return true;
    }
    return false;
  }

  /**
   * Log error for debugging
   */
  private logError(errorInfo: ErrorInfo) {
    this.errorLog.push(errorInfo);

    // Keep log size manageable
    if (this.errorLog.length > this.maxLogSize) {
      this.errorLog = this.errorLog.slice(-this.maxLogSize);
    }

    // Console logging for development
    console.error('🚨 Error:', {
      code: errorInfo.code,
      message: errorInfo.message,
      context: errorInfo.context
    });
  }

  /**
   * Retry operation with exponential backoff
   */
  async retry<T>(
    operation: () => Promise<T>,
    options: {
      maxAttempts?: number;
      initialDelay?: number;
      maxDelay?: number;
      feature?: string;
    } = {}
  ): Promise<T> {
    const {
      maxAttempts = 3,
      initialDelay = 1000,
      maxDelay = 10000,
      feature = 'unknown'
    } = options;

    let lastError: any;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error;
        const errorInfo = this.handleError(error, { feature, operation: 'retry', data: { attempt } });

        if (!errorInfo.retryable || attempt === maxAttempts) {
          throw error;
        }

        // Calculate backoff delay
        const delay = Math.min(
          initialDelay * Math.pow(2, attempt - 1),
          maxDelay
        );

        console.log(`🔄 Retry attempt ${attempt}/${maxAttempts} for ${feature} in ${delay}ms`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw lastError;
  }

  /**
   * Get recent errors
   */
  getRecentErrors(count: number = 10): ErrorInfo[] {
    return this.errorLog.slice(-count);
  }

  /**
   * Get errors by feature
   */
  getErrorsByFeature(feature: string): ErrorInfo[] {
    return this.errorLog.filter(error => error.context?.feature === feature);
  }

  /**
   * Clear error log
   */
  clearErrors() {
    this.errorLog = [];
  }

  /**
   * Get error statistics
   */
  getErrorStats(): {
    total: number;
    retryable: number;
    byCode: Record<string, number>;
    byFeature: Record<string, number>;
  } {
    const stats = {
      total: this.errorLog.length,
      retryable: this.errorLog.filter(e => e.retryable).length,
      byCode: {} as Record<string, number>,
      byFeature: {} as Record<string, number>
    };

    this.errorLog.forEach(error => {
      stats.byCode[error.code] = (stats.byCode[error.code] || 0) + 1;
      const feature = error.context?.feature || 'unknown';
      stats.byFeature[feature] = (stats.byFeature[feature] || 0) + 1;
    });

    return stats;
  }
}

export const errorHandler = new ErrorHandler();
