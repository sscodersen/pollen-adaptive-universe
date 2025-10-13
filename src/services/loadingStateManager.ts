/**
 * Universal Loading State Manager
 * Manages loading states across all AI-powered features
 */

export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface LoadingStateInfo {
  state: LoadingState;
  progress?: number;
  message?: string;
  error?: string;
  timestamp: number;
}

class LoadingStateManager {
  private states = new Map<string, LoadingStateInfo>();
  private listeners = new Map<string, Set<(state: LoadingStateInfo) => void>>();

  /**
   * Set loading state for a feature
   */
  setState(feature: string, state: LoadingState, options?: {
    progress?: number;
    message?: string;
    error?: string;
  }) {
    const stateInfo: LoadingStateInfo = {
      state,
      progress: options?.progress,
      message: options?.message,
      error: options?.error,
      timestamp: Date.now()
    };

    this.states.set(feature, stateInfo);
    this.notifyListeners(feature, stateInfo);
  }

  /**
   * Get current loading state
   */
  getState(feature: string): LoadingStateInfo {
    return this.states.get(feature) || {
      state: 'idle',
      timestamp: Date.now()
    };
  }

  /**
   * Subscribe to state changes
   */
  subscribe(feature: string, callback: (state: LoadingStateInfo) => void): () => void {
    if (!this.listeners.has(feature)) {
      this.listeners.set(feature, new Set());
    }
    
    this.listeners.get(feature)!.add(callback);

    // Return unsubscribe function
    return () => {
      const listeners = this.listeners.get(feature);
      if (listeners) {
        listeners.delete(callback);
      }
    };
  }

  /**
   * Notify all listeners of state change
   */
  private notifyListeners(feature: string, state: LoadingStateInfo) {
    const listeners = this.listeners.get(feature);
    if (listeners) {
      listeners.forEach(callback => callback(state));
    }
  }

  /**
   * Set loading with automatic success after async operation
   */
  async withLoading<T>(
    feature: string,
    operation: () => Promise<T>,
    messages?: {
      loading?: string;
      success?: string;
      error?: string;
    }
  ): Promise<T> {
    this.setState(feature, 'loading', {
      message: messages?.loading || 'Loading...'
    });

    try {
      const result = await operation();
      this.setState(feature, 'success', {
        message: messages?.success || 'Success'
      });

      // Auto-reset to idle after 2 seconds
      setTimeout(() => {
        if (this.getState(feature).state === 'success') {
          this.setState(feature, 'idle');
        }
      }, 2000);

      return result;
    } catch (error) {
      this.setState(feature, 'error', {
        error: messages?.error || (error as Error).message
      });

      // Auto-reset to idle after 5 seconds
      setTimeout(() => {
        if (this.getState(feature).state === 'error') {
          this.setState(feature, 'idle');
        }
      }, 5000);

      throw error;
    }
  }

  /**
   * Update progress for a loading operation
   */
  updateProgress(feature: string, progress: number, message?: string) {
    const currentState = this.getState(feature);
    if (currentState.state === 'loading') {
      this.setState(feature, 'loading', {
        progress: Math.min(100, Math.max(0, progress)),
        message: message || currentState.message
      });
    }
  }

  /**
   * Clear all states
   */
  clearAll() {
    this.states.clear();
  }

  /**
   * Get all active loading features
   */
  getActiveLoading(): string[] {
    return Array.from(this.states.entries())
      .filter(([_, state]) => state.state === 'loading')
      .map(([feature]) => feature);
  }

  /**
   * Check if any feature is loading
   */
  isAnyLoading(): boolean {
    return Array.from(this.states.values()).some(state => state.state === 'loading');
  }
}

export const loadingStateManager = new LoadingStateManager();

// Feature keys for consistency
export const LoadingFeatures = {
  SEARCH: 'search',
  AI_SEARCH_SUGGESTIONS: 'ai_search_suggestions',
  FEED_POSTS: 'feed_posts',
  NEWS_ARTICLES: 'news_articles',
  MUSIC_GENERATION: 'music_generation',
  SHOP_PRODUCTS: 'shop_products',
  ENTERTAINMENT_CONTENT: 'entertainment_content',
  AI_DETECTOR: 'ai_detector',
  CROP_ANALYZER: 'crop_analyzer',
  WELLNESS_TIPS: 'wellness_tips',
  COMMUNITY_POSTS: 'community_posts',
  ANALYTICS: 'analytics',
  CONTINUOUS_GENERATION: 'continuous_generation',
  POLLEN_AI_STATUS: 'pollen_ai_status'
} as const;
