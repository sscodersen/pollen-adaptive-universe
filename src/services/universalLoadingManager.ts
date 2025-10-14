/**
 * Universal Loading State Manager
 * Phase 15: Consistent UX with centralized loading states
 */

import { useState, useEffect } from 'react';

export interface LoadingState {
  id: string;
  operation: string;
  message: string;
  progress?: number;
  startTime: number;
}

// React hook for loading states
export function useLoadingManager() {
  const [loadingStates, setLoadingStates] = useState<LoadingState[]>([]);
  const [globalLoading, setGlobalLoading] = useState(false);

  useEffect(() => {
    const handleUpdate = (event: CustomEvent) => {
      setLoadingStates(event.detail.states || []);
      setGlobalLoading(event.detail.states.length > 0);
    };

    window.addEventListener('loading-state-changed' as any, handleUpdate);
    return () => window.removeEventListener('loading-state-changed' as any, handleUpdate);
  }, []);

  return {
    loadingStates,
    globalLoading,
    addLoading: (id: string, operation: string, message: string) => {
      loadingManager.startLoading(id, operation, message);
    },
    updateProgress: (id: string, progress: number, message?: string) => {
      loadingManager.updateProgress(id, progress, message);
    },
    removeLoading: (id: string) => {
      loadingManager.stopLoading(id);
    },
    isLoading: (id: string) => {
      return loadingManager.isLoading(id);
    },
    getLoadingState: (id: string) => {
      return loadingManager.getLoadingState(id);
    },
    getAllLoadingStates: () => {
      return loadingStates;
    },
    clearAll: () => {
      loadingManager.clearAll();
    }
  };
}

// Loading manager service for non-React contexts
class UniversalLoadingManager {
  private static instance: UniversalLoadingManager;
  private loadingStates = new Map<string, LoadingState>();

  static getInstance() {
    if (!UniversalLoadingManager.instance) {
      UniversalLoadingManager.instance = new UniversalLoadingManager();
    }
    return UniversalLoadingManager.instance;
  }

  startLoading(id: string, operation: string, message: string): void {
    this.loadingStates.set(id, {
      id,
      operation,
      message,
      startTime: Date.now()
    });
    this.broadcastUpdate();
  }

  updateProgress(id: string, progress: number, message?: string): void {
    const state = this.loadingStates.get(id);
    if (state) {
      this.loadingStates.set(id, {
        ...state,
        progress,
        ...(message && { message })
      });
      this.broadcastUpdate();
    }
  }

  stopLoading(id: string): void {
    const state = this.loadingStates.get(id);
    if (state) {
      const duration = Date.now() - state.startTime;
      console.log(`✅ ${state.operation} completed in ${duration}ms`);
    }
    this.loadingStates.delete(id);
    this.broadcastUpdate();
  }

  isLoading(id: string): boolean {
    return this.loadingStates.has(id);
  }

  getLoadingState(id: string): LoadingState | undefined {
    return this.loadingStates.get(id);
  }

  clearAll(): void {
    this.loadingStates.clear();
    this.broadcastUpdate();
  }

  private broadcastUpdate(): void {
    // Dispatch custom event for React components to listen to
    window.dispatchEvent(new CustomEvent('loading-state-changed', {
      detail: { states: Array.from(this.loadingStates.values()) }
    }));
  }
}

export const loadingManager = UniversalLoadingManager.getInstance();
