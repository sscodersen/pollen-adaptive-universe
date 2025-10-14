/**
 * Global Loading Indicator Component
 * Phase 15: Visual feedback for loading states
 */

import React from 'react';
import { useLoadingManager } from '@/services/universalLoadingManager';
import { Loader2 } from 'lucide-react';
import { Progress } from '@/components/ui/progress';

export const GlobalLoadingIndicator: React.FC = () => {
  const { getAllLoadingStates, globalLoading } = useLoadingManager();
  const loadingStates = getAllLoadingStates();

  if (!globalLoading || loadingStates.length === 0) {
    return null;
  }

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 max-w-sm">
      {loadingStates.slice(0, 3).map((state) => (
        <div
          key={state.id}
          className="bg-surface-secondary/95 backdrop-blur-sm border border-border rounded-lg p-4 shadow-lg animate-in slide-in-from-right"
        >
          <div className="flex items-center gap-3">
            <Loader2 className="w-4 h-4 animate-spin text-primary" />
            <div className="flex-1">
              <p className="text-sm font-medium">{state.operation}</p>
              <p className="text-xs text-muted-foreground">{state.message}</p>
              {state.progress !== undefined && (
                <Progress value={state.progress} className="mt-2 h-1" />
              )}
            </div>
          </div>
        </div>
      ))}
      {loadingStates.length > 3 && (
        <div className="text-xs text-center text-muted-foreground">
          +{loadingStates.length - 3} more operations
        </div>
      )}
    </div>
  );
};
