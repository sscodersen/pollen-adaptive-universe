/**
 * Content Availability Indicator
 * Phase 15: Visual feedback for content loading sections
 */

import React, { useEffect, useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Loader2, AlertCircle, CheckCircle2, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { contentHealthCheck } from '@/services/contentHealthCheck';

interface ContentAvailabilityIndicatorProps {
  section: string;
  onRetry?: () => void;
  minItems?: number;
}

export const ContentAvailabilityIndicator: React.FC<ContentAvailabilityIndicatorProps> = ({
  section,
  onRetry,
  minItems = 1
}) => {
  const [health, setHealth] = useState<any>(null);
  const [isRetrying, setIsRetrying] = useState(false);

  useEffect(() => {
    const updateHealth = () => {
      const sectionHealth = contentHealthCheck.getSectionHealth(section);
      setHealth(sectionHealth);
    };

    updateHealth();
    const unsubscribe = contentHealthCheck.onHealthChange(updateHealth);

    return unsubscribe;
  }, [section]);

  const handleRetry = async () => {
    if (onRetry) {
      setIsRetrying(true);
      try {
        await onRetry();
      } finally {
        setTimeout(() => setIsRetrying(false), 1000);
      }
    }
  };

  if (!health) {
    return (
      <Card className="border-dashed">
        <CardContent className="flex flex-col items-center justify-center py-12 space-y-4">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
          <div className="text-center">
            <p className="text-sm font-medium">Loading {section} content...</p>
            <p className="text-xs text-muted-foreground mt-1">
              AI is generating personalized content for you
            </p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (health.status === 'failed' && health.itemCount === 0) {
    return (
      <Card className="border-destructive/50 bg-destructive/5">
        <CardContent className="flex flex-col items-center justify-center py-12 space-y-4">
          <AlertCircle className="w-8 h-8 text-destructive" />
          <div className="text-center space-y-2">
            <p className="text-sm font-medium">Content temporarily unavailable</p>
            <p className="text-xs text-muted-foreground">
              {health.errors[0] || 'Unable to generate content at this time'}
            </p>
            {health.recommendations.length > 0 && (
              <p className="text-xs text-muted-foreground italic">
                {health.recommendations[0]}
              </p>
            )}
          </div>
          {onRetry && (
            <Button 
              onClick={handleRetry} 
              variant="outline" 
              size="sm"
              disabled={isRetrying}
            >
              {isRetrying ? (
                <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Retrying...</>
              ) : (
                <><RefreshCw className="w-4 h-4 mr-2" /> Retry</>
              )}
            </Button>
          )}
        </CardContent>
      </Card>
    );
  }

  if (health.status === 'degraded') {
    return (
      <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-3 mb-4">
        <div className="flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-yellow-500" />
          <div className="flex-1">
            <p className="text-sm font-medium">Content updates may be delayed</p>
            <p className="text-xs text-muted-foreground">
              Last updated: {new Date(health.lastUpdate).toLocaleTimeString()}
            </p>
          </div>
        </div>
      </div>
    );
  }

  // Healthy status - show success briefly then hide
  if (health.itemCount >= minItems) {
    return (
      <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-2 mb-4 animate-in fade-in">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-green-500" />
          <p className="text-xs text-muted-foreground">
            {health.itemCount} items loaded • Updated {new Date(health.lastUpdate).toLocaleTimeString()}
          </p>
        </div>
      </div>
    );
  }

  return null;
};
