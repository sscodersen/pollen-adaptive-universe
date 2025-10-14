/**
 * In-App Error Notifications and Feedback Prompts
 * Phase 15: Encourage user feedback on errors
 */

import React, { useEffect, useState } from 'react';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { AlertTriangle, XCircle, MessageCircle, X } from 'lucide-react';
import { loggingService } from '@/services/loggingService';

export interface ErrorNotificationProps {
  error: {
    title: string;
    message: string;
    severity?: 'warning' | 'error' | 'critical';
    feature?: string;
  };
  onFeedback?: () => void;
  onDismiss?: () => void;
  autoShowFeedback?: boolean;
}

export const ErrorNotification: React.FC<ErrorNotificationProps> = ({
  error,
  onFeedback,
  onDismiss,
  autoShowFeedback = true
}) => {
  const [showFeedbackPrompt, setShowFeedbackPrompt] = useState(false);

  useEffect(() => {
    if (autoShowFeedback && error.severity === 'critical') {
      // Auto-show feedback form for critical errors
      setTimeout(() => setShowFeedbackPrompt(true), 2000);
    }
  }, [autoShowFeedback, error.severity]);

  const handleFeedback = () => {
    loggingService.logUserInteraction('click_error_feedback', 'error_notification', {
      errorTitle: error.title,
      feature: error.feature
    });
    setShowFeedbackPrompt(false);
    if (onFeedback) {
      onFeedback();
    }
  };

  const handleDismiss = () => {
    loggingService.logUserInteraction('dismiss_error', 'error_notification', {
      errorTitle: error.title
    });
    if (onDismiss) {
      onDismiss();
    }
  };

  const getIcon = () => {
    switch (error.severity) {
      case 'critical':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'error':
        return <AlertTriangle className="w-5 h-5 text-orange-500" />;
      default:
        return <AlertTriangle className="w-5 h-5 text-yellow-500" />;
    }
  };

  const getVariant = (): 'default' | 'destructive' => {
    return error.severity === 'critical' || error.severity === 'error' ? 'destructive' : 'default';
  };

  return (
    <div className="space-y-3">
      <Alert variant={getVariant()} className="relative">
        {onDismiss && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleDismiss}
            className="absolute top-2 right-2 w-6 h-6 p-0"
          >
            <X className="w-4 h-4" />
          </Button>
        )}
        <div className="flex items-start gap-3">
          {getIcon()}
          <div className="flex-1 space-y-1">
            <AlertTitle>{error.title}</AlertTitle>
            <AlertDescription>{error.message}</AlertDescription>
          </div>
        </div>
      </Alert>

      {showFeedbackPrompt && (
        <Alert className="bg-primary/5 border-primary/20">
          <MessageCircle className="w-5 h-5 text-primary" />
          <AlertTitle>Help us fix this issue</AlertTitle>
          <AlertDescription className="space-y-3">
            <p>We detected an error. Your feedback can help us resolve it faster.</p>
            <div className="flex gap-2">
              <Button onClick={handleFeedback} size="sm" className="flex-1">
                Share Feedback
              </Button>
              <Button 
                onClick={() => setShowFeedbackPrompt(false)} 
                variant="outline" 
                size="sm"
              >
                Not now
              </Button>
            </div>
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
};

// Global error notification hook
export function useErrorNotification() {
  const [error, setError] = useState<ErrorNotificationProps['error'] | null>(null);

  const showError = (errorData: ErrorNotificationProps['error']) => {
    setError(errorData);
    loggingService.logError({
      type: 'user_facing_error',
      message: errorData.message,
      metadata: { title: errorData.title, feature: errorData.feature, severity: errorData.severity }
    });
  };

  const clearError = () => setError(null);

  return { error, showError, clearError };
}
