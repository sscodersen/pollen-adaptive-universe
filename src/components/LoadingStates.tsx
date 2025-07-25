import React from 'react';
import { Loader2, Wifi, WifiOff } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  text?: string;
  className?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  size = 'md', 
  text,
  className = '' 
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  };

  return (
    <div className={`flex items-center justify-center gap-2 ${className}`}>
      <Loader2 className={`${sizeClasses[size]} animate-spin text-primary`} />
      {text && <span className="text-muted-foreground text-sm">{text}</span>}
    </div>
  );
};

interface ContentLoadingProps {
  title?: string;
  description?: string;
  rows?: number;
}

export const ContentLoading: React.FC<ContentLoadingProps> = ({ 
  title = "Loading content...",
  description = "Please wait while we fetch the latest information",
  rows = 3
}) => {
  return (
    <div className="space-y-4 p-6">
      <div className="text-center space-y-2">
        <LoadingSpinner size="lg" />
        <h3 className="text-lg font-semibold text-foreground">{title}</h3>
        <p className="text-muted-foreground text-sm">{description}</p>
      </div>
      
      {/* Skeleton cards */}
      <div className="space-y-4 mt-8">
        {Array.from({ length: rows }).map((_, i) => (
          <Card key={i} className="bg-surface-secondary border-border">
            <CardContent className="p-4">
              <div className="animate-pulse space-y-3">
                <div className="h-4 bg-surface-tertiary rounded w-3/4"></div>
                <div className="space-y-2">
                  <div className="h-3 bg-surface-tertiary rounded"></div>
                  <div className="h-3 bg-surface-tertiary rounded w-5/6"></div>
                </div>
                <div className="flex space-x-2">
                  <div className="h-6 bg-surface-tertiary rounded w-16"></div>
                  <div className="h-6 bg-surface-tertiary rounded w-20"></div>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

interface NetworkErrorProps {
  onRetry?: () => void;
  title?: string;
  description?: string;
}

export const NetworkError: React.FC<NetworkErrorProps> = ({
  onRetry,
  title = "Connection Error",
  description = "Unable to connect to the server. Please check your internet connection."
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center space-y-4">
      <div className="w-16 h-16 bg-status-error/20 rounded-full flex items-center justify-center">
        <WifiOff className="w-8 h-8 text-red-400" />
      </div>
      
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-foreground">{title}</h3>
        <p className="text-muted-foreground text-sm max-w-md">{description}</p>
      </div>
      
      {onRetry && (
        <Button onClick={onRetry} variant="outline" className="mt-4">
          <Wifi className="w-4 h-4 mr-2" />
          Try Again
        </Button>
      )}
    </div>
  );
};

interface EmptyStateProps {
  title: string;
  description: string;
  action?: React.ReactNode;
  icon?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  description,
  action,
  icon
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center space-y-4">
      {icon && (
        <div className="w-16 h-16 bg-surface-tertiary rounded-full flex items-center justify-center">
          {icon}
        </div>
      )}
      
      <div className="space-y-2">
        <h3 className="text-lg font-semibold text-foreground">{title}</h3>
        <p className="text-muted-foreground text-sm max-w-md">{description}</p>
      </div>
      
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
};

// Page-level loading wrapper
interface PageLoadingProps {
  children: React.ReactNode;
  isLoading: boolean;
  error?: Error | null;
  onRetry?: () => void;
  loadingComponent?: React.ReactNode;
  errorComponent?: React.ReactNode;
}

export const PageLoading: React.FC<PageLoadingProps> = ({
  children,
  isLoading,
  error,
  onRetry,
  loadingComponent,
  errorComponent
}) => {
  if (error) {
    return errorComponent || <NetworkError onRetry={onRetry} />;
  }
  
  if (isLoading) {
    return loadingComponent || <ContentLoading />;
  }
  
  return <>{children}</>;
};