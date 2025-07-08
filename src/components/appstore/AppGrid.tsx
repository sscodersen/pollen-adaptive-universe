import React from 'react';
import { AppCard } from './AppCard';
import { App } from '../AppStorePage';
import { Skeleton } from '@/components/ui/skeleton';

interface AppGridProps {
  isLoading: boolean;
  apps: App[];
}

export const AppGrid: React.FC<AppGridProps> = ({ isLoading, apps }) => {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {Array.from({ length: 12 }).map((_, index) => (
          <div key={index} className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6">
            <Skeleton className="w-full h-48 rounded-lg mb-4" />
            <Skeleton className="h-6 w-3/4 mb-2" />
            <Skeleton className="h-4 w-full mb-2" />
            <Skeleton className="h-4 w-2/3 mb-4" />
            <div className="flex justify-between items-center">
              <Skeleton className="h-8 w-20" />
              <Skeleton className="h-8 w-24" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (apps.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-400 text-lg">No apps found matching your criteria.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
      {apps.map((app) => (
        <AppCard key={app.id} app={app} />
      ))}
    </div>
  );
};