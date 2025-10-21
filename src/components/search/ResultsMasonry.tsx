import React from 'react';
import type { SearchResult } from '@/types/search';
import { ContentFeedCard } from './widgets/ContentFeedCard';
import { TrendBurstCard } from './widgets/TrendBurstCard';
import { WellnessCard } from './widgets/WellnessCard';
import { ShoppingCard } from './widgets/ShoppingCard';
import { MediaCard } from './widgets/MediaCard';
import { EntertainmentCard } from './widgets/EntertainmentCard';
import { AssistantCard } from './widgets/AssistantCard';
import { Skeleton } from '@/components/ui/skeleton';

interface ResultsMasonryProps {
  results: SearchResult[];
  isLoading: boolean;
}

export const ResultsMasonry: React.FC<ResultsMasonryProps> = ({ results, isLoading }) => {
  if (isLoading) {
    return (
      <div className="container mx-auto px-4 pb-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="space-y-3">
              <Skeleton className="h-48 w-full rounded-2xl" />
              <Skeleton className="h-4 w-3/4" />
              <Skeleton className="h-4 w-1/2" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="container mx-auto px-4 pb-12">
        <div className="flex flex-col items-center justify-center py-20 text-center">
          <div className="text-6xl mb-4">🔍</div>
          <h3 className="text-2xl font-semibold text-gray-700 dark:text-gray-200 mb-2">
            Start Your Search
          </h3>
          <p className="text-gray-500 dark:text-gray-400 max-w-md">
            Enter a query above to discover AI-powered content, trends, wellness tips, and more.
          </p>
        </div>
      </div>
    );
  }

  const renderWidget = (result: SearchResult) => {
    switch (result.type) {
      case 'content':
        return <ContentFeedCard key={result.id} result={result} />;
      case 'trends':
      case 'news':
        return <TrendBurstCard key={result.id} result={result} />;
      case 'wellness':
        return <WellnessCard key={result.id} result={result} />;
      case 'shopping':
        return <ShoppingCard key={result.id} result={result} />;
      case 'music':
      case 'media':
        return <MediaCard key={result.id} result={result} />;
      case 'entertainment':
      case 'education':
        return <EntertainmentCard key={result.id} result={result} />;
      case 'assistant':
        return <AssistantCard key={result.id} result={result} />;
      default:
        return <ContentFeedCard key={result.id} result={result} />;
    }
  };

  return (
    <div className="container mx-auto px-4 pb-12">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100">
          Search Results ({results.length})
        </h2>
        <p className="text-gray-600 dark:text-gray-400">
          AI-powered content generated just for you
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
        {results.map(renderWidget)}
      </div>
    </div>
  );
};
