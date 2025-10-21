import React from 'react';
import { SearchHero } from './SearchHero';
import { ResultsMasonry } from './ResultsMasonry';
import type { SearchResult } from '@/types/search';

interface SearchCanvasProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  onSearch: (query: string) => void;
  results: SearchResult[];
  isLoading: boolean;
}

export const SearchCanvas: React.FC<SearchCanvasProps> = ({
  searchQuery,
  onSearchChange,
  onSearch,
  results,
  isLoading
}) => {
  return (
    <div className="relative min-h-screen w-full overflow-x-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 dark:from-gray-900 dark:via-purple-900/20 dark:to-blue-900/20" />
      
      <div className="relative z-10">
        <SearchHero
          searchQuery={searchQuery}
          onSearchChange={onSearchChange}
          onSearch={onSearch}
          isLoading={isLoading}
        />
        
        <ResultsMasonry results={results} isLoading={isLoading} />
      </div>
    </div>
  );
};
