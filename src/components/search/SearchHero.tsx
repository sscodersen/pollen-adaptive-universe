import React, { useState } from 'react';
import { Search, Sparkles } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

interface SearchHeroProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  onSearch: (query: string) => void;
  isLoading: boolean;
}

export const SearchHero: React.FC<SearchHeroProps> = ({
  searchQuery,
  onSearchChange,
  onSearch,
  isLoading
}) => {
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      onSearch(searchQuery);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[40vh] px-4 py-8">
      <div className="w-full max-w-3xl space-y-6 text-center">
        <div className="space-y-3">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-full border border-gray-200/50 dark:border-gray-700/50 shadow-lg">
            <Sparkles className="w-5 h-5 text-purple-500" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-200">
              Pollen AI Universe
            </span>
          </div>
          
          <h1 className="text-4xl md:text-6xl font-bold bg-gradient-to-r from-purple-600 via-blue-600 to-pink-600 bg-clip-text text-transparent">
            Ask Pollen Anything
          </h1>
          
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Search for anything - get AI-powered feed posts, news, trends, wellness tips, products, music, videos, and more.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="relative">
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-600 via-blue-600 to-pink-600 rounded-full blur opacity-30 group-hover:opacity-50 transition duration-300"></div>
            
            <div className="relative flex items-center gap-2 bg-white dark:bg-gray-800 rounded-full shadow-2xl p-2">
              <Search className="w-6 h-6 text-gray-400 ml-4" />
              <Input
                type="text"
                value={searchQuery}
                onChange={(e) => onSearchChange(e.target.value)}
                placeholder="Search for news, trends, wellness, products, music, smart home control..."
                className="flex-1 border-0 bg-transparent text-lg focus-visible:ring-0 focus-visible:ring-offset-0"
              />
              <Button
                type="submit"
                disabled={isLoading || !searchQuery.trim()}
                className="rounded-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 transition-all duration-300"
              >
                {isLoading ? (
                  <>
                    <Sparkles className="w-5 h-5 animate-spin mr-2" />
                    Generating
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5 mr-2" />
                    Search
                  </>
                )}
              </Button>
            </div>
          </div>
        </form>

        <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-gray-500 dark:text-gray-400">
          <span className="px-3 py-1 bg-white/60 dark:bg-gray-800/60 backdrop-blur rounded-full">
            🎯 Trending Topics
          </span>
          <span className="px-3 py-1 bg-white/60 dark:bg-gray-800/60 backdrop-blur rounded-full">
            💪 Wellness Tips
          </span>
          <span className="px-3 py-1 bg-white/60 dark:bg-gray-800/60 backdrop-blur rounded-full">
            🎵 Music & Audio
          </span>
          <span className="px-3 py-1 bg-white/60 dark:bg-gray-800/60 backdrop-blur rounded-full">
            🛍️ Smart Shopping
          </span>
          <span className="px-3 py-1 bg-white/60 dark:bg-gray-800/60 backdrop-blur rounded-full">
            🏠 Smart Home
          </span>
          <span className="px-3 py-1 bg-white/60 dark:bg-gray-800/60 backdrop-blur rounded-full">
            🤖 AI Tools
          </span>
        </div>
      </div>
    </div>
  );
};
