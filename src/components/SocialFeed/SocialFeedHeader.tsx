import React, { memo } from 'react';
import { Globe, Search } from 'lucide-react';
import { Input } from "@/components/ui/input";

interface SocialFeedHeaderProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  postsCount: number;
  isLoading: boolean;
}

export const SocialFeedHeader = memo(({ 
  searchQuery, 
  onSearchChange, 
  postsCount, 
  isLoading 
}: SocialFeedHeaderProps) => {
  return (
    <div className="sticky top-0 z-10 bg-card backdrop-blur-sm border-b border-border">
      <div className="p-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
              <Globe className="w-8 h-8 text-cyan-400" />
              Global Feed
            </h1>
            <p className="text-gray-400">
              Real-time insights • Ranked content • AI-curated quality
            </p>
          </div>
          <div className="flex items-center space-x-3">
            <div className="px-4 py-2 bg-green-500/10 text-green-400 rounded-full text-sm font-medium border border-green-500/20 flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span>Live Feed</span>
            </div>
            <div className="text-sm text-gray-500">
              {isLoading ? 'Loading...' : `${postsCount} posts`}
            </div>
          </div>
        </div>
        
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <Input
            type="text"
            placeholder="Search posts, trends, or users..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="pl-10 bg-muted/50 border-border focus:border-primary"
          />
        </div>
      </div>
    </div>
  );
});

SocialFeedHeader.displayName = 'SocialFeedHeader';