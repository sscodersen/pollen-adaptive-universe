import React from 'react';
import { Search, Filter } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface AppStoreFiltersProps {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  sortBy: string;
  setSortBy: (sort: string) => void;
  filter: string;
  setFilter: (filter: string) => void;
  categories: string[];
}

export const AppStoreFilters: React.FC<AppStoreFiltersProps> = ({
  searchQuery,
  setSearchQuery,
  sortBy,
  setSortBy,
  filter,
  setFilter,
  categories
}) => {
  return (
    <div className="mb-6 space-y-4">
      <div className="flex flex-col lg:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <Input
            placeholder="Search apps..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 bg-gray-900/50 border-gray-700 text-white placeholder-gray-400"
          />
        </div>
        
        <div className="flex gap-3">
          <Select value={sortBy} onValueChange={setSortBy}>
            <SelectTrigger className="w-40 bg-gray-900/50 border-gray-700">
              <SelectValue placeholder="Sort by" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="significance">Relevance</SelectItem>
              <SelectItem value="rating">Rating</SelectItem>
              <SelectItem value="downloads">Downloads</SelectItem>
              <SelectItem value="price">Price</SelectItem>
            </SelectContent>
          </Select>

          <Select value={filter} onValueChange={setFilter}>
            <SelectTrigger className="w-40 bg-gray-900/50 border-gray-700">
              <Filter className="w-4 h-4 mr-2" />
              <SelectValue placeholder="Filter" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Apps</SelectItem>
              <SelectItem value="featured">Featured</SelectItem>
              <SelectItem value="trending">Trending</SelectItem>
              <SelectItem value="free">Free</SelectItem>
              {categories.map((category) => (
                <SelectItem key={category} value={category}>
                  {category}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  );
};