
import React from 'react';
import { Search, ShoppingBag, TrendingUp, Tag } from 'lucide-react';

interface FilterControlsProps {
  searchQuery: string;
  setSearchQuery: (query: string) => void;
  sortBy: string;
  setSortBy: (sort: string) => void;
  filter: string;
  setFilter: (filter: string) => void;
  categories: string[];
}

export const FilterControls: React.FC<FilterControlsProps> = ({
  searchQuery,
  setSearchQuery,
  sortBy,
  setSortBy,
  filter,
  setFilter,
  categories,
}) => {
  const renderFilterButton = (id: string, label: string, icon: React.ReactNode) => (
    <button
      onClick={() => setFilter(id)}
      className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
        filter === id
          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
          : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 border border-gray-700/30'
      }`}
    >
      {icon}
      <span className="text-sm font-medium">{label}</span>
    </button>
  );

  return (
    <>
      {/* Search and Sort */}
      <div className="flex items-center justify-between space-x-4 mb-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search products, categories, features..."
            className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg pl-10 pr-4 py-3 text-white placeholder-gray-400 focus:border-cyan-500/50 focus:outline-none"
          />
        </div>

        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-2 text-white text-sm focus:outline-none focus:border-cyan-500/50"
        >
          <option value="significance">Sort by Relevance</option>
          <option value="price">Sort by Price</option>
          <option value="rating">Sort by Rating</option>
          <option value="discount">Sort by Discount</option>
        </select>
      </div>

      {/* Category Filters */}
      <div className="flex space-x-2 overflow-x-auto pb-2">
        {renderFilterButton('all', 'All Products', <ShoppingBag className="w-4 h-4" />)}
        {renderFilterButton('trending', 'Trending', <TrendingUp className="w-4 h-4" />)}
        {renderFilterButton('discounted', 'On Sale', <Tag className="w-4 h-4" />)}
        {categories.map((category) => (
          <button
            key={category}
            onClick={() => setFilter(category)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
              filter === category
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 border border-gray-700/30'
            }`}
          >
            <span className="text-sm font-medium">{category}</span>
          </button>
        ))}
      </div>
    </>
  );
};
