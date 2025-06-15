
import React from 'react';
import { useProducts } from '../hooks/useProducts';
import { ShopHeader } from './shop/ShopHeader';
import { FilterControls } from './shop/FilterControls';
import { ProductGrid } from './shop/ProductGrid';

export const ShopHub = () => {
  const {
    isLoading,
    refetch,
    sortedProducts,
    categories,
    filter,
    setFilter,
    searchQuery,
    setSearchQuery,
    sortBy,
    setSortBy,
  } = useProducts();

  return (
    <div className="flex-1 min-h-screen bg-gradient-to-br from-gray-950 to-gray-900 relative">
      {/* Hero Header */}
      <div className="w-full bg-gradient-to-br from-cyan-800/60 via-cyan-900/80 to-purple-950/90 shadow-lg border-b border-cyan-800/30 animate-fade-in">
        <div className="max-w-5xl mx-auto px-6 py-12 flex flex-col md:flex-row items-center md:justify-between">
          <div className="w-full md:w-2/3 mb-6 md:mb-0 animate-fade-in">
            <h2 className="text-4xl md:text-5xl font-extrabold text-white mb-4 leading-tight drop-shadow-sm">
              Smart Shopping Hub
            </h2>
            <p className="text-lg md:text-xl text-cyan-100 mb-5 drop-shadow-sm">
              Discover <span className="font-semibold text-cyan-300">AI-curated</span> products from trusted brands, verified sellers, and top categories.
              <br />
              <span className="text-purple-200/90">All recommendations are handpicked by Adaptive Intelligence for real-time trends and relevance.</span>
            </p>
            <div className="flex flex-wrap gap-3">
              {/* Example feature highlights */}
              <span className="bg-cyan-600/20 text-cyan-200 font-semibold rounded-full px-4 py-1 border border-cyan-400/40 text-xs">
                Trending: Oura Ring, Smart Home, AI Gear
              </span>
              <span className="bg-purple-700/20 text-purple-200 rounded-full px-4 py-1 border border-purple-500/30 text-xs">
                Verified Seller Links
              </span>
              <span className="bg-green-600/15 text-green-300 rounded-full px-4 py-1 border border-green-400/30 text-xs">
                Real-Time Discounts
              </span>
            </div>
          </div>
          {/* Optional illustration */}
          <div className="w-full md:w-1/3 flex justify-center items-center animate-fade-in">
            <img
              src="https://images.unsplash.com/photo-1531297484001-80022131f5a1?auto=format&fit=crop&w=350&q=80"
              alt="Smart Shopping"
              className="rounded-xl shadow-lg border-2 border-cyan-700/30 object-cover w-64 h-40 md:w-72 md:h-44"
            />
          </div>
        </div>
      </div>
      {/* Sticky Shop Filter Bar */}
      <div className="sticky top-0 z-20 bg-gray-900/95 backdrop-blur-md border-b border-cyan-800/20 shadow-sm">
        <div className="max-w-5xl mx-auto px-6 pt-4 pb-2">
          <ShopHeader loading={isLoading} onRefresh={() => refetch()} />
          <FilterControls
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            sortBy={sortBy}
            setSortBy={setSortBy}
            filter={filter}
            setFilter={setFilter}
            categories={categories}
          />
        </div>
      </div>
      {/* Product Grid */}
      <div className="max-w-5xl mx-auto px-6 py-8 animate-fade-in">
        <ProductGrid isLoading={isLoading} products={sortedProducts} />
      </div>
    </div>
  );
};

