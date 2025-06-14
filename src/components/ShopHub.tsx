
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
    <div className="flex-1 bg-gray-950">
      <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50">
        <div className="p-6">
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
      <div className="p-6">
        <ProductGrid isLoading={isLoading} products={sortedProducts} />
      </div>
    </div>
  );
};
