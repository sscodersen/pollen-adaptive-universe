import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { pollenAI } from '../services/pollenAI';
import { Product } from '../types';
import { rankItems } from '../services/generalRanker';

export const useProducts = () => {
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('significance');

  // Query for all products (for when there's no search)
  const { data: allProducts = [], isLoading: isLoadingAll, refetch: refetchAll } = useQuery<Product[]>({
    queryKey: ['products', 'all'],
    queryFn: () => pollenAI.getRankedProducts(),
    staleTime: 60 * 1000, // 1 minute
    refetchInterval: 15000, // Refresh every 15 seconds for continuous generation
  });

  // Query for searched products, only enabled when searchQuery is not empty
  const { data: searchedProducts = [], isLoading: isLoadingSearch, refetch: refetchSearch } = useQuery<Product[]>({
    queryKey: ['products', 'search', searchQuery],
    queryFn: () => pollenAI.searchProducts(searchQuery),
    stleTime: 5 * 60 * 1000,
    enabled: !!searchQuery,
  });

  const isLoading = isLoadingAll || isLoadingSearch;
  const refetch = () => {
    refetchAll();
    if (searchQuery) {
      refetchSearch();
    }
  };

  const products = searchQuery ? searchedProducts : allProducts;

  const filteredProducts = useMemo(() => {
    // Search is now handled by the backend, so we just handle local filters
    return products.filter(product => {
      if (filter === 'trending') return product.trending;
      if (filter === 'discounted') return !!(product.discount && product.discount > 0);
      if (filter === 'all') return true;
      return product.category === filter;
    });
  }, [products, filter]);

  // Use the new generalRanker for all sorted products
  const sortedProducts = useMemo(() => {
    return rankItems(filteredProducts, { type: "shop", sortBy: sortBy });
  }, [filteredProducts, sortBy]);

  const categories = useMemo(() => {
    // Use allProducts to populate categories so they don't change with search
    return [...new Set(allProducts.map(p => p.category))];
  }, [allProducts]);

  return {
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
  };
};
