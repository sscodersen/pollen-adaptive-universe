
import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { pollenAI } from '../services/pollenAI';
import { Product } from '../types';

export const useProducts = () => {
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('significance');

  // Query for all products (for when there's no search)
  const { data: allProducts = [], isLoading: isLoadingAll, refetch: refetchAll } = useQuery<Product[]>({
    queryKey: ['products', 'all'],
    queryFn: () => pollenAI.getRankedProducts(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Query for searched products, only enabled when searchQuery is not empty
  const { data: searchedProducts = [], isLoading: isLoadingSearch, refetch: refetchSearch } = useQuery<Product[]>({
    queryKey: ['products', 'search', searchQuery],
    queryFn: () => pollenAI.searchProducts(searchQuery),
    staleTime: 5 * 60 * 1000,
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

  const sortedProducts = useMemo(() => {
    // If there's a search query, we trust the backend's relevance sorting
    // unless the user explicitly chooses another sort option.
    if (searchQuery && sortBy === 'significance') {
      return filteredProducts;
    }
    
    return [...filteredProducts].sort((a, b) => {
      if (sortBy === 'price') return parseFloat(a.price.replace(/[^0-9.]/g, '')) - parseFloat(b.price.replace(/[^0-9.]/g, ''));
      if (sortBy === 'rating') return b.rating - a.rating;
      if (sortBy === 'discount') return (b.discount || 0) - (a.discount || 0);
      // default to significance
      return b.significance - a.significance;
    });
  }, [filteredProducts, sortBy, searchQuery]);

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
