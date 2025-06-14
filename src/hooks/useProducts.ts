
import { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { pollenAI } from '../services/pollenAI';
import { Product } from '../types';

export const useProducts = () => {
  const { data: products = [], isLoading, refetch } = useQuery<Product[]>({
    queryKey: ['products'],
    queryFn: () => pollenAI.getRankedProducts(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('significance');

  const filteredProducts = useMemo(() => {
    return products.filter(product => {
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return product.name.toLowerCase().includes(query) ||
               product.description.toLowerCase().includes(query) ||
               product.category.toLowerCase().includes(query) ||
               product.tags.some(tag => tag.toLowerCase().includes(query));
      }
      if (filter === 'trending') return product.trending;
      if (filter === 'discounted') return !!(product.discount && product.discount > 0);
      if (filter === 'all') return true;
      return product.category === filter;
    });
  }, [products, searchQuery, filter]);

  const sortedProducts = useMemo(() => {
    return [...filteredProducts].sort((a, b) => {
      if (sortBy === 'price') return parseFloat(a.price.replace(/[^0-9.]/g, '')) - parseFloat(b.price.replace(/[^0-9.]/g, ''));
      if (sortBy === 'rating') return b.rating - a.rating;
      if (sortBy === 'discount') return (b.discount || 0) - (a.discount || 0);
      // default to significance
      return b.significance - a.significance;
    });
  }, [filteredProducts, sortBy]);

  const categories = useMemo(() => {
    return [...new Set(products.map(p => p.category))];
  }, [products]);

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
