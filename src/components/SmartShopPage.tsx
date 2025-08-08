
import React, { useState, useEffect, useCallback } from 'react';
import { contentOrchestrator } from '../services/contentOrchestrator';
import { ShopContent } from '../services/unifiedContentEngine';
import { Product } from '../types/shop';
import { ShopHeader } from './shop/ShopHeader';
import { FilterControls } from './shop/FilterControls';
import { ProductGrid } from './shop/ProductGrid';

// Removed hardcoded templates - now using AI-generated products

export const SmartShopPage = () => {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('significance');
  const [filter, setFilter] = useState('all');

  // Using unified content engine now

  const loadProducts = useCallback(async () => {
    setLoading(true);
    try {
      const strategy = {
        diversity: 0.8,
        freshness: 0.7,
        personalization: 0.4,
        qualityThreshold: 6.5,
        trendingBoost: 1.4
      };
      
      const { content: shopContent } = await contentOrchestrator.generateContent({ type: 'shop', count: 16, strategy });
      const convertedProducts: Product[] = shopContent.map((item) => {
        const shopItem = item as ShopContent;
        return {
          id: shopItem.id,
          name: shopItem.name,
          description: shopItem.description,
          price: shopItem.price,
          originalPrice: shopItem.originalPrice,
          discount: shopItem.discount,
          rating: shopItem.rating,
          reviews: shopItem.reviews,
          category: shopItem.category,
          brand: shopItem.brand,
          tags: shopItem.tags,
          link: shopItem.link,
          inStock: shopItem.inStock,
          trending: shopItem.trending,
          significance: shopItem.significance,
          features: shopItem.features,
          seller: shopItem.seller,
          views: shopItem.views,
          rank: shopItem.rank || 0,
          quality: shopItem.quality,
          impact: shopItem.impact === 'critical' ? 'premium' : shopItem.impact as 'low' | 'medium' | 'high' | 'premium'
        };
      });
      
      setProducts(convertedProducts);
    } catch (error) {
      console.error('Failed to load products:', error);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    loadProducts();
    const interval = setInterval(loadProducts, 45000);
    return () => clearInterval(interval);
  }, [loadProducts]);

  const categories = [...new Set(products.map(p => p.category))];

  const filteredProducts = products.filter(product => {
    const matchesSearch = !searchQuery || 
      product.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.category.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.brand.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    const matchesFilter = filter === 'all' || 
      filter === 'trending' && product.trending ||
      filter === 'discounted' && (product.discount || 0) > 0 ||
      product.category === filter;
    
    return matchesSearch && matchesFilter;
  }).sort((a, b) => {
    switch (sortBy) {
      case 'price':
        return parseFloat(a.price.replace('$', '')) - parseFloat(b.price.replace('$', ''));
      case 'rating':
        return b.rating - a.rating;
      case 'discount':
        return (b.discount || 0) - (a.discount || 0);
      default:
        return b.significance - a.significance;
    }
  });

  return (
    <div className="flex-1 bg-gray-950 min-h-0 flex flex-col p-6">
      <ShopHeader loading={loading} onRefresh={loadProducts} />
      
      <FilterControls
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        sortBy={sortBy}
        setSortBy={setSortBy}
        filter={filter}
        setFilter={setFilter}
        categories={categories}
      />

      <ProductGrid isLoading={loading} products={filteredProducts} />
    </div>
  );
};
