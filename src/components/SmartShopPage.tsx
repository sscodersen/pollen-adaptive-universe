
import React, { useState, useEffect, useCallback } from 'react';
import { contentOrchestrator } from '../services/contentOrchestrator';
import { ShopContent } from '../services/unifiedContentEngine';
import { Product } from '../types/shop';
import { ShopHeader } from './shop/ShopHeader';
import { FilterControls } from './shop/FilterControls';
import { ProductGrid } from './shop/ProductGrid';
import { enhancedTrendEngine } from '../services/enhancedTrendEngine';
import { reRankProducts } from '../services/shopReRanker';
import { ExternalLink, Store, TrendingUp } from 'lucide-react';

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
      } as const;

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
      
      // Blacklist: drop unwanted products
      const { filterBlacklisted, isBlacklistedText } = await import('../lib/blacklist');
      const safeProducts = convertedProducts.filter(p => 
        !isBlacklistedText(p.name) && 
        !isBlacklistedText(p.description) && 
        !p.tags?.some(t => isBlacklistedText(t))
      );
      
      const ranked = reRankProducts(safeProducts, enhancedTrendEngine.getTrends());
      setProducts(ranked);
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
      product.tags?.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
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
    <div className="flex-1 bg-gray-950 min-h-0 flex flex-col">
      {/* App Store Style Header */}
      <div className="bg-gradient-to-r from-indigo-600/20 to-purple-600/20 border-b border-gray-800/50 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl border border-blue-500/30">
                <Store className="w-8 h-8 text-blue-400" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white mb-2">Smart Marketplace</h1>
                <p className="text-gray-400">Discover curated products from external sources</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-cyan-400 bg-cyan-500/10 px-3 py-2 rounded-lg border border-cyan-500/30">
              <ExternalLink className="w-4 h-4" />
              <span>External Links</span>
            </div>
          </div>
          
          {/* Featured Stats */}
          <div className="grid grid-cols-3 gap-6 mt-6">
            <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800/50">
              <div className="flex items-center space-x-2 text-green-400 mb-2">
                <TrendingUp className="w-5 h-5" />
                <span className="font-semibold">Trending</span>
              </div>
              <p className="text-2xl font-bold text-white">{filteredProducts.filter(p => p.trending).length}</p>
              <p className="text-gray-400 text-sm">Hot products</p>
            </div>
            <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800/50">
              <div className="flex items-center space-x-2 text-purple-400 mb-2">
                <Store className="w-5 h-5" />
                <span className="font-semibold">Categories</span>
              </div>
              <p className="text-2xl font-bold text-white">{categories.length}</p>
              <p className="text-gray-400 text-sm">Available now</p>
            </div>
            <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800/50">
              <div className="flex items-center space-x-2 text-orange-400 mb-2">
                <ExternalLink className="w-5 h-5" />
                <span className="font-semibold">External</span>
              </div>
              <p className="text-2xl font-bold text-white">100%</p>
              <p className="text-gray-400 text-sm">Real sources</p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex-1 p-6">
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
    </div>
  );
};
