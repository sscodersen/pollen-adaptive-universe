import React, { useState, useEffect } from 'react';
import { Search, ThumbsUp, ThumbsDown, Sparkles, TrendingUp, Star, ShoppingCart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { contentOrchestrator } from '@/services/contentOrchestrator';
import { ShopContent } from '@/services/unifiedContentEngine';
import { personalizationEngine } from '@/services/personalizationEngine';
import { votingService } from '@/services/votingService';
import { shopEnhancements } from '@/services/shopEnhancements';
import { ContentAvailabilityIndicator } from '@/components/ContentAvailabilityIndicator';

interface EnhancedShopProps {
  onNavigate: (screen: string) => void;
}

export function EnhancedShop({ onNavigate }: EnhancedShopProps) {
  const [products, setProducts] = useState<ShopContent[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  const categories = ['all', 'electronics', 'apps', 'wellness', 'productivity'];

  useEffect(() => {
    loadProducts();
  }, [selectedCategory]);

  const loadProducts = async () => {
    setLoading(true);
    try {
      const query = selectedCategory === 'all' 
        ? 'trending products and innovative apps'
        : `${selectedCategory} products and apps`;

      const response = await contentOrchestrator.generateContent<ShopContent>({
        type: 'shop',
        query,
        count: 20,
        strategy: {
          diversity: 0.9,
          freshness: 0.85,
          personalization: 0.9,
          qualityThreshold: 8.0,
          trendingBoost: 0.75
        }
      });

      const personalized = await personalizationEngine.generateRecommendations(
        response.content,
        20
      );

      const enriched = votingService.enrichPostsWithVotes(personalized) as ShopContent[];
      
      // Update price history for wishlisted items
      const wishlistItems = shopEnhancements.getWishlist();
      enriched.forEach(product => {
        const wishlisted = wishlistItems.find(item => item.product.id === product.id);
        if (wishlisted && product.price) {
          const priceNum = parseFloat(product.price.replace('$', ''));
          shopEnhancements.updatePriceHistory(product.id, priceNum);
        }
      });
      
      setProducts(enriched);
    } catch (error) {
      console.error('Failed to load products:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleVote = async (productId: string, voteType: 'up' | 'down') => {
    const product = products.find(p => p.id === productId);
    if (!product || !product.votes) return;

    const newVotes = await votingService.vote(productId, voteType, {
      upvotes: product.votes.upvotes,
      downvotes: product.votes.downvotes,
      score: product.votes.score
    });

    setProducts(prev => prev.map(p => 
      p.id === productId ? { ...p, votes: { ...newVotes, userVote: votingService.getUserVote(productId) } } : p
    ));

    personalizationEngine.trackBehavior({
      action: 'like',
      contentId: productId,
      contentType: 'shop',
      metadata: { vote: voteType, category: selectedCategory }
    });
  };

  const filteredProducts = searchQuery
    ? products.filter(p => 
        p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.description.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : products;

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-500 rounded-2xl">
              <ShoppingCart className="w-8 h-8 text-white" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                AI Smart Shop
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Discover products with AI-powered recommendations
              </p>
            </div>
          </div>

          {/* Search */}
          <div className="relative max-w-2xl">
            <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <Input
              type="text"
              placeholder="Search products..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-12 h-12"
            />
          </div>
        </div>

        {/* Categories */}
        <div className="flex gap-2 mb-8 overflow-x-auto scrollbar-thin pb-2">
          {categories.map(cat => (
            <Button
              key={cat}
              variant={selectedCategory === cat ? 'default' : 'outline'}
              onClick={() => setSelectedCategory(cat)}
              className="whitespace-nowrap capitalize"
            >
              {cat}
            </Button>
          ))}
        </div>

        {/* Products Grid */}
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="animate-pulse bg-white/60 dark:bg-gray-800/60 rounded-2xl p-6 h-80" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredProducts.map(product => (
              <div
                key={product.id}
                className="group bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm rounded-2xl p-6 border border-gray-200 dark:border-gray-700 hover:shadow-2xl hover:scale-[1.02] transition-all duration-300"
              >
                {/* Product Header */}
                <div className="flex items-start justify-between mb-4">
                  <div className="flex-1">
                    <h3 className="font-bold text-lg text-gray-900 dark:text-white mb-2 line-clamp-2">
                      {product.name}
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                      {product.brand}
                    </p>
                  </div>
                  {product.trending && (
                    <span className="px-3 py-1 bg-gradient-to-r from-orange-500/20 to-red-500/20 rounded-full flex items-center gap-1">
                      <TrendingUp className="w-3 h-3 text-orange-600" />
                      <span className="text-xs font-medium text-orange-600">Trending</span>
                    </span>
                  )}
                </div>

                {/* AI Description */}
                <div className="mb-4 p-3 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl">
                  <div className="flex items-center gap-2 mb-2">
                    <Sparkles className="w-4 h-4 text-purple-600" />
                    <span className="text-xs font-medium text-purple-600">AI Description</span>
                  </div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-3">
                    {product.description}
                  </p>
                </div>

                {/* Price & Rating */}
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <div className="flex items-baseline gap-2">
                      <span className="text-2xl font-bold text-gray-900 dark:text-white">
                        {product.price}
                      </span>
                      {product.originalPrice && (
                        <span className="text-sm text-gray-500 line-through">
                          {product.originalPrice}
                        </span>
                      )}
                    </div>
                    {product.discount > 0 && (
                      <span className="text-xs font-medium text-green-600">
                        {product.discount}% OFF
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
                    <span className="text-sm font-medium">{product.rating}</span>
                    <span className="text-xs text-gray-500">({product.reviews})</span>
                  </div>
                </div>

                {/* Voting */}
                {product.votes && (
                  <div className="flex items-center gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
                    <button
                      onClick={() => handleVote(product.id, 'up')}
                      className={`flex items-center gap-1 px-3 py-1 rounded-lg transition-colors ${
                        product.votes.userVote === 'up'
                          ? 'bg-green-500 text-white'
                          : 'bg-gray-100 dark:bg-gray-700 hover:bg-green-100 dark:hover:bg-green-900/30'
                      }`}
                    >
                      <ThumbsUp className="w-4 h-4" />
                      <span className="text-sm font-medium">{product.votes.upvotes}</span>
                    </button>
                    <button
                      onClick={() => handleVote(product.id, 'down')}
                      className={`flex items-center gap-1 px-3 py-1 rounded-lg transition-colors ${
                        product.votes.userVote === 'down'
                          ? 'bg-red-500 text-white'
                          : 'bg-gray-100 dark:bg-gray-700 hover:bg-red-100 dark:hover:bg-red-900/30'
                      }`}
                    >
                      <ThumbsDown className="w-4 h-4" />
                      <span className="text-sm font-medium">{product.votes.downvotes}</span>
                    </button>
                    <span className="ml-auto text-sm font-medium text-gray-600 dark:text-gray-400">
                      Score: {product.votes.score}
                    </span>
                  </div>
                )}

                {/* Stock Status */}
                <div className="mt-4">
                  <Button className="w-full" disabled={!product.inStock}>
                    {product.inStock ? 'Add to Cart' : 'Out of Stock'}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
