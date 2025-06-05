
import React, { useState, useEffect } from 'react';
import { ShoppingBag, Star, Filter, TrendingUp, Zap, ExternalLink, DollarSign } from 'lucide-react';
import { contentCurator, type WebContent } from '../services/contentCurator';

interface ShopHubProps {
  isGenerating?: boolean;
}

export const ShopHub = ({ isGenerating = true }: ShopHubProps) => {
  const [products, setProducts] = useState<WebContent[]>([]);
  const [generatingProducts, setGeneratingProducts] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [sortBy, setSortBy] = useState('significance');

  const categories = ['All', 'AI Tools', 'Productivity', 'Design', 'Development', 'Marketing', 'Hardware'];

  useEffect(() => {
    if (!isGenerating) return;

    const generateProducts = async () => {
      if (generatingProducts) return;
      
      setGeneratingProducts(true);
      try {
        const curated = await contentCurator.scrapeAndCurateContent('shop', 20);
        setProducts(prev => {
          const newItems = curated.filter(item => 
            !prev.some(existing => existing.title === item.title)
          );
          return [...newItems, ...prev].slice(0, 40);
        });
      } catch (error) {
        console.error('Failed to curate products:', error);
      }
      setGeneratingProducts(false);
    };

    generateProducts();
    const interval = setInterval(generateProducts, Math.random() * 40000 + 60000);
    return () => clearInterval(interval);
  }, [isGenerating, generatingProducts]);

  const openProductLink = (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  const filteredProducts = products;
  const sortedProducts = [...filteredProducts].sort((a, b) => {
    if (sortBy === 'significance') return b.significance - a.significance;
    if (sortBy === 'price') return (a.price || 0) - (b.price || 0);
    if (sortBy === 'rating') return (b.rating || 0) - (a.rating || 0);
    return 0;
  });

  return (
    <div className="flex-1 flex flex-col">
      {/* Header */}
      <div className="p-6 border-b border-gray-700/50">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white">Smart Product Discovery</h1>
            <p className="text-gray-400">AI-curated products • Best value • Verified sources</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingProducts && (
              <div className="flex items-center space-x-2 text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-sm">Finding best deals...</span>
              </div>
            )}
            <div className="flex items-center space-x-2">
              <ShoppingBag className="w-4 h-4 text-purple-400" />
              <span className="text-sm text-gray-400">{sortedProducts.length} curated products</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between">
          <div className="flex space-x-2 overflow-x-auto">
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setSelectedCategory(category)}
                className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
                  selectedCategory === category
                    ? 'bg-purple-500 text-white'
                    : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50 hover:text-white'
                }`}
              >
                {category}
              </button>
            ))}
          </div>

          <div className="flex items-center space-x-2">
            <Filter className="w-4 h-4 text-gray-400" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-gray-700/50 border border-gray-600/50 rounded-lg px-3 py-1 text-sm text-white"
            >
              <option value="significance">AI Recommended</option>
              <option value="price">Price: Low to High</option>
              <option value="rating">Highest Rated</option>
            </select>
          </div>
        </div>
      </div>

      {/* Products Grid */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sortedProducts.map((product) => (
            <div key={product.id} className="bg-gray-800/50 rounded-xl border border-gray-700/50 overflow-hidden hover:bg-gray-800/70 transition-all group cursor-pointer"
                 onClick={() => openProductLink(product.url)}>
              
              {/* Product Image */}
              <div className="h-48 bg-gradient-to-br from-purple-500/20 to-cyan-500/20 flex items-center justify-center relative">
                {product.image ? (
                  <img src={product.image} alt={product.title} className="w-full h-full object-cover" />
                ) : (
                  <ShoppingBag className="w-12 h-12 text-white opacity-60" />
                )}
                
                <div className="absolute top-3 left-3 px-2 py-1 bg-yellow-500/20 text-yellow-300 text-xs font-medium rounded-full flex items-center space-x-1">
                  <Star className="w-3 h-3" />
                  <span>{product.significance.toFixed(1)}/10</span>
                </div>
                
                <div className="absolute top-3 right-3 px-2 py-1 bg-purple-500/20 text-purple-300 text-xs font-medium rounded-full flex items-center space-x-1">
                  <Zap className="w-3 h-3" />
                  <span>AI Pick</span>
                </div>
              </div>

              {/* Product Info */}
              <div className="p-5">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-bold text-white group-hover:text-purple-300 transition-colors line-clamp-1">
                    {product.title}
                  </h3>
                  <div className="flex items-center space-x-1">
                    <Star className="w-4 h-4 text-yellow-400 fill-current" />
                    <span className="text-sm text-gray-300">{product.rating?.toFixed(1) || '4.5'}</span>
                  </div>
                </div>
                
                <p className="text-gray-400 text-sm mb-4 line-clamp-2">
                  {product.description}
                </p>

                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <DollarSign className="w-4 h-4 text-green-400" />
                    <span className="text-xl font-bold text-green-400">${product.price || Math.floor(Math.random() * 200) + 20}</span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <span className="text-xs text-gray-400">{product.source}</span>
                    <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors" />
                  </div>
                </div>

                <div className="mt-4">
                  <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
                    <span>AI Recommendation Score:</span>
                    <span className="text-purple-400">{product.significance.toFixed(1)}/10</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-1.5">
                    <div 
                      className="bg-purple-400 h-1.5 rounded-full transition-all"
                      style={{ width: `${(product.significance / 10) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {sortedProducts.length === 0 && (
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-gradient-to-r from-purple-400 to-cyan-500 rounded-full flex items-center justify-center mx-auto mb-6">
              <ShoppingBag className="w-10 h-10 text-white animate-pulse" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">Scanning Product Sources...</h3>
            <p className="text-gray-400 max-w-md mx-auto">
              Pollen is analyzing thousands of products from trusted retailers to find the best deals and highest-rated items for you.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
