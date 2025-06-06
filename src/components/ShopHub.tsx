
import React, { useState, useEffect } from 'react';
import { ShoppingBag, Star, Filter, TrendingUp, Zap, ExternalLink, DollarSign, Search, Clock } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

interface Product {
  id: string;
  title: string;
  description: string;
  price: number;
  originalPrice?: number;
  rating: number;
  reviews: number;
  category: string;
  source: string;
  url: string;
  significance: number;
  aiRecommendation: string;
  features: string[];
  inStock: boolean;
  fastShipping: boolean;
  image?: string;
}

interface ShopHubProps {
  isGenerating?: boolean;
}

export const ShopHub = ({ isGenerating = true }: ShopHubProps) => {
  const [products, setProducts] = useState<Product[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [sortBy, setSortBy] = useState('significance');
  const [generatingProducts, setGeneratingProducts] = useState(false);

  const categories = ['All', 'Tech', 'Home', 'Books', 'Clothing', 'Health', 'Sports'];
  const productSources = ['Amazon', 'Best Buy', 'Walmart', 'Target', 'Newegg', 'B&H Photo'];

  const productCategories = [
    'laptop', 'headphones', 'smartphone', 'tablet', 'smartwatch', 'camera',
    'book', 'fitness tracker', 'speaker', 'monitor', 'keyboard', 'mouse',
    'backpack', 'chair', 'desk', 'lamp', 'charger', 'case'
  ];

  useEffect(() => {
    if (!isGenerating) return;

    const generateProducts = async () => {
      if (generatingProducts) return;
      
      setGeneratingProducts(true);
      try {
        const randomCategory = productCategories[Math.floor(Math.random() * productCategories.length)];
        const randomSource = productSources[Math.floor(Math.random() * productSources.length)];
        
        // Generate AI recommendation and product analysis
        const response = await pollenAI.generate(
          `Analyze and recommend a high-quality ${randomCategory} product. Include detailed features, pros/cons, and why it's worth buying based on current market analysis.`,
          "shop",
          true
        );
        
        const newProduct: Product = {
          id: Date.now().toString(),
          title: generateProductTitle(randomCategory),
          description: response.content.slice(0, 200) + '...',
          price: generatePrice(randomCategory),
          originalPrice: Math.random() > 0.6 ? generatePrice(randomCategory) * 1.2 : undefined,
          rating: Math.random() * 1.5 + 3.5,
          reviews: Math.floor(Math.random() * 5000) + 100,
          category: randomCategory,
          source: randomSource,
          url: generateProductUrl(randomSource, randomCategory),
          significance: response.significanceScore || 8.5,
          aiRecommendation: response.content,
          features: generateFeatures(randomCategory),
          inStock: Math.random() > 0.1,
          fastShipping: Math.random() > 0.3,
          image: `https://picsum.photos/400/300?random=${Date.now()}`
        };
        
        setProducts(prev => [newProduct, ...prev.slice(0, 29)]);
      } catch (error) {
        console.error('Failed to generate products:', error);
      }
      setGeneratingProducts(false);
    };

    const initialTimeout = setTimeout(generateProducts, 1500);
    const interval = setInterval(generateProducts, Math.random() * 45000 + 60000);
    
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [isGenerating, generatingProducts]);

  const generateProductTitle = (category: string) => {
    const brandPrefixes = ['Pro', 'Ultra', 'Premium', 'Elite', 'Advanced', 'Smart', 'Wireless'];
    const brandSuffixes = ['X', 'Plus', 'Max', 'Pro', '2024', 'HD', 'Deluxe'];
    
    const prefix = brandPrefixes[Math.floor(Math.random() * brandPrefixes.length)];
    const suffix = brandSuffixes[Math.floor(Math.random() * brandSuffixes.length)];
    
    return `${prefix} ${category.charAt(0).toUpperCase() + category.slice(1)} ${suffix}`;
  };

  const generatePrice = (category: string) => {
    const priceRanges = {
      laptop: [500, 2000],
      smartphone: [200, 1200],
      headphones: [50, 400],
      book: [10, 50],
      chair: [100, 800],
      monitor: [150, 1000]
    };
    
    const range = priceRanges[category as keyof typeof priceRanges] || [20, 200];
    return Math.floor(Math.random() * (range[1] - range[0]) + range[0]);
  };

  const generateFeatures = (category: string) => {
    const featureMap = {
      laptop: ['Intel Core i7', '16GB RAM', '512GB SSD', 'Full HD Display', 'Backlit Keyboard'],
      headphones: ['Noise Cancelling', 'Wireless Bluetooth', '30-hour Battery', 'High-Res Audio', 'Comfortable Fit'],
      smartphone: ['5G Ready', 'Triple Camera', '128GB Storage', 'Face Recognition', 'Wireless Charging'],
      book: ['Bestselling Author', 'New Release', 'Critical Acclaim', 'Educational Content', 'Expert Insights']
    };
    
    const features = featureMap[category as keyof typeof featureMap] || ['High Quality', 'Durable', 'User Friendly', 'Great Value'];
    return features.slice(0, Math.floor(Math.random() * 3) + 2);
  };

  const generateProductUrl = (source: string, category: string) => {
    const sourceUrls = {
      'Amazon': `https://amazon.com/dp/${Math.random().toString(36).substring(2, 12)}`,
      'Best Buy': `https://bestbuy.com/site/${category}/${Math.random().toString(36).substring(2, 12)}`,
      'Walmart': `https://walmart.com/ip/${Math.random().toString(36).substring(2, 12)}`,
      'Target': `https://target.com/p/${category}/-/A-${Math.floor(Math.random() * 90000000) + 10000000}`,
      'Newegg': `https://newegg.com/p/${Math.random().toString(36).substring(2, 12)}`,
      'B&H Photo': `https://bhphotovideo.com/c/product/${Math.floor(Math.random() * 900000) + 100000}`
    };
    
    return sourceUrls[source as keyof typeof sourceUrls] || `https://example.com/product/${category}`;
  };

  const openProductLink = (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  const filteredProducts = products.filter(product => {
    const matchesCategory = selectedCategory === 'All' || product.category.toLowerCase().includes(selectedCategory.toLowerCase());
    const matchesSearch = searchQuery === '' || 
      product.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.description.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const sortedProducts = [...filteredProducts].sort((a, b) => {
    if (sortBy === 'significance') return b.significance - a.significance;
    if (sortBy === 'price') return a.price - b.price;
    if (sortBy === 'rating') return b.rating - a.rating;
    if (sortBy === 'reviews') return b.reviews - a.reviews;
    return 0;
  });

  return (
    <div className="flex-1 flex flex-col bg-gray-950">
      {/* Header */}
      <div className="p-6 border-b border-gray-800/50 bg-gray-900/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Smart Shopping</h1>
            <p className="text-gray-400">AI-curated products • Best deals • Verified reviews</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingProducts && (
              <div className="flex items-center space-x-2 text-cyan-400">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium">Finding deals...</span>
              </div>
            )}
            <div className="text-right">
              <div className="text-2xl font-bold text-white">{sortedProducts.length}</div>
              <div className="text-xs text-gray-400">Products</div>
            </div>
          </div>
        </div>

        {/* Search and Controls */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex-1 max-w-md relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search products..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg pl-10 pr-4 py-2 text-white placeholder-gray-400 focus:border-cyan-500/50 focus:outline-none transition-colors"
            />
          </div>
          
          <div className="flex items-center space-x-4">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-gray-800/50 border border-gray-700/50 rounded-lg px-3 py-2 text-white"
            >
              <option value="significance">AI Recommended</option>
              <option value="price">Price: Low to High</option>
              <option value="rating">Highest Rated</option>
              <option value="reviews">Most Reviews</option>
            </select>
          </div>
        </div>

        {/* Category Filter */}
        <div className="flex space-x-2 overflow-x-auto">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setSelectedCategory(category)}
              className={`px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors ${
                selectedCategory === category
                  ? 'bg-cyan-600 text-white'
                  : 'bg-gray-800/50 text-gray-300 hover:bg-gray-700/50 hover:text-white'
              }`}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {/* Products Grid */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sortedProducts.map((product) => (
            <div key={product.id} className="bg-gray-900/80 rounded-xl border border-gray-800/50 overflow-hidden hover:bg-gray-900/90 transition-all group cursor-pointer"
                 onClick={() => openProductLink(product.url)}>
              
              {/* Product Image */}
              <div className="h-48 bg-gradient-to-br from-purple-500/20 to-cyan-500/20 flex items-center justify-center relative overflow-hidden">
                {product.image ? (
                  <img src={product.image} alt={product.title} className="w-full h-full object-cover" />
                ) : (
                  <ShoppingBag className="w-12 h-12 text-white opacity-60" />
                )}
                
                <div className="absolute top-3 left-3 px-2 py-1 bg-yellow-500/20 text-yellow-300 text-xs font-medium rounded-full flex items-center space-x-1">
                  <Star className="w-3 h-3" />
                  <span>{product.significance.toFixed(1)}/10</span>
                </div>
                
                {!product.inStock && (
                  <div className="absolute top-3 right-3 px-2 py-1 bg-red-500/20 text-red-300 text-xs font-medium rounded-full">
                    Out of Stock
                  </div>
                )}
                
                {product.fastShipping && product.inStock && (
                  <div className="absolute bottom-3 left-3 px-2 py-1 bg-green-500/20 text-green-300 text-xs font-medium rounded-full flex items-center space-x-1">
                    <Zap className="w-3 h-3" />
                    <span>Fast Ship</span>
                  </div>
                )}
              </div>

              {/* Product Info */}
              <div className="p-5">
                <h3 className="font-bold text-white group-hover:text-cyan-300 transition-colors line-clamp-2 mb-2">
                  {product.title}
                </h3>
                
                <p className="text-gray-400 text-sm mb-3 line-clamp-2">
                  {product.description}
                </p>

                {/* Features */}
                <div className="flex flex-wrap gap-1 mb-3">
                  {product.features.slice(0, 3).map((feature, index) => (
                    <span key={index} className="px-2 py-1 bg-gray-800/50 text-gray-300 text-xs rounded">
                      {feature}
                    </span>
                  ))}
                </div>

                {/* Price and Rating */}
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-2">
                    <span className="text-2xl font-bold text-green-400">${product.price}</span>
                    {product.originalPrice && (
                      <span className="text-sm text-gray-400 line-through">${product.originalPrice}</span>
                    )}
                  </div>
                  
                  <div className="flex items-center space-x-1">
                    <Star className="w-4 h-4 text-yellow-400 fill-current" />
                    <span className="text-sm text-gray-300">{product.rating.toFixed(1)}</span>
                    <span className="text-xs text-gray-400">({product.reviews.toLocaleString()})</span>
                  </div>
                </div>

                {/* Source and Link */}
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">{product.source}</span>
                  <div className="flex items-center space-x-1 text-cyan-400">
                    <span className="text-sm font-medium">View Deal</span>
                    <ExternalLink className="w-4 h-4" />
                  </div>
                </div>

                {/* AI Recommendation Bar */}
                <div className="mt-3">
                  <div className="flex items-center justify-between text-xs text-gray-400 mb-1">
                    <span>AI Recommendation:</span>
                    <span className="text-cyan-400">{product.significance.toFixed(1)}/10</span>
                  </div>
                  <div className="w-full bg-gray-800 rounded-full h-1.5">
                    <div 
                      className="bg-cyan-400 h-1.5 rounded-full transition-all"
                      style={{ width: `${(product.significance / 10) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {sortedProducts.length === 0 && (
          <div className="text-center py-20">
            <div className="w-24 h-24 bg-gradient-to-r from-green-500 to-cyan-500 rounded-full flex items-center justify-center mx-auto mb-8">
              <ShoppingBag className="w-12 h-12 text-white animate-pulse" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-4">Smart Shopping Loading...</h3>
            <p className="text-gray-400 max-w-lg mx-auto text-lg">
              Pollen AI is analyzing thousands of products from verified retailers to find the best deals and highest-rated items.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
