
import React, { useState, useEffect } from 'react';
import { ShoppingBag, Star, TrendingUp, ExternalLink, Heart, Filter, Search, Award, Trophy, Target, Zap, Eye, Share2, Bookmark } from 'lucide-react';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  originalPrice?: number;
  rating: number;
  reviews: number;
  category: string;
  image: string;
  seller: string;
  shipping: string;
  significance: number;
  trending: boolean;
  rank: number;
  savings?: number;
  tags: string[];
  url: string; // Real product URL
  verified: boolean;
}

interface ShopHubProps {
  isGenerating?: boolean;
}

export const ShopHub = ({ isGenerating = true }: ShopHubProps) => {
  const [products, setProducts] = useState<Product[]>([]);
  const [generatingProduct, setGeneratingProduct] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('significance');
  const [priceRange, setPriceRange] = useState([0, 1000]);

  const categories = ['All', 'Tech', 'Health', 'Home', 'Fashion', 'Books', 'Sports', 'Travel'];

  const trendingProducts = [
    { category: 'AI Hardware', growth: '+234%', products: 1547 },
    { category: 'Sustainable Tech', growth: '+189%', products: 923 },
    { category: 'Health Monitors', growth: '+156%', products: 742 },
    { category: 'Smart Home', growth: '+134%', products: 856 },
    { category: 'Productivity Tools', growth: '+112%', products: 634 },
    { category: 'Learning Devices', growth: '+98%', products: 445 }
  ];

  const productTemplates = [
    {
      category: 'tech',
      items: [
        'Revolutionary AI-powered coding assistant device',
        'Quantum-inspired productivity enhancement tool',
        'Neural interface development kit for creators',
        'Advanced biometric health monitoring system',
        'Sustainable energy harvesting portable charger',
        'Holographic display projection device',
        'Voice-controlled smart workspace organizer',
        'Blockchain-secured personal data vault'
      ]
    },
    {
      category: 'health',
      items: [
        'Personalized nutrition optimization supplement',
        'Sleep quality enhancement smart mattress',
        'Mental wellness tracking meditation device',
        'DNA-based fitness optimization program',
        'Stress reduction biofeedback wearable',
        'Cognitive performance enhancement nootropics',
        'Recovery acceleration therapy device',
        'Longevity lifestyle optimization kit'
      ]
    },
    {
      category: 'home',
      items: [
        'Self-cleaning smart home maintenance system',
        'Air quality optimization purification unit',
        'Energy-efficient climate control system',
        'Automated hydroponic growing station',
        'Security enhancement surveillance network',
        'Noise cancellation environmental system',
        'Smart lighting circadian rhythm optimizer',
        'Waste reduction composting automation'
      ]
    }
  ];

  useEffect(() => {
    if (!isGenerating) return;

    const generateProduct = async () => {
      if (generatingProduct) return;
      
      setGeneratingProduct(true);
      try {
        const categoryKey = Object.keys(productTemplates)[Math.floor(Math.random() * productTemplates.length)];
        const category = productTemplates.find(c => c.category === categoryKey);
        const productName = category?.items[Math.floor(Math.random() * category.items.length)] || 'Smart Device';
        
        const basePrice = Math.floor(Math.random() * 800) + 50;
        const discount = Math.random() > 0.6 ? Math.floor(Math.random() * 40) + 10 : 0;
        const finalPrice = discount > 0 ? basePrice * (1 - discount/100) : basePrice;
        
        const significance = Math.random() * 3 + 7; // 7-10 range
        const rating = Math.random() * 1.5 + 3.5; // 3.5-5 range
        const reviews = Math.floor(Math.random() * 5000) + 100;
        
        const sellers = [
          'TechInnovate Labs', 'FutureWare Solutions', 'Quantum Dynamics', 'BioTech Innovations',
          'Smart Living Co.', 'NextGen Technologies', 'Advanced Systems Inc.', 'Innovation Hub Store'
        ];
        
        // Generate real product URLs (in real implementation, these would be actual affiliate links)
        const realUrls = [
          'https://amazon.com/dp/example1',
          'https://shopify.com/products/example2',
          'https://target.com/p/example3',
          'https://bestbuy.com/site/example4'
        ];

        const newProduct: Product = {
          id: Date.now().toString(),
          name: productName,
          description: generateProductDescription(productName, categoryKey),
          price: Math.round(finalPrice),
          originalPrice: discount > 0 ? basePrice : undefined,
          rating: Math.round(rating * 10) / 10,
          reviews,
          category: categoryKey,
          image: `/api/placeholder/300/300`,
          seller: sellers[Math.floor(Math.random() * sellers.length)],
          shipping: Math.random() > 0.5 ? 'Free shipping' : `$${Math.floor(Math.random() * 15) + 5} shipping`,
          significance,
          trending: significance > 8.5,
          rank: products.length + 1,
          savings: discount > 0 ? basePrice - finalPrice : undefined,
          tags: generateProductTags(categoryKey, productName),
          url: realUrls[Math.floor(Math.random() * realUrls.length)],
          verified: Math.random() > 0.3
        };
        
        setProducts(prev => [newProduct, ...prev.slice(0, 19)]
          .sort((a, b) => {
            if (sortBy === 'significance') return b.significance - a.significance;
            if (sortBy === 'price') return a.price - b.price;
            if (sortBy === 'rating') return b.rating - a.rating;
            return b.reviews - a.reviews;
          })
        );
      } catch (error) {
        console.error('Failed to generate product:', error);
      }
      setGeneratingProduct(false);
    };

    const initialTimeout = setTimeout(generateProduct, 1500);
    const interval = setInterval(generateProduct, Math.random() * 35000 + 25000);
    
    return () => {
      clearTimeout(initialTimeout);
      clearInterval(interval);
    };
  }, [isGenerating, generatingProduct, products.length, sortBy]);

  const generateProductDescription = (name: string, category: string) => {
    const descriptions = {
      tech: [
        `Revolutionary ${name.toLowerCase()} featuring cutting-edge technology and AI-powered optimization. Designed for professionals and enthusiasts who demand the best performance and reliability.`,
        `Advanced ${name.toLowerCase()} with breakthrough innovations that redefine what's possible. Industry-leading features combined with user-friendly design for exceptional results.`,
        `Next-generation ${name.toLowerCase()} engineered for superior performance and efficiency. Premium materials and sophisticated algorithms deliver unmatched capabilities.`
      ],
      health: [
        `Science-backed ${name.toLowerCase()} developed by leading researchers and health experts. Clinically tested formula designed to optimize your wellness journey naturally and effectively.`,
        `Premium ${name.toLowerCase()} combining traditional wisdom with modern innovation. Carefully crafted to support your health goals with proven ingredients and advanced delivery systems.`,
        `Professional-grade ${name.toLowerCase()} trusted by healthcare practitioners worldwide. Evidence-based approach to wellness with measurable results and safety assurance.`
      ],
      home: [
        `Innovative ${name.toLowerCase()} that transforms your living space with smart technology and elegant design. Energy-efficient solution that enhances comfort while reducing environmental impact.`,
        `Premium ${name.toLowerCase()} engineered for modern homes and lifestyles. Seamless integration with existing systems and intuitive controls for effortless operation.`,
        `Sophisticated ${name.toLowerCase()} combining functionality with aesthetic appeal. Durable construction and advanced features designed to exceed expectations.`
      ]
    };
    
    const categoryDescriptions = descriptions[category as keyof typeof descriptions] || descriptions.tech;
    return categoryDescriptions[Math.floor(Math.random() * categoryDescriptions.length)];
  };

  const generateProductTags = (category: string, name: string) => {
    const baseTags = {
      tech: ['#Innovation', '#AI', '#Smart', '#Premium'],
      health: ['#Wellness', '#Natural', '#Clinically-Tested', '#Health'],
      home: ['#SmartHome', '#Energy-Efficient', '#Modern', '#Comfort']
    };
    
    const popularTags = ['#BestSeller', '#Trending', '#HighlyRated', '#Verified'];
    const categoryTags = baseTags[category as keyof typeof baseTags] || baseTags.tech;
    
    return [...categoryTags.slice(0, 2), ...popularTags.slice(0, 2)];
  };

  const getRankBadge = (rank: number) => {
    if (rank <= 3) return { icon: Trophy, color: 'text-yellow-400 bg-yellow-400/20' };
    if (rank <= 10) return { icon: Award, color: 'text-blue-400 bg-blue-400/20' };
    return { icon: Target, color: 'text-gray-400 bg-gray-400/20' };
  };

  const filteredProducts = products.filter(product => {
    const matchesCategory = selectedCategory === 'All' || product.category === selectedCategory.toLowerCase();
    const matchesSearch = searchQuery === '' || 
      product.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesPrice = product.price >= priceRange[0] && product.price <= priceRange[1];
    return matchesCategory && matchesSearch && matchesPrice;
  });

  return (
    <div className="flex-1 flex flex-col bg-gray-950">
      {/* Header */}
      <div className="p-6 border-b border-gray-800/50 bg-gray-900/50 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Smart Shopping Hub</h1>
            <p className="text-gray-400">AI-curated products • Real-time price tracking • Intelligent recommendations</p>
          </div>
          <div className="flex items-center space-x-4">
            {generatingProduct && (
              <div className="flex items-center space-x-2 text-cyan-400">
                <div className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
                <span className="text-sm font-medium">Finding products...</span>
              </div>
            )}
            <div className="text-right">
              <div className="text-2xl font-bold text-white">{filteredProducts.length}</div>
              <div className="text-xs text-gray-400">Curated products</div>
            </div>
          </div>
        </div>

        {/* Trending Categories */}
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Trending Categories</h3>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
            {trendingProducts.map((trend) => (
              <div key={trend.category} className="p-3 bg-gray-800/50 rounded-lg border border-gray-700/30 hover:border-cyan-500/30 transition-colors cursor-pointer">
                <p className="text-sm font-medium text-white">{trend.category}</p>
                <div className="flex items-center justify-between mt-1">
                  <span className="text-xs text-gray-400">{trend.products}</span>
                  <span className="text-xs text-green-400">{trend.growth}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Search and Filters */}
        <div className="flex items-center space-x-4 mb-4">
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
          
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="bg-gray-800/50 border border-gray-700/50 rounded-lg px-3 py-2 text-white text-sm focus:border-cyan-500/50 focus:outline-none"
          >
            <option value="significance">By Significance</option>
            <option value="price">By Price</option>
            <option value="rating">By Rating</option>
            <option value="reviews">By Reviews</option>
          </select>
        </div>

        {/* Category Tabs */}
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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredProducts.map((product, index) => {
            const rankBadge = getRankBadge(product.rank);
            
            return (
              <div key={product.id} className="bg-gray-900/80 rounded-2xl border border-gray-800/50 overflow-hidden hover:bg-gray-900/90 transition-all duration-200 backdrop-blur-sm group">
                {/* Product Image */}
                <div className="relative">
                  <div className="w-full h-48 bg-gray-800/50 flex items-center justify-center">
                    <ShoppingBag className="w-16 h-16 text-gray-600" />
                  </div>
                  
                  {/* Badges */}
                  <div className="absolute top-3 left-3 flex flex-col space-y-2">
                    <div className={`flex items-center space-x-1 px-2 py-1 ${rankBadge.color} text-xs rounded-full`}>
                      <rankBadge.icon className="w-3 h-3" />
                      <span>#{product.rank}</span>
                    </div>
                    
                    {product.trending && (
                      <div className="flex items-center space-x-1 px-2 py-1 bg-orange-500/20 text-orange-400 text-xs rounded-full">
                        <TrendingUp className="w-3 h-3" />
                        <span>Hot</span>
                      </div>
                    )}
                  </div>

                  <div className="absolute top-3 right-3">
                    <div className="flex items-center space-x-1 px-2 py-1 bg-cyan-500/20 text-cyan-400 text-xs rounded-full">
                      <Zap className="w-3 h-3" />
                      <span>{product.significance.toFixed(1)}</span>
                    </div>
                  </div>

                  {product.savings && (
                    <div className="absolute bottom-3 left-3">
                      <div className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full">
                        Save ${product.savings}
                      </div>
                    </div>
                  )}
                </div>

                {/* Product Info */}
                <div className="p-4">
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="font-semibold text-white text-sm line-clamp-2 group-hover:text-cyan-300 transition-colors">
                      {product.name}
                    </h3>
                    {product.verified && <Star className="w-4 h-4 text-blue-400 fill-current flex-shrink-0" />}
                  </div>
                  
                  <p className="text-gray-400 text-xs line-clamp-2 mb-3">
                    {product.description}
                  </p>

                  {/* Tags */}
                  <div className="flex flex-wrap gap-1 mb-3">
                    {product.tags.slice(0, 3).map((tag) => (
                      <span key={tag} className="px-2 py-1 bg-gray-800/50 text-cyan-400 text-xs rounded">
                        {tag}
                      </span>
                    ))}
                  </div>

                  {/* Rating and Reviews */}
                  <div className="flex items-center space-x-2 mb-3">
                    <div className="flex items-center space-x-1">
                      <Star className="w-4 h-4 text-yellow-400 fill-current" />
                      <span className="text-sm font-medium text-white">{product.rating}</span>
                    </div>
                    <span className="text-xs text-gray-400">({product.reviews.toLocaleString()} reviews)</span>
                  </div>

                  {/* Price */}
                  <div className="flex items-center space-x-2 mb-3">
                    <span className="text-lg font-bold text-white">${product.price}</span>
                    {product.originalPrice && (
                      <span className="text-sm text-gray-400 line-through">${product.originalPrice}</span>
                    )}
                  </div>

                  {/* Seller Info */}
                  <div className="text-xs text-gray-400 mb-4">
                    Sold by {product.seller} • {product.shipping}
                  </div>

                  {/* Actions */}
                  <div className="flex space-x-2">
                    <a
                      href={product.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex-1 bg-gradient-to-r from-cyan-500 to-purple-500 text-white px-4 py-2 rounded-lg text-sm font-medium hover:from-cyan-600 hover:to-purple-600 transition-all flex items-center justify-center space-x-2"
                    >
                      <span>View Product</span>
                      <ExternalLink className="w-4 h-4" />
                    </a>
                    <button className="p-2 bg-gray-800/50 rounded-lg hover:bg-gray-700/50 transition-colors">
                      <Heart className="w-4 h-4 text-gray-400 hover:text-red-400" />
                    </button>
                    <button className="p-2 bg-gray-800/50 rounded-lg hover:bg-gray-700/50 transition-colors">
                      <Share2 className="w-4 h-4 text-gray-400 hover:text-white" />
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {filteredProducts.length === 0 && (
          <div className="text-center py-20">
            <div className="w-24 h-24 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full flex items-center justify-center mx-auto mb-8">
              <ShoppingBag className="w-12 h-12 text-white animate-pulse" />
            </div>
            <h3 className="text-2xl font-bold text-white mb-4">Smart Shopping Engine Loading...</h3>
            <p className="text-gray-400 max-w-lg mx-auto text-lg">
              Pollen AI is analyzing global markets and curating the best products based on quality, value, and innovation.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
