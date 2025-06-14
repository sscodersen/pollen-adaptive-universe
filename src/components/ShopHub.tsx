
import React, { useState, useEffect, useCallback } from 'react';
import { ShoppingBag, ExternalLink, Star, TrendingUp, Search, Tag, Heart, Share2, RefreshCw } from 'lucide-react';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

// The ScrapedContent interface is no longer needed as we are creating products directly.

interface ShopHubProps {
  isGenerating?: boolean;
}

interface Product {
  id: string;
  name: string;
  description: string;
  price: string;
  originalPrice?: string;
  rating: number;
  reviews: number;
  category: string;
  tags: string[];
  significance: number;
  trending: boolean;
  link: string;
  seller: string;
  discount?: number;
  features: string[];
  inStock: boolean;
}

const mockProductTemplates = [
  {
    title: 'Professional AI Development Workstation - RTX 4090 Setup',
    description: 'Complete AI/ML development setup with cutting-edge GPU and optimized cooling',
    content: 'Pre-configured workstation designed specifically for AI development and machine learning tasks. Features RTX 4090 GPU, 128GB RAM, and optimized cooling system for sustained high-performance computing.',
    author: 'TechPro Solutions',
    tags: ['AI Hardware', 'Workstation', 'GPU', 'Development'],
    path: 'product',
    metadata: { price: 4999.99, productCategory: 'Health Tech', features: ['RTX 4090', '128GB DDR5 RAM', 'Liquid Cooling'] }
  },
  {
    title: 'Advanced Neural Network Training Software Suite',
    description: 'Professional-grade ML training platform with visual workflow builder',
    content: 'Comprehensive software suite for training and deploying neural networks. Includes visual workflow builder, automated hyperparameter tuning, and production deployment tools.',
    author: 'AI Solutions Inc.',
    tags: ['Software', 'AI Training', 'Machine Learning', 'Tools'],
    path: 'software',
    metadata: { price: 999, productCategory: 'Health Tech', features: ['Visual Workflow Builder', 'AutoML', 'Deployment Tools'] }
  },
  {
    title: 'Allbirds Wool Runners',
    description: 'Sustainable and comfortable everyday sneakers made from merino wool and recycled materials.',
    content: 'Sustainable and comfortable everyday sneakers made from merino wool and recycled materials.',
    author: 'Allbirds',
    tags: ['Sustainable', 'Footwear', 'Comfort'],
    path: 'products/womens-wool-runners',
    metadata: { price: 95.00, productCategory: 'Fashion', link: 'https://www.allbirds.com/products/womens-wool-runners', features: ['Merino Wool Upper', 'SweetFoam™ midsole', 'Machine washable'], discount: 0 }
  },
  {
    title: 'Click & Grow Smart Garden 3',
    description: 'An innovative indoor garden that cares for itself. Grow fresh, flavourful herbs, fruits or vegetables in your home.',
    content: 'An innovative indoor garden that cares for itself. Grow fresh, flavourful herbs, fruits or vegetables in your home.',
    author: 'Click & Grow',
    tags: ['Gardening', 'Smart Home', 'Wellness'],
    path: 'products/the-smart-garden-3',
    metadata: { price: 99.95, productCategory: 'Home Goods', link: 'https://www.clickandgrow.com/products/the-smart-garden-3', features: ['Automated watering', 'Perfect amount of light', '3 complimentary plant pods'], discount: 0 }
  },
  {
    title: 'Fellow Stagg EKG Electric Kettle',
    description: 'A beautifully designed electric kettle for the perfect pour-over coffee. Variable temperature control and a precision spout.',
    content: 'A beautifully designed electric kettle for the perfect pour-over coffee. Variable temperature control and a precision spout.',
    author: 'Fellow',
    tags: ['Coffee', 'Kitchen', 'Design'],
    path: 'products/stagg-ekg-electric-pour-over-kettle',
    metadata: { price: 165.00, originalPrice: 195.00, productCategory: 'Home Goods', link: 'https://fellowproducts.com/products/stagg-ekg-electric-pour-over-kettle', features: ['Variable temperature control', 'LCD screen', 'Precision pour spout'], discount: 15 }
  },
  {
    title: 'Patagonia Nano Puff Jacket',
    description: 'Warm, windproof, water-resistant—the Nano Puff® Jacket uses incredibly lightweight and highly compressible 60-g PrimaLoft® Gold Insulation Eco.',
    content: 'Warm, windproof, water-resistant—the Nano Puff® Jacket uses incredibly lightweight and highly compressible 60-g PrimaLoft® Gold Insulation Eco.',
    author: 'Patagonia',
    tags: ['Outdoor', 'Sustainable', 'Recycled'],
    path: 'product/mens-nano-puff-jacket/84212.html',
    metadata: { price: 239.00, productCategory: 'Fashion', link: 'https://www.patagonia.com/product/mens-nano-puff-jacket/84212.html', features: ['100% recycled shell', 'PrimaLoft® Gold Insulation', 'Fair Trade Certified™'], discount: 0 }
  },
  {
    title: 'Theragun Mini',
    description: 'A portable, powerful percussive therapy device. Compact but powerful, the Theragun mini is the most agile massage device that goes wherever you do.',
    content: 'A portable, powerful percussive therapy device. Compact but powerful, the Theragun mini is the most agile massage device that goes wherever you do.',
    author: 'Therabody',
    tags: ['Fitness', 'Recovery', 'Health'],
    path: 'us/en-us/mini-us.html',
    metadata: { price: 199.00, productCategory: 'Wellness', link: 'https://www.therabody.com/us/en-us/mini-us.html', features: ['QuietForce Technology', '3 Speed Settings', '150-minute battery life'], discount: 0 }
  },
  {
    title: 'Blueland The Clean Essentials Kit',
    description: 'A revolutionary way to clean your home without plastic waste. Reusable bottles and tablet refills for hand soap, multi-surface cleaner, and more.',
    content: 'A revolutionary way to clean your home without plastic waste. Reusable bottles and tablet refills for hand soap, multi-surface cleaner, and more.',
    author: 'Blueland',
    tags: ['Sustainable', 'Cleaning', 'Zero Waste'],
    path: 'products/the-clean-essentials',
    metadata: { price: 39.00, productCategory: 'Eco-Friendly', link: 'https://www.blueland.com/products/the-clean-essentials', features: ['Reduces plastic waste', 'Non-toxic formulas', 'Reusable bottles'], discount: 0 }
  },
  {
    title: 'Oura Ring Gen3',
    description: 'A smart ring that tracks your sleep, activity, recovery, temperature, heart rate, stress, and more.',
    content: 'A smart ring that tracks your sleep, activity, recovery, temperature, heart rate, stress, and more.',
    author: 'Oura',
    tags: ['Wearable', 'Health', 'Sleep Tracking'],
    path: '/',
    metadata: { price: 299.00, productCategory: 'Health Tech', link: 'https://ouraring.com/', features: ['24/7 heart rate monitoring', 'Advanced sleep analysis', '7-day battery life'], discount: 0 }
  },
  {
    title: 'LARQ Bottle PureVis™',
    description: 'The world’s first self-cleaning water bottle and water purification system. It uses PureVis technology to eliminate up to 99% of bio-contaminants.',
    content: 'The world’s first self-cleaning water bottle and water purification system. It uses PureVis technology to eliminate up to 99% of bio-contaminants.',
    author: 'LARQ',
    tags: ['Health', 'Outdoors', 'Tech', 'Sustainable'],
    path: 'product/larq-bottle-purevis',
    metadata: { price: 99.00, productCategory: 'Wellness', link: 'https://www.livelarq.com/product/larq-bottle-purevis', features: ['Self-cleaning mode', 'Eliminates bacteria & viruses', 'Keeps water cold 24h'], discount: 0 }
  }
];

export const ShopHub = ({ isGenerating = false }: ShopHubProps) => {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('significance');

  const categories = [
    'Fashion', 'Home Goods', 'Wellness', 'Eco-Friendly', 'Health Tech'
  ];

  const loadProducts = useCallback(() => {
    setLoading(true);
    
    const shuffledTemplates = [...mockProductTemplates].sort(() => Math.random() - 0.5);

    const newProducts = shuffledTemplates.map((template, index): Product => {
      const metadata = template.metadata as any;
      const priceNum = metadata.price || 0;
      const originalPriceNum = metadata.originalPrice;
      const discountNum = metadata.discount || 0;

      const scored = significanceAlgorithm.scoreContent(template.content, 'shop', 'Market Analysis');

      return {
        id: `${template.title.replace(/\s+/g, '-')}-${index}`,
        name: template.title,
        description: template.description,
        price: `$${priceNum.toFixed(2)}`,
        originalPrice: originalPriceNum ? `$${originalPriceNum.toFixed(2)}` : undefined,
        rating: metadata.rating || (Math.random() * 1.5 + 3.5),
        reviews: Math.floor(Math.random() * 5000) + 100,
        category: metadata.productCategory || 'General',
        tags: template.tags || [],
        significance: scored.significanceScore,
        trending: scored.significanceScore > 7.5 || discountNum > 10,
        link: metadata.link || '#',
        seller: template.author || 'Mock Seller',
        discount: discountNum,
        features: metadata.features || [],
        inStock: Math.random() > 0.05,
      };
    });

    setProducts(newProducts.sort((a, b) => b.significance - a.significance));
    setLoading(false);
  }, []);

  useEffect(() => {
    loadProducts();
  }, [loadProducts]);

  const filteredProducts = products.filter(product => {
    if (searchQuery) {
      return product.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
             product.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
             product.category.toLowerCase().includes(searchQuery.toLowerCase()) ||
             product.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    }
    if (filter === 'trending') return product.trending;
    if (filter === 'discounted') return product.discount && product.discount > 0;
    if (filter === 'all') return true;
    return product.category === filter;
  });

  const sortedProducts = [...filteredProducts].sort((a, b) => {
    if (sortBy === 'significance') return b.significance - a.significance;
    if (sortBy === 'price') return parseFloat(a.price.replace(/[^0-9.]/g, '')) - parseFloat(b.price.replace(/[^0-9.]/g, ''));
    if (sortBy === 'rating') return b.rating - a.rating;
    if (sortBy === 'discount') return (b.discount || 0) - (a.discount || 0);
    return b.significance - a.significance;
  });

  return (
    <div className="flex-1 bg-gray-950">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2">Smart Shopping</h1>
              <p className="text-gray-400">AI-curated products • Real marketplace links • Verified sellers</p>
            </div>
            <div className="flex items-center space-x-3">
              <div className="px-4 py-2 bg-green-500/10 text-green-400 rounded-full text-sm font-medium border border-green-500/20 flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Live Prices</span>
              </div>
              <button
                onClick={loadProducts}
                className="p-2.5 bg-gray-800/50 hover:bg-gray-700/50 rounded-lg transition-colors text-gray-400 hover:text-cyan-400 disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={loading}
                aria-label="Refresh products"
              >
                <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>

          {/* Search and Filters */}
          <div className="flex items-center justify-between space-x-4 mb-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search products, categories, features..."
                className="w-full bg-gray-800/50 border border-gray-700/50 rounded-lg pl-10 pr-4 py-3 text-white placeholder-gray-400 focus:border-cyan-500/50 focus:outline-none"
              />
            </div>

            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-gray-800/50 border border-gray-700/50 rounded-lg px-4 py-2 text-white text-sm focus:outline-none focus:border-cyan-500/50"
            >
              <option value="significance">Sort by Relevance</option>
              <option value="price">Sort by Price</option>
              <option value="rating">Sort by Rating</option>
              <option value="discount">Sort by Discount</option>
            </select>
          </div>

          {/* Category Filters */}
          <div className="flex space-x-2 overflow-x-auto pb-2">
            <button
              onClick={() => setFilter('all')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
                filter === 'all'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 border border-gray-700/30'
              }`}
            >
              <ShoppingBag className="w-4 h-4" />
              <span className="text-sm font-medium">All Products</span>
            </button>
            <button
              onClick={() => setFilter('trending')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
                filter === 'trending'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 border border-gray-700/30'
              }`}
            >
              <TrendingUp className="w-4 h-4" />
              <span className="text-sm font-medium">Trending</span>
            </button>
            <button
              onClick={() => setFilter('discounted')}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
                filter === 'discounted'
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 border border-gray-700/30'
              }`}
            >
              <Tag className="w-4 h-4" />
              <span className="text-sm font-medium">On Sale</span>
            </button>
            {categories.map((category) => (
              <button
                key={category}
                onClick={() => setFilter(category)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all whitespace-nowrap ${
                  filter === category
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                    : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50 border border-gray-700/30'
                }`}
              >
                <span className="text-sm font-medium">{category}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Products Grid */}
      <div className="p-6">
        {loading && products.length === 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(9)].map((_, i) => (
              <div key={i} className="bg-gray-900/50 rounded-xl p-6 border border-gray-800/50 animate-pulse">
                <div className="w-full h-48 bg-gray-700 rounded-lg mb-4"></div>
                <div className="w-3/4 h-4 bg-gray-700 rounded mb-2"></div>
                <div className="w-full h-3 bg-gray-700 rounded mb-4"></div>
                <div className="flex justify-between items-center mb-4">
                  <div className="w-20 h-6 bg-gray-700 rounded"></div>
                  <div className="w-16 h-4 bg-gray-700 rounded"></div>
                </div>
                <div className="w-full h-10 bg-gray-700 rounded"></div>
              </div>
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {sortedProducts.map((product) => (
              <div key={product.id} className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-all group">
                {/* Product Image Placeholder */}
                <div className="w-full h-48 bg-gradient-to-br from-gray-800 to-gray-700 rounded-lg mb-4 flex items-center justify-center group-hover:from-cyan-500/20 group-hover:to-purple-500/20 transition-all">
                  <ShoppingBag className="w-16 h-16 text-gray-500 group-hover:text-cyan-400 transition-colors" />
                </div>

                {/* Product Info */}
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-2">
                    <span className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-xs border border-purple-500/30">
                      {product.category}
                    </span>
                    <div className="flex items-center space-x-2">
                      {product.trending && (
                        <TrendingUp className="w-4 h-4 text-cyan-400" />
                      )}
                      <div className={`px-2 py-1 rounded text-xs font-medium ${
                        product.significance > 8 
                          ? 'bg-green-500/20 text-green-300'
                          : 'bg-cyan-500/20 text-cyan-300'
                      }`}>
                        {product.significance.toFixed(1)}
                      </div>
                    </div>
                  </div>

                  <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-cyan-300 transition-colors">
                    {product.name}
                  </h3>

                  <p className="text-gray-400 text-sm mb-3 line-clamp-2">
                    {product.description}
                  </p>

                  {/* Features */}
                  <div className="mb-4">
                    <div className="flex flex-wrap gap-1">
                      {product.features.slice(0, 3).map((feature, index) => (
                        <span key={index} className="px-2 py-1 bg-gray-700/50 text-gray-300 rounded text-xs">
                          {feature}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Rating and Reviews */}
                  <div className="flex items-center space-x-4 mb-4">
                    <div className="flex items-center space-x-1">
                      <Star className="w-4 h-4 text-yellow-400 fill-current" />
                      <span className="text-sm text-white font-medium">{product.rating.toFixed(1)}</span>
                    </div>
                    <span className="text-xs text-gray-400">({product.reviews.toLocaleString()} reviews)</span>
                    <span className="text-xs text-gray-500">by {product.seller}</span>
                  </div>

                  {/* Price */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      <span className="text-xl font-bold text-white">{product.price}</span>
                      {product.originalPrice && (
                        <span className="text-sm text-gray-400 line-through">{product.originalPrice}</span>
                      )}
                      {product.discount && product.discount > 0 && (
                        <span className="px-2 py-1 bg-red-500/20 text-red-300 rounded text-xs font-medium">
                          -{product.discount}%
                        </span>
                      )}
                    </div>
                    <div className={`px-2 py-1 rounded text-xs ${
                      product.inStock 
                        ? 'bg-green-500/20 text-green-300'
                        : 'bg-red-500/20 text-red-300'
                    }`}>
                      {product.inStock ? 'Available' : 'Out of Stock'}
                    </div>
                  </div>

                  {/* Tags */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    {product.tags.slice(0, 3).map((tag, index) => (
                      <span key={index} className="px-2 py-1 bg-gray-600/20 text-gray-400 rounded text-xs border border-gray-600/30">
                        #{tag}
                      </span>
                    ))}
                  </div>
                </div>

                {/* Actions */}
                <div className="flex items-center space-x-3">
                  <a
                    href={product.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex-1 bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 px-4 py-2 rounded-lg font-medium transition-all flex items-center justify-center space-x-2 text-white"
                  >
                    <ShoppingBag className="w-4 h-4" />
                    <span>Visit Site</span>
                    <ExternalLink className="w-4 h-4" />
                  </a>
                  <button className="p-2 bg-gray-800/50 hover:bg-gray-700/50 rounded-lg transition-colors">
                    <Heart className="w-4 h-4 text-gray-400 hover:text-red-400 transition-colors" />
                  </button>
                  <button className="p-2 bg-gray-800/50 hover:bg-gray-700/50 rounded-lg transition-colors">
                    <Share2 className="w-4 h-4 text-gray-400 hover:text-blue-400 transition-colors" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
