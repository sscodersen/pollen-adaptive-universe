
import React, { useState, useEffect, useCallback } from 'react';
import { ShoppingBag, ExternalLink, Star, TrendingUp, Award, Filter, Search, Tag, Heart, Share2 } from 'lucide-react';
import { significanceAlgorithm } from '../services/significanceAlgorithm';

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

export const ShopHub = ({ isGenerating = false }: ShopHubProps) => {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('significance');

  const categories = [
    'AI Tools', 'Productivity', 'Health Tech', 'Smart Home', 'Education', 
    'Creative Tools', 'Sustainable Tech', 'Fitness', 'Professional Software'
  ];

  const generateProducts = useCallback(async () => {
    const productTemplates = [
      {
        name: 'AI-Powered Code Assistant Pro',
        description: 'Advanced AI coding companion that understands context, generates optimized code, and provides real-time debugging assistance across 40+ programming languages.',
        price: '$29.99/mo',
        originalPrice: '$49.99/mo',
        category: 'AI Tools',
        tags: ['AI', 'Programming', 'Productivity'],
        seller: 'TechFlow Solutions',
        link: 'https://example.com/ai-code-assistant',
        features: ['Multi-language support', 'Real-time debugging', 'Code optimization', 'Team collaboration'],
        discount: 40
      },
      {
        name: 'Neural Audio Enhancement Suite',
        description: 'Professional-grade AI audio processing software that removes noise, enhances clarity, and optimizes sound quality using machine learning algorithms.',
        price: '$149.99',
        originalPrice: '$199.99',
        category: 'Creative Tools',
        tags: ['Audio', 'AI', 'Professional'],
        seller: 'SoundTech Pro',
        link: 'https://example.com/neural-audio',
        features: ['AI noise reduction', 'Voice enhancement', 'Batch processing', 'VST plugin support'],
        discount: 25
      },
      {
        name: 'Smart Productivity Tracker',
        description: 'Intelligent time tracking and productivity analysis tool that uses AI to identify patterns, suggest optimizations, and boost work efficiency.',
        price: '$19.99/mo',
        category: 'Productivity',
        tags: ['Productivity', 'Analytics', 'Time Management'],
        seller: 'Efficient Systems',
        link: 'https://example.com/productivity-tracker',
        features: ['AI insights', 'Goal tracking', 'Team reports', 'Mobile app'],
        discount: 0
      },
      {
        name: 'Sustainable Energy Monitor',
        description: 'Smart home energy monitoring system that tracks usage, identifies waste, and provides AI-powered recommendations for reducing carbon footprint.',
        price: '$89.99',
        originalPrice: '$119.99',
        category: 'Sustainable Tech',
        tags: ['Green Tech', 'Smart Home', 'Energy'],
        seller: 'EcoTech Innovations',
        link: 'https://example.com/energy-monitor',
        features: ['Real-time monitoring', 'AI recommendations', 'Mobile alerts', 'Solar integration'],
        discount: 25
      },
      {
        name: 'Adaptive Learning Platform',
        description: 'Personalized education platform that adapts to individual learning styles using AI, providing customized courses and progress tracking.',
        price: '$39.99/mo',
        originalPrice: '$59.99/mo',
        category: 'Education',
        tags: ['Education', 'AI', 'Personalized Learning'],
        seller: 'LearnSmart Academy',
        link: 'https://example.com/adaptive-learning',
        features: ['Adaptive content', 'Progress analytics', 'Expert mentors', 'Certification'],
        discount: 33
      },
      {
        name: 'AI Fitness Coach Premium',
        description: 'Intelligent fitness tracking app that creates personalized workout plans, monitors progress, and adjusts routines based on performance data.',
        price: '$14.99/mo',
        category: 'Fitness',
        tags: ['Fitness', 'AI', 'Health'],
        seller: 'FitTech Solutions',
        link: 'https://example.com/ai-fitness',
        features: ['Custom workouts', 'Progress tracking', 'Nutrition guidance', 'Wearable sync'],
        discount: 0
      },
      {
        name: 'Smart Document Scanner Pro',
        description: 'AI-powered document digitization tool that automatically detects, crops, enhances, and organizes scanned documents with OCR capabilities.',
        price: '$79.99',
        originalPrice: '$99.99',
        category: 'Productivity',
        tags: ['Document Management', 'AI', 'OCR'],
        seller: 'DocuScan Technologies',
        link: 'https://example.com/document-scanner',
        features: ['AI enhancement', 'OCR text recognition', 'Cloud sync', 'Batch processing'],
        discount: 20
      },
      {
        name: 'Creative AI Image Generator',
        description: 'Professional AI image generation software for creating unique artwork, designs, and visual content using advanced neural networks.',
        price: '$24.99/mo',
        originalPrice: '$39.99/mo',
        category: 'Creative Tools',
        tags: ['AI Art', 'Image Generation', 'Creative'],
        seller: 'ArtTech Studios',
        link: 'https://example.com/ai-image-generator',
        features: ['High-res output', 'Style transfer', 'Batch generation', 'Commercial license'],
        discount: 37
      },
      {
        name: 'Smart Home Security Hub',
        description: 'Intelligent security system with AI-powered threat detection, facial recognition, and automated response protocols for comprehensive home protection.',
        price: '$199.99',
        originalPrice: '$299.99',
        category: 'Smart Home',
        tags: ['Security', 'Smart Home', 'AI'],
        seller: 'SecureHome Tech',
        link: 'https://example.com/security-hub',
        features: ['AI threat detection', 'Facial recognition', 'Mobile alerts', '24/7 monitoring'],
        discount: 33
      },
      {
        name: 'Professional Data Analytics Suite',
        description: 'Comprehensive business intelligence platform with AI-driven insights, predictive analytics, and automated reporting for data-driven decisions.',
        price: '$99.99/mo',
        originalPrice: '$149.99/mo',
        category: 'Professional Software',
        tags: ['Analytics', 'Business Intelligence', 'AI'],
        seller: 'DataPro Solutions',
        link: 'https://example.com/analytics-suite',
        features: ['Predictive analytics', 'Custom dashboards', 'API integration', 'Team collaboration'],
        discount: 33
      }
    ];

    const products = productTemplates.map((template, index) => {
      const scored = significanceAlgorithm.scoreContent(template.description, 'shop', 'Market Analysis');
      
      return {
        id: (Date.now() + index).toString(),
        name: template.name,
        description: template.description,
        price: template.price,
        originalPrice: template.originalPrice,
        rating: Math.random() * 1.5 + 3.5, // 3.5-5.0
        reviews: Math.floor(Math.random() * 5000) + 100,
        category: template.category,
        tags: template.tags,
        significance: scored.significanceScore,
        trending: scored.significanceScore > 7.5 || template.discount > 30,
        link: template.link,
        seller: template.seller,
        discount: template.discount,
        features: template.features,
        inStock: Math.random() > 0.1 // 90% in stock
      };
    });

    return products;
  }, []);

  const loadProducts = useCallback(async () => {
    setLoading(true);
    const newProducts = await generateProducts();
    setProducts(newProducts.sort((a, b) => b.significance - a.significance));
    setLoading(false);
  }, [generateProducts]);

  useEffect(() => {
    loadProducts();
    const interval = setInterval(loadProducts, 120000); // Refresh every 2 minutes
    return () => clearInterval(interval);
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
              <p className="text-gray-400">AI-curated products • Algorithm-ranked • Real marketplace links</p>
            </div>
            <div className="flex items-center space-x-3">
              <div className="px-4 py-2 bg-green-500/10 text-green-400 rounded-full text-sm font-medium border border-green-500/20 flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Live Prices</span>
              </div>
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
            {categories.slice(0, 4).map((category) => (
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
        {loading ? (
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
                      {product.inStock ? 'In Stock' : 'Out of Stock'}
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
                    <span>View Product</span>
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
