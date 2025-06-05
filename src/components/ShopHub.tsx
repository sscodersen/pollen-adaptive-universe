
import React, { useState, useEffect } from 'react';
import { ShoppingBag, Star, Filter, TrendingUp, Zap } from 'lucide-react';
import { pollenAI } from '../services/pollenAI';

interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  rating: number;
  category: string;
  image: string;
  trending: boolean;
  aiRecommended: boolean;
}

interface ShopHubProps {
  isGenerating?: boolean;
}

export const ShopHub = ({ isGenerating = true }: ShopHubProps) => {
  const [products, setProducts] = useState<Product[]>([]);
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [sortBy, setSortBy] = useState('recommended');

  const categories = ['All', 'AI Tools', 'Productivity', 'Design', 'Development', 'Marketing'];

  useEffect(() => {
    generateProducts();
  }, []);

  const generateProducts = async () => {
    const sampleProducts: Product[] = [
      {
        id: '1',
        name: 'Pollen AI Pro',
        description: 'Advanced AI reasoning and automation platform',
        price: 99,
        rating: 4.9,
        category: 'AI Tools',
        image: 'bg-gradient-to-br from-cyan-500 to-purple-500',
        trending: true,
        aiRecommended: true
      },
      {
        id: '2',
        name: 'Neural Workflow Designer',
        description: 'Visual tool for creating AI-powered workflows',
        price: 49,
        rating: 4.7,
        category: 'Productivity',
        image: 'bg-gradient-to-br from-green-500 to-blue-500',
        trending: false,
        aiRecommended: true
      },
      {
        id: '3',
        name: 'Smart Content Generator',
        description: 'AI-powered content creation for all platforms',
        price: 29,
        rating: 4.5,
        category: 'Marketing',
        image: 'bg-gradient-to-br from-orange-500 to-red-500',
        trending: true,
        aiRecommended: false
      }
    ];

    setProducts(sampleProducts);
  };

  const filteredProducts = selectedCategory === 'All' 
    ? products 
    : products.filter(product => product.category === selectedCategory);

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">AI-Powered Shop</h1>
          <p className="text-gray-400">Discover tools and services optimized by AI recommendations</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white"
          >
            <option value="recommended">AI Recommended</option>
            <option value="rating">Highest Rated</option>
            <option value="price">Price: Low to High</option>
            <option value="trending">Trending</option>
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
                ? 'bg-purple-500 text-white'
                : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50 hover:text-white'
            }`}
          >
            {category}
          </button>
        ))}
      </div>

      {/* Featured Section */}
      <div className="bg-gradient-to-r from-purple-500/20 to-cyan-500/20 border border-purple-500/30 rounded-xl p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Zap className="w-5 h-5 text-yellow-400" />
          <h3 className="text-lg font-semibold text-white">AI Recommendations for You</h3>
        </div>
        <p className="text-gray-300 mb-4">Based on your usage patterns and preferences, these tools will boost your productivity by 40%</p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {filteredProducts.filter(p => p.aiRecommended).slice(0, 3).map((product) => (
            <div key={product.id} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
              <div className={`h-32 ${product.image} rounded-lg mb-4 flex items-center justify-center`}>
                <ShoppingBag className="w-8 h-8 text-white" />
              </div>
              <h4 className="font-semibold text-white mb-2">{product.name}</h4>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-green-400">${product.price}</span>
                <div className="flex items-center space-x-1">
                  <Star className="w-4 h-4 text-yellow-400 fill-current" />
                  <span className="text-sm text-gray-300">{product.rating}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Products Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredProducts.map((product) => (
          <div key={product.id} className="bg-gray-800 rounded-xl overflow-hidden border border-gray-700 hover:border-gray-600 transition-all group">
            <div className={`h-48 ${product.image} flex items-center justify-center relative`}>
              <ShoppingBag className="w-12 h-12 text-white opacity-80" />
              
              {product.trending && (
                <div className="absolute top-3 left-3 px-2 py-1 bg-orange-500 text-white text-xs font-medium rounded-full flex items-center space-x-1">
                  <TrendingUp className="w-3 h-3" />
                  <span>Trending</span>
                </div>
              )}
              
              {product.aiRecommended && (
                <div className="absolute top-3 right-3 px-2 py-1 bg-purple-500 text-white text-xs font-medium rounded-full flex items-center space-x-1">
                  <Zap className="w-3 h-3" />
                  <span>AI Pick</span>
                </div>
              )}
            </div>

            <div className="p-6">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-bold text-white group-hover:text-purple-300 transition-colors">
                  {product.name}
                </h3>
                <div className="flex items-center space-x-1">
                  <Star className="w-4 h-4 text-yellow-400 fill-current" />
                  <span className="text-sm text-gray-300">{product.rating}</span>
                </div>
              </div>
              
              <p className="text-gray-400 text-sm mb-4 line-clamp-2">
                {product.description}
              </p>

              <div className="flex items-center justify-between">
                <div>
                  <span className="text-2xl font-bold text-green-400">${product.price}</span>
                  <span className="text-gray-400 text-sm ml-2">/month</span>
                </div>
                <button className="bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-600 hover:to-cyan-600 px-4 py-2 rounded-lg font-medium text-white transition-all">
                  Add to Cart
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
