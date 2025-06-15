
import React, { useState, useEffect, useCallback } from 'react';
import { ShoppingBag, Star, Heart, Share2, TrendingUp, Award, Search, ExternalLink, Zap } from 'lucide-react';
import { significanceAlgorithm } from '../services/significanceAlgorithm';
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

interface Product {
  id: string;
  name: string;
  description: string;
  price: string;
  originalPrice?: string;
  discount?: number;
  rating: number;
  reviews: number;
  category: string;
  brand: string;
  tags: string[];
  link: string;
  inStock: boolean;
  trending: boolean;
  significance: number;
  features: string[];
  seller: string;
}

export const SmartShopPage = () => {
  const [products, setProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  const productTemplates = [
    {
      name: 'AI-Powered Wireless Earbuds Pro',
      description: 'Advanced noise cancellation with AI-driven sound optimization that adapts to your environment in real-time.',
      category: 'Audio',
      brand: 'TechFlow',
      features: ['Active Noise Cancellation', 'AI Sound Optimization', '30-hour battery life'],
      price: '$199',
      originalPrice: '$299',
      discount: 33,
      tags: ['AI', 'Audio', 'Wireless', 'Premium']
    },
    {
      name: 'Smart Home Security System 2024',
      description: 'Complete home security with AI facial recognition, 24/7 monitoring, and smart alerts.',
      category: 'Smart Home',
      brand: 'SecureLife',
      features: ['AI Facial Recognition', '4K Cameras', 'Mobile App Control'],
      price: '$449',
      originalPrice: '$599',
      discount: 25,
      tags: ['Security', 'Smart Home', 'AI', 'Surveillance']
    },
    {
      name: 'Ergonomic Gaming Chair Pro',
      description: 'Professional gaming chair with memory foam, RGB lighting, and ergonomic design for 12+ hour sessions.',
      category: 'Gaming',
      brand: 'GameThrone',
      features: ['Memory Foam Cushioning', 'RGB Lighting', 'Adjustable Height'],
      price: '$329',
      discount: 0,
      tags: ['Gaming', 'Ergonomic', 'RGB', 'Comfort']
    },
    {
      name: 'Sustainable Water Bottle with UV-C',
      description: 'Self-cleaning water bottle with UV-C sterilization technology and temperature control.',
      category: 'Health',
      brand: 'EcoClean',
      features: ['UV-C Sterilization', 'Temperature Control', 'BPA-Free'],
      price: '$89',
      originalPrice: '$119',
      discount: 25,
      tags: ['Health', 'Sustainable', 'Technology', 'Eco-Friendly']
    },
    {
      name: 'Professional Drone 4K Camera',
      description: 'High-performance drone with 4K recording, obstacle avoidance, and 45-minute flight time.',
      category: 'Photography',
      brand: 'SkyVision',
      features: ['4K Recording', 'Obstacle Avoidance', '45min Flight Time'],
      price: '$899',
      originalPrice: '$1199',
      discount: 25,
      tags: ['Drone', 'Photography', 'Professional', '4K']
    },
    {
      name: 'Smart Fitness Mirror',
      description: 'Interactive fitness mirror with AI personal trainer, real-time form correction, and workout tracking.',
      category: 'Fitness',
      brand: 'FitReflect',
      features: ['AI Personal Trainer', 'Form Correction', 'Workout Tracking'],
      price: '$1299',
      originalPrice: '$1599',
      discount: 19,
      tags: ['Fitness', 'AI', 'Smart Mirror', 'Health']
    },
    {
      name: 'Wireless Charging Desk Pad',
      description: 'Premium leather desk pad with built-in wireless charging zones for multiple devices.',
      category: 'Office',
      brand: 'WorkSpace',
      features: ['Wireless Charging', 'Premium Leather', 'Multiple Device Support'],
      price: '$159',
      discount: 0,
      tags: ['Office', 'Wireless Charging', 'Premium', 'Productivity']
    },
    {
      name: 'Smart Plant Care System',
      description: 'Automated plant care with soil sensors, watering system, and mobile app monitoring.',
      category: 'Home & Garden',
      brand: 'GreenThumb',
      features: ['Automated Watering', 'Soil Sensors', 'Mobile App'],
      price: '$129',
      originalPrice: '$179',
      discount: 28,
      tags: ['Smart Home', 'Plants', 'Automation', 'Gardening']
    }
  ];

  const generateProduct = useCallback(async () => {
    const template = productTemplates[Math.floor(Math.random() * productTemplates.length)];
    
    const scored = significanceAlgorithm.scoreContent(template.description, 'shop', template.brand);
    
    const product: Product = {
      id: Date.now().toString() + Math.random(),
      name: template.name,
      description: template.description,
      price: template.price,
      originalPrice: template.originalPrice,
      discount: template.discount,
      rating: Number((Math.random() * 1.5 + 3.5).toFixed(1)),
      reviews: Math.floor(Math.random() * 5000) + 100,
      category: template.category,
      brand: template.brand,
      tags: template.tags,
      link: `https://example.com/product/${template.name.toLowerCase().replace(/\s+/g, '-')}`,
      inStock: Math.random() > 0.1,
      trending: scored.significanceScore > 7.5,
      significance: scored.significanceScore,
      features: template.features,
      seller: template.brand
    };

    return product;
  }, []);

  const loadProducts = useCallback(async () => {
    setLoading(true);
    const newProducts = await Promise.all(
      Array.from({ length: 12 }, () => generateProduct())
    );
    setProducts(newProducts.sort((a, b) => b.significance - a.significance));
    setLoading(false);
  }, [generateProduct]);

  useEffect(() => {
    loadProducts();
    const interval = setInterval(loadProducts, 60000);
    return () => clearInterval(interval);
  }, [loadProducts]);

  const filteredProducts = products.filter(product => {
    const matchesSearch = !searchQuery || 
      product.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.category.toLowerCase().includes(searchQuery.toLowerCase()) ||
      product.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    return matchesSearch;
  });

  return (
    <div className="flex-1 bg-gray-950">
      {/* Header with Search */}
      <div className="sticky top-0 z-10 bg-gray-900/95 backdrop-blur-sm border-b border-gray-800/50">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                <ShoppingBag className="w-8 h-8 text-cyan-400" />
                Smart Shop
              </h1>
              <p className="text-gray-400">AI-curated quality products • Real-time deals • Smart recommendations</p>
            </div>
            <div className="flex items-center space-x-3">
              <div className="px-4 py-2 bg-green-500/10 text-green-400 rounded-full text-sm font-medium border border-green-500/20 flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>Live Deals</span>
              </div>
            </div>
          </div>
          
          {/* Search Bar */}
          <div className="relative max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <Input
              type="text"
              placeholder="Search products, categories, brands..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 bg-gray-800/50 border-gray-700 text-white placeholder-gray-400 focus:border-cyan-500"
            />
          </div>
        </div>
      </div>

      {/* Products Grid */}
      <div className="p-6">
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {[...Array(12)].map((_, i) => (
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
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredProducts.map((product) => (
              <div key={product.id} className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-all group">
                {/* Product Image Placeholder */}
                <div className="w-full h-48 bg-gradient-to-br from-gray-800 to-gray-700 rounded-lg mb-4 flex items-center justify-center group-hover:from-cyan-500/20 group-hover:to-purple-500/20 transition-all relative">
                  <ShoppingBag className="w-16 h-16 text-gray-500 group-hover:text-cyan-400 transition-colors" />
                  {product.trending && (
                    <div className="absolute top-3 left-3 px-2 py-1 bg-red-500/20 text-red-300 rounded text-xs font-medium border border-red-500/30 flex items-center gap-1">
                      <TrendingUp className="w-3 h-3" />
                      Trending
                    </div>
                  )}
                  {product.discount && product.discount > 0 && (
                    <div className="absolute top-3 right-3 px-2 py-1 bg-green-500/20 text-green-300 rounded text-xs font-medium border border-green-500/30">
                      -{product.discount}%
                    </div>
                  )}
                </div>

                {/* Product Info */}
                <div className="flex flex-col h-full">
                  <div className="flex items-center justify-between mb-2">
                    <span className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-xs border border-purple-500/30">
                      {product.category}
                    </span>
                    <div className={`px-2 py-1 rounded text-xs font-medium ${
                      product.significance > 8 
                        ? 'bg-green-500/20 text-green-300 border border-green-500/30'
                        : 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                    }`}>
                      {product.significance.toFixed(1)} ★
                    </div>
                  </div>

                  <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-cyan-300 transition-colors line-clamp-2">
                    {product.name}
                  </h3>

                  <p className="text-gray-400 text-sm mb-3 line-clamp-2 flex-grow">
                    {product.description}
                  </p>

                  {/* Features */}
                  <div className="mb-4">
                    <div className="flex flex-wrap gap-1 mb-3">
                      {product.features.slice(0, 2).map((feature, index) => (
                        <span key={index} className="px-2 py-1 bg-gray-700/50 text-gray-300 rounded text-xs">
                          {feature}
                        </span>
                      ))}
                    </div>
                  </div>
                  
                  {/* Rating and Reviews */}
                  <div className="flex items-center space-x-3 mb-4 text-sm">
                    <div className="flex items-center space-x-1">
                      <Star className="w-4 h-4 text-yellow-400 fill-current" />
                      <span className="text-white font-medium">{product.rating}</span>
                    </div>
                    <span className="text-gray-400">({product.reviews.toLocaleString()})</span>
                  </div>

                  {/* Price */}
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-2">
                      <span className="text-xl font-bold text-white">{product.price}</span>
                      {product.originalPrice && (
                        <span className="text-sm text-gray-400 line-through">{product.originalPrice}</span>
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

                  {/* Actions */}
                  <div className="flex items-center space-x-2 mt-auto">
                    <Button 
                      className="flex-1 bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 text-white"
                      size="sm"
                    >
                      <ShoppingBag className="w-4 h-4 mr-2" />
                      Buy Now
                      <ExternalLink className="w-4 h-4 ml-2" />
                    </Button>
                    <Button variant="outline" size="sm" className="border-gray-700 text-gray-300 hover:bg-gray-800">
                      <Heart className="w-4 h-4" />
                    </Button>
                    <Button variant="outline" size="sm" className="border-gray-700 text-gray-300 hover:bg-gray-800">
                      <Share2 className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
