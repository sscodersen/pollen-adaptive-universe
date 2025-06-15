
import React from 'react';
import { Product } from '../../types';
import { ShoppingBag, ExternalLink, Star, TrendingUp, Heart, Share2 } from 'lucide-react';

interface ProductCardProps {
  product: Product;
}

export const ProductCard: React.FC<ProductCardProps> = ({ product }) => {
  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-all group animate-fade-in">
      {/* Product Image Placeholder */}
      <div className="w-full h-48 bg-gradient-to-br from-gray-800 to-gray-700 rounded-lg mb-4 flex items-center justify-center group-hover:from-cyan-500/20 group-hover:to-purple-500/20 transition-all">
        <ShoppingBag className="w-16 h-16 text-gray-500 group-hover:text-cyan-400 transition-colors" />
      </div>

      {/* Product Info */}
      <div className="mb-4 flex flex-col h-full">
        <div className="flex items-center justify-between mb-2">
          <span className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-xs border border-purple-500/30">
            {product.category}
          </span>
          <div className="flex items-center space-x-2">
            {product.trending && (
              <span title="Trending">
                <TrendingUp className="w-4 h-4 text-cyan-400" />
              </span>
            )}
            <div className={`px-2 py-1 rounded text-xs font-medium ${
              product.significance > 8 
                ? 'bg-green-500/20 text-green-300'
                : 'bg-cyan-500/20 text-cyan-300'
            }`} title={`Relevance: ${product.significance.toFixed(1)}`}>
              {product.significance.toFixed(1)}
            </div>
          </div>
        </div>

        <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-cyan-300 transition-colors truncate" title={product.name}>
          {product.name}
        </h3>

        <p className="text-gray-400 text-sm mb-3 line-clamp-2" title={product.description}>
          {product.description}
        </p>

        {/* Features */}
        {product.features && product.features.length > 0 && (
          <div className="mb-4">
            <div className="flex flex-wrap gap-1">
              {product.features.slice(0, 3).map((feature, index) => (
                <span key={index} className="px-2 py-1 bg-gray-700/50 text-gray-300 rounded text-xs">
                  {feature}
                </span>
              ))}
            </div>
          </div>
        )}
        
        <div className="mt-auto">
          {/* Rating and Reviews */}
          <div className="flex items-center space-x-4 mb-4 text-xs items-baseline">
            <div className="flex items-center space-x-1">
              <Star className="w-4 h-4 text-yellow-400 fill-current" />
              <span className="text-sm text-white font-medium">{product.rating.toFixed(1)}</span>
            </div>
            <span className="text-gray-400">({product.reviews.toLocaleString()} reviews)</span>
            <span className="text-gray-500 truncate">by {product.seller}</span>
          </div>

          {/* Price */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2 items-baseline">
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
          {product.tags && product.tags.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-4">
              {product.tags.slice(0, 3).map((tag, index) => (
                <span key={index} className="px-2 py-1 bg-gray-600/20 text-gray-400 rounded text-xs border border-gray-600/30">
                  #{tag}
                </span>
              ))}
            </div>
          )}

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
      </div>
    </div>
  );
};
