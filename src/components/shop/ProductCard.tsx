
import React from 'react';
import { Product } from '../../types/shop';
import { ShoppingBag, ExternalLink, Star, TrendingUp, Heart, Share2 } from 'lucide-react';

interface ProductCardProps {
  product: Product;
}

export const ProductCard: React.FC<ProductCardProps> = ({ product }) => {
  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-4 hover:bg-gray-900/70 transition-all group animate-fade-in flex flex-col h-full">
      {/* Product Image Placeholder */}
      <div className="w-full h-32 bg-gradient-to-br from-gray-800 to-gray-700 rounded-lg mb-3 flex items-center justify-center group-hover:from-cyan-500/20 group-hover:to-purple-500/20 transition-all">
        <ShoppingBag className="w-12 h-12 text-gray-500 group-hover:text-cyan-400 transition-colors" />
      </div>

      {/* Product Info */}
      <div className="flex flex-col flex-1">
        <div className="flex items-center justify-between mb-2">
          <span className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded text-xs border border-purple-500/30">
            {product.category}
          </span>
          <div className="flex items-center space-x-1">
            {product.trending && <TrendingUp className="w-3 h-3 text-cyan-400" />}
            <div className={`px-1 py-0.5 rounded text-xs font-medium ${
              product.significance > 8 
                ? 'bg-green-500/20 text-green-300'
                : 'bg-cyan-500/20 text-cyan-300'
            }`}>
              {product.significance.toFixed(1)}
            </div>
          </div>
        </div>

        <h3 className="text-sm font-semibold text-white mb-2 group-hover:text-cyan-300 transition-colors line-clamp-2" title={product.name}>
          {product.name}
        </h3>

        <p className="text-gray-400 text-xs mb-2 line-clamp-2 flex-1" title={product.description}>
          {product.description}
        </p>

        {/* Features */}
        {product.features && product.features.length > 0 && (
          <div className="mb-2">
            <div className="flex flex-wrap gap-1">
              {product.features.slice(0, 2).map((feature, index) => (
                <span key={index} className="px-1 py-0.5 bg-gray-700/50 text-gray-300 rounded text-xs truncate">
                  {feature}
                </span>
              ))}
            </div>
          </div>
        )}
        
        <div className="mt-auto space-y-2">
          {/* Rating and Price */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1">
              <Star className="w-3 h-3 text-yellow-400 fill-current" />
              <span className="text-xs text-white font-medium">{product.rating.toFixed(1)}</span>
            </div>
            <div className="flex items-center space-x-1">
              <span className="text-sm font-bold text-white">{product.price}</span>
              {product.discount && product.discount > 0 && (
                <span className="px-1 py-0.5 bg-red-500/20 text-red-300 rounded text-xs">
                  -{product.discount}%
                </span>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center space-x-2">
            <a
              href={product.link}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 bg-gradient-to-r from-cyan-500 to-purple-500 hover:from-cyan-600 hover:to-purple-600 px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center justify-center space-x-1 text-white"
            >
              <ShoppingBag className="w-3 h-3" />
              <span>Visit</span>
            </a>
            <button className="p-1.5 bg-gray-800/50 hover:bg-gray-700/50 rounded-lg transition-colors">
              <Heart className="w-3 h-3 text-gray-400 hover:text-red-400 transition-colors" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
