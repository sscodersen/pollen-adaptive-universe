import React from 'react';
import { App } from '../AppStorePage';
import { Download, ExternalLink, Star, TrendingUp, Heart, Share2, Smartphone } from 'lucide-react';

interface AppCardProps {
  app: App;
}

export const AppCard: React.FC<AppCardProps> = ({ app }) => {
  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-6 hover:bg-gray-900/70 transition-all group animate-fade-in">
      {/* App Icon Placeholder */}
      <div className="w-full h-48 bg-gradient-to-br from-gray-800 to-gray-700 rounded-lg mb-4 flex items-center justify-center group-hover:from-blue-500/20 group-hover:to-purple-500/20 transition-all">
        <Smartphone className="w-16 h-16 text-gray-500 group-hover:text-blue-400 transition-colors" />
      </div>

      {/* App Info */}
      <div className="mb-4 flex flex-col h-full">
        <div className="flex items-center justify-between mb-2">
          <span className="px-2 py-1 bg-blue-500/20 text-blue-300 rounded text-xs border border-blue-500/30">
            {app.category}
          </span>
          <div className="flex items-center space-x-2">
            {app.trending && (
              <span title="Trending">
                <TrendingUp className="w-4 h-4 text-cyan-400" />
              </span>
            )}
            {app.featured && (
              <span className="px-2 py-1 bg-yellow-500/20 text-yellow-300 rounded text-xs border border-yellow-500/30">
                Featured
              </span>
            )}
            <div className={`px-2 py-1 rounded text-xs font-medium ${
              app.significance > 8 
                ? 'bg-green-500/20 text-green-300'
                : 'bg-cyan-500/20 text-cyan-300'
            }`} title={`Relevance: ${app.significance.toFixed(1)}`}>
              {app.significance.toFixed(1)}
            </div>
          </div>
        </div>

        <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-blue-300 transition-colors truncate" title={app.name}>
          {app.name}
        </h3>

        <p className="text-gray-400 text-sm mb-3 line-clamp-2" title={app.description}>
          {app.description}
        </p>

        <div className="mt-auto">
          {/* Rating and Reviews */}
          <div className="flex items-center justify-between mb-4 text-xs">
            <div className="flex items-center space-x-1">
              <Star className="w-4 h-4 text-yellow-400 fill-current" />
              <span className="text-sm text-white font-medium">{app.rating.toFixed(1)}</span>
              <span className="text-gray-400">({app.reviews.toLocaleString()})</span>
            </div>
            <span className="text-gray-500 truncate">{app.downloads} downloads</span>
          </div>

          {/* Developer */}
          <div className="mb-3">
            <span className="text-xs text-gray-400">by {app.developer}</span>
          </div>

          {/* Price */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2 items-baseline">
              <span className={`text-xl font-bold ${app.price === 'Free' ? 'text-green-400' : 'text-white'}`}>
                {app.price}
              </span>
              {app.originalPrice && (
                <span className="text-sm text-gray-400 line-through">{app.originalPrice}</span>
              )}
              {app.discount && app.discount > 0 && (
                <span className="px-2 py-1 bg-red-500/20 text-red-300 rounded text-xs font-medium">
                  -{app.discount}%
                </span>
              )}
            </div>
            <div className="text-xs text-gray-400">
              {app.size} • v{app.version}
            </div>
          </div>

          {/* Tags */}
          {app.tags && app.tags.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-4">
              {app.tags.slice(0, 3).map((tag, index) => (
                <span key={index} className="px-2 py-1 bg-gray-600/20 text-gray-400 rounded text-xs border border-gray-600/30">
                  #{tag}
                </span>
              ))}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center space-x-3">
            <a
              href={app.downloadLink}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 px-4 py-2 rounded-lg font-medium transition-all flex items-center justify-center space-x-2 text-white"
            >
              <Download className="w-4 h-4" />
              <span>Download</span>
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