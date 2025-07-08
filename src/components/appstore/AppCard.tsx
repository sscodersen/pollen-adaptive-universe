import React from 'react';
import { App } from '../AppStorePage';
import { Download, ExternalLink, Star, TrendingUp, Heart, Share2, Smartphone } from 'lucide-react';

interface AppCardProps {
  app: App;
}

export const AppCard: React.FC<AppCardProps> = ({ app }) => {
  return (
    <div className="bg-gray-900/50 rounded-xl border border-gray-800/50 p-4 hover:bg-gray-900/70 transition-all group animate-fade-in flex flex-col h-full">
      {/* App Icon Placeholder */}
      <div className="w-full h-32 bg-gradient-to-br from-gray-800 to-gray-700 rounded-lg mb-3 flex items-center justify-center group-hover:from-blue-500/20 group-hover:to-purple-500/20 transition-all">
        <Smartphone className="w-12 h-12 text-gray-500 group-hover:text-blue-400 transition-colors" />
      </div>

      {/* App Info */}
      <div className="flex flex-col flex-1">
        <div className="flex items-center justify-between mb-2">
          <span className="px-2 py-1 bg-blue-500/20 text-blue-300 rounded text-xs border border-blue-500/30">
            {app.category}
          </span>
          <div className="flex items-center space-x-1">
            {app.trending && <TrendingUp className="w-3 h-3 text-cyan-400" />}
            {app.featured && (
              <span className="px-1 py-0.5 bg-yellow-500/20 text-yellow-300 rounded text-xs border border-yellow-500/30">
                Featured
              </span>
            )}
            <div className={`px-1 py-0.5 rounded text-xs font-medium ${
              app.significance > 8 
                ? 'bg-green-500/20 text-green-300'
                : 'bg-cyan-500/20 text-cyan-300'
            }`}>
              {app.significance.toFixed(1)}
            </div>
          </div>
        </div>

        <h3 className="text-sm font-semibold text-white mb-2 group-hover:text-blue-300 transition-colors line-clamp-2" title={app.name}>
          {app.name}
        </h3>

        <p className="text-gray-400 text-xs mb-2 line-clamp-2 flex-1" title={app.description}>
          {app.description}
        </p>

        <div className="mt-auto space-y-2">
          {/* Rating and Downloads */}
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center space-x-1">
              <Star className="w-3 h-3 text-yellow-400 fill-current" />
              <span className="text-xs text-white font-medium">{app.rating.toFixed(1)}</span>
            </div>
            <span className="text-gray-500 text-xs">{app.downloads}</span>
          </div>

          {/* Developer */}
          <div className="text-xs text-gray-400 truncate">by {app.developer}</div>

          {/* Price and Version */}
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-1">
              <span className={`text-sm font-bold ${app.price === 'Free' ? 'text-green-400' : 'text-white'}`}>
                {app.price}
              </span>
              {app.discount && app.discount > 0 && (
                <span className="px-1 py-0.5 bg-red-500/20 text-red-300 rounded text-xs">
                  -{app.discount}%
                </span>
              )}
            </div>
            <div className="text-xs text-gray-400">v{app.version}</div>
          </div>

          {/* Tags */}
          {app.tags && app.tags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {app.tags.slice(0, 2).map((tag, index) => (
                <span key={index} className="px-1 py-0.5 bg-gray-600/20 text-gray-400 rounded text-xs">
                  #{tag}
                </span>
              ))}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center space-x-2">
            <a
              href={app.downloadLink}
              target="_blank"
              rel="noopener noreferrer"
              className="flex-1 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 px-3 py-1.5 rounded-lg text-xs font-medium transition-all flex items-center justify-center space-x-1 text-white"
            >
              <Download className="w-3 h-3" />
              <span>Download</span>
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