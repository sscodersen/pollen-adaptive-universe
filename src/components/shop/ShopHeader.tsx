
import React from 'react';
import { RefreshCw } from 'lucide-react';

interface ShopHeaderProps {
  loading: boolean;
  onRefresh: () => void;
}

export const ShopHeader: React.FC<ShopHeaderProps> = ({ loading, onRefresh }) => {
  return (
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
          onClick={onRefresh}
          className="p-2.5 bg-gray-800/50 hover:bg-gray-700/50 rounded-lg transition-colors text-gray-400 hover:text-cyan-400 disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={loading}
          aria-label="Refresh products"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>
    </div>
  );
};
